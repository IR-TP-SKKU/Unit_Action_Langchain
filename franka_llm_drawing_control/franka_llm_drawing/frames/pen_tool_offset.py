"""Pen-tip to end-effector offset compensation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np

from franka_llm_drawing.frames.transforms import invert_transform, make_transform, transform_pose
from franka_llm_drawing.trajectory.path_primitives import PoseSample


@dataclass(frozen=True)
class CartesianTrajectoryPoint:
    """Base-frame trajectory point with both tip and end-effector poses."""

    t: float
    p_base_tip: np.ndarray
    R_base_tip: np.ndarray
    T_base_tip: np.ndarray
    T_base_ee: np.ndarray
    pen_state: Literal["up", "down"]
    source_action_name: str | None
    stroke_id: str | None = None
    pen_contact_desired: bool = False
    desired_normal_force_n: float | None = None


def tip_pose_to_ee_pose(T_base_tip: np.ndarray, T_ee_tip: np.ndarray) -> np.ndarray:
    """Compute ``T_base_ee = T_base_tip @ inverse(T_ee_tip)``."""

    return transform_pose(T_base_tip, invert_transform(T_ee_tip))


def ee_pose_to_tip_pose(T_base_ee: np.ndarray, T_ee_tip: np.ndarray) -> np.ndarray:
    """Compute ``T_base_tip = T_base_ee @ T_ee_tip``."""

    return transform_pose(T_base_ee, T_ee_tip)


def samples_to_cartesian_trajectory(
    samples_board: Iterable[PoseSample],
    T_base_board: np.ndarray,
    T_ee_tip: np.ndarray,
) -> list[CartesianTrajectoryPoint]:
    """Convert board-frame pen-tip samples into base-frame tip and EE poses."""

    output: list[CartesianTrajectoryPoint] = []
    for sample in samples_board:
        T_board_tip = make_transform(sample.rotation, sample.position)
        T_base_tip = transform_pose(T_base_board, T_board_tip)
        T_base_ee = tip_pose_to_ee_pose(T_base_tip, T_ee_tip)
        output.append(
            CartesianTrajectoryPoint(
                t=sample.t,
                p_base_tip=T_base_tip[:3, 3].copy(),
                R_base_tip=T_base_tip[:3, :3].copy(),
                T_base_tip=T_base_tip,
                T_base_ee=T_base_ee,
                pen_state="down" if sample.pen_contact_desired else "up",
                source_action_name=sample.source_action,
                stroke_id=sample.stroke_id,
                pen_contact_desired=sample.pen_contact_desired,
                desired_normal_force_n=sample.desired_normal_force_n,
            )
        )
    return output
