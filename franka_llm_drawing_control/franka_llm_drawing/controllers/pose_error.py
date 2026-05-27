"""Pose error utilities for Cartesian control."""

from __future__ import annotations

import numpy as np


def orientation_error_axis_angle(R_des: np.ndarray, R_cur: np.ndarray) -> np.ndarray:
    """Return a 3D log-map orientation error vector.

    The error is computed from ``R_err = R_des @ R_cur.T`` and is suitable as a
    small-angle angular correction term in the parent frame.
    """

    R_d = _rotation(R_des, "R_des")
    R_c = _rotation(R_cur, "R_cur")
    R_err = R_d @ R_c.T
    skew_vec = np.array(
        [
            R_err[2, 1] - R_err[1, 2],
            R_err[0, 2] - R_err[2, 0],
            R_err[1, 0] - R_err[0, 1],
        ]
    )
    cos_theta = float(np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0))
    theta = float(np.arccos(cos_theta))
    if theta < 1e-9:
        return 0.5 * skew_vec
    sin_theta = float(np.sin(theta))
    if abs(sin_theta) < 1e-9:
        axis = _axis_for_pi_rotation(R_err)
        return theta * axis
    return theta / (2.0 * sin_theta) * skew_vec


def pose_error_6d(
    p_des: np.ndarray,
    R_des: np.ndarray,
    p_cur: np.ndarray,
    R_cur: np.ndarray,
    position_weight: float = 1.0,
    orientation_weight: float = 1.0,
) -> np.ndarray:
    """Return weighted ``[position_error, orientation_error]`` with shape ``(6,)``."""

    p_d = _vector3(p_des, "p_des")
    p_c = _vector3(p_cur, "p_cur")
    pos_err = float(position_weight) * (p_d - p_c)
    ori_err = float(orientation_weight) * orientation_error_axis_angle(R_des, R_cur)
    return np.concatenate([pos_err, ori_err])


def _vector3(value: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {arr.shape}.")
    return arr


def _rotation(value: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.shape != (3, 3):
        raise ValueError(f"{name} must have shape (3, 3), got {arr.shape}.")
    return arr


def _axis_for_pi_rotation(R_err: np.ndarray) -> np.ndarray:
    diagonal = np.diag(R_err)
    axis = np.zeros(3)
    index = int(np.argmax(diagonal))
    axis[index] = np.sqrt(max(diagonal[index] + 1.0, 0.0) / 2.0)
    if axis[index] < 1e-9:
        return np.array([1.0, 0.0, 0.0])
    for j in range(3):
        if j != index:
            axis[j] = R_err[j, index] / (2.0 * axis[index])
    norm = float(np.linalg.norm(axis))
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0])
    return axis / norm
