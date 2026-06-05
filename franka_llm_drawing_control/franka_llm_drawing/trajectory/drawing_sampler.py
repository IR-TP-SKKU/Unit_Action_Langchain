"""Convert LLM DrawingPlan actions into board-frame pose samples."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

import numpy as np

from franka_llm_drawing.llm_bridge.plan_loader import drawing_plan_from_obj
from franka_llm_drawing.llm_bridge.plan_schema import DrawingPlan, PrimitiveAction
from franka_llm_drawing.llm_bridge.primitive_parser import (
    get_float,
    get_point,
    get_str,
    optional_point,
)
from franka_llm_drawing.trajectory.path_primitives import PoseSample, sample_arc, sample_line


def sample_drawing_plan(
    plan: DrawingPlan | Mapping[str, Any],
    dt: float,
    default_speed_m_s: float,
    hover_height_m: float,
    draw_height_m: float,
    default_normal_force_n: float | None = None,
) -> list[PoseSample]:
    """Sample a DrawingPlan into board-frame desired pen-tip poses.

    Valid action order is enforced. The first motion must establish a current
    position with ``move_to_start`` before pen contact or drawing actions.
    """

    drawing_plan = drawing_plan_from_obj(plan)
    if not drawing_plan.actions:
        raise ValueError("DrawingPlan contains no actions.")

    samples: list[PoseSample] = []
    current_position: np.ndarray | None = None
    pen_state = "up"
    rotation = np.eye(3)

    for action in drawing_plan.actions:
        if action.frame != "board":
            raise ValueError(f"Unsupported action frame {action.frame!r}; expected 'board'.")

        start_time = samples[-1].t if samples else 0.0
        new_samples: list[PoseSample]

        if action.name == "move_to_start":
            _require_pen_state(pen_state, "up", action)
            target = get_point(action.params, "target", default_z=hover_height_m)
            target[2] = float(action.params.get("hover_height_m", target[2]))
            if current_position is None:
                new_samples = [
                    PoseSample(
                        t=start_time,
                        position=target,
                        rotation=rotation.copy(),
                        linear_velocity=np.zeros(3),
                        linear_acceleration=np.zeros(3),
                        pen_contact_desired=False,
                        source_action=action.name,
                        stroke_id=action.stroke_id,
                    )
                ]
            else:
                new_samples = sample_line(
                    current_position,
                    target,
                    get_float(action.params, "speed_m_s", default_speed_m_s),
                    dt,
                    start_time=start_time,
                    rotation=rotation,
                    pen_contact_desired=False,
                    source_action=action.name,
                    stroke_id=action.stroke_id,
                )
            current_position = target.copy()

        elif action.name == "align_pen_orientation":
            # The final end-effector convention belongs to the simulator backend.
            # Offline samples carry a fixed board-frame tip rotation placeholder.
            rotation = np.eye(3)
            continue

        elif action.name == "pen_down":
            _require_current(current_position, action)
            _require_pen_state(pen_state, "up", action)
            target = optional_point(action.params, "target", default_z=draw_height_m)
            if target is None:
                target = np.array([current_position[0], current_position[1], draw_height_m])
            target[2] = float(target[2])
            new_samples = sample_line(
                current_position,
                target,
                get_float(action.params, "speed_m_s", default_speed_m_s),
                dt,
                start_time=start_time,
                rotation=rotation,
                pen_contact_desired=False,
                source_action=action.name,
                stroke_id=action.stroke_id,
            )
            if new_samples:
                new_samples[-1] = replace(
                    new_samples[-1],
                    pen_contact_desired=True,
                    desired_normal_force_n=default_normal_force_n,
                )
            current_position = target.copy()
            pen_state = "down"

        elif action.name in {"draw_line", "draw_line_to"}:
            _require_current(current_position, action)
            _require_pen_state(pen_state, "down", action)
            if action.name == "draw_line":
                explicit_start = optional_point(action.params, "start", default_z=draw_height_m)
                if explicit_start is not None and not np.allclose(
                    explicit_start,
                    current_position,
                    atol=1e-6,
                ):
                    raise ValueError(
                        "draw_line start does not match current sampler position."
                    )
                target = get_point(action.params, "end", default_z=draw_height_m)
            else:
                target = get_point(action.params, "target", default_z=draw_height_m)
            target[2] = float(target[2])
            new_samples = sample_line(
                current_position,
                target,
                get_float(action.params, "speed_m_s", default_speed_m_s),
                dt,
                start_time=start_time,
                rotation=rotation,
                pen_contact_desired=True,
                desired_normal_force_n=default_normal_force_n,
                source_action=action.name,
                stroke_id=action.stroke_id,
            )
            current_position = target.copy()

        elif action.name == "draw_arc":
            _require_current(current_position, action)
            _require_pen_state(pen_state, "down", action)
            center = get_point(action.params, "center", default_z=draw_height_m)
            radius = get_float(action.params, "radius_m")
            theta0 = get_float(action.params, "start_angle_rad")
            theta1 = get_float(action.params, "end_angle_rad")
            direction = get_str(action.params, "direction")
            expected_start = center + np.array(
                [radius * np.cos(theta0), radius * np.sin(theta0), 0.0]
            )
            if not np.allclose(expected_start, current_position, atol=1e-5):
                raise ValueError("draw_arc start point does not match current sampler position.")
            new_samples = sample_arc(
                center,
                radius,
                theta0,
                theta1,
                direction,
                get_float(action.params, "speed_m_s", default_speed_m_s),
                dt,
                start_time=start_time,
                rotation=rotation,
                pen_contact_desired=True,
                desired_normal_force_n=default_normal_force_n,
                source_action=action.name,
                stroke_id=action.stroke_id,
            )
            current_position = new_samples[-1].position.copy()

        elif action.name == "pen_up":
            _require_current(current_position, action)
            _require_pen_state(pen_state, "down", action)
            lift_height = get_float(action.params, "lift_height_m", hover_height_m)
            target = np.array([current_position[0], current_position[1], lift_height])
            new_samples = sample_line(
                current_position,
                target,
                get_float(action.params, "speed_m_s", default_speed_m_s),
                dt,
                start_time=start_time,
                rotation=rotation,
                pen_contact_desired=False,
                source_action=action.name,
                stroke_id=action.stroke_id,
            )
            if new_samples:
                new_samples[0] = replace(
                    new_samples[0],
                    pen_contact_desired=True,
                    desired_normal_force_n=default_normal_force_n,
                )
                new_samples[-1] = replace(
                    new_samples[-1],
                    pen_contact_desired=False,
                    desired_normal_force_n=None,
                )
            current_position = target
            pen_state = "up"

        else:
            raise ValueError(f"Unsupported action name: {action.name!r}.")

        _append_without_duplicate_time(samples, new_samples)

    return samples


def _append_without_duplicate_time(
    samples: list[PoseSample],
    new_samples: list[PoseSample],
) -> None:
    # Preserve segment-boundary samples because their source action and contact
    # flags are useful diagnostics even when the timestamp equals the previous
    # segment endpoint.
    samples.extend(new_samples)


def _require_current(current_position: np.ndarray | None, action: PrimitiveAction) -> None:
    if current_position is None:
        raise ValueError(f"{action.name} requires a current position; call move_to_start first.")


def _require_pen_state(actual: str, expected: str, action: PrimitiveAction) -> None:
    if actual != expected:
        raise ValueError(f"{action.name} requires pen_state == {expected!r}.")
