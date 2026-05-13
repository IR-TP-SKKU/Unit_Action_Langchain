"""Validation helpers for planner inputs and generated strokes."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from robot_drawing_planner.geometry import ArcStroke, LineStroke, Stroke
from robot_drawing_planner.schemas import Board, DrawingGoal, ParsedGoal, Point2D
from robot_drawing_planner.units import coordinate_to_meters, to_meters

SUPPORTED_SHAPES = {"circle", "square", "triangle", "letter"}
SUPPORTED_LETTERS = {"A", "H", "L", "T", "O"}
LOW_LEVEL_ROBOT_FIELDS = {
    "joint_angle",
    "joint_angles",
    "joint_positions",
    "ik",
    "fk",
    "jacobian",
    "isaac_command",
    "isaac_sim_command",
    "trajectory",
    "workspace_feasible",
    "end_effector_pose",
}


class PlannerValidationError(ValueError):
    """Raised when a drawing goal is outside this module's scope."""


def normalize_goal(parsed: ParsedGoal) -> DrawingGoal:
    """Validate support and convert a ParsedGoal to board-frame meters."""

    kind = parsed.object_type.strip().lower()
    letter = parsed.letter
    if len(kind) == 1 and kind.upper() in SUPPORTED_LETTERS:
        letter = kind.upper()
        kind = "letter"

    if kind not in SUPPORTED_SHAPES:
        raise PlannerValidationError(
            f"Unsupported shape '{parsed.object_type}'. Supported shapes: "
            f"{', '.join(sorted(SUPPORTED_SHAPES))}."
        )

    if kind == "letter":
        if not letter:
            raise PlannerValidationError("Letter goals require a letter value.")
        letter = letter.upper()
        if letter not in SUPPORTED_LETTERS:
            raise PlannerValidationError(
                f"Unsupported letter '{letter}'. Supported letters: "
                f"{', '.join(sorted(SUPPORTED_LETTERS))}."
            )
    elif letter is not None:
        raise PlannerValidationError("letter must be omitted unless object_type is letter.")

    try:
        size_m = to_meters(parsed.size, parsed.unit)
        center = Point2D(
            x_m=coordinate_to_meters(parsed.center_x, parsed.center_unit),
            y_m=coordinate_to_meters(parsed.center_y, parsed.center_unit),
        )
    except ValueError as exc:
        raise PlannerValidationError(str(exc)) from exc

    return DrawingGoal(kind=kind, letter=letter, size_m=size_m, center=center)


def validate_strokes_within_board(strokes: list[Stroke], board: Board) -> None:
    """Check that sampled stroke geometry stays inside board boundaries."""

    for stroke in strokes:
        for point in _sample_stroke_points(stroke):
            if not point_is_inside_board(point, board):
                raise PlannerValidationError(
                    "Drawing geometry exceeds board boundaries "
                    f"({board.width_m}m x {board.height_m}m, origin=center)."
                )


def point_is_inside_board(point: Point2D, board: Board, tolerance: float = 1e-9) -> bool:
    """Return whether a point lies within the known drawing board."""

    half_width = board.width_m / 2.0
    half_height = board.height_m / 2.0
    return (
        -half_width - tolerance <= point.x_m <= half_width + tolerance
        and -half_height - tolerance <= point.y_m <= half_height + tolerance
    )


def validate_no_low_level_robot_fields(payload: Any) -> None:
    """Reject IK, FK, Jacobian, Isaac Sim, or trajectory-like fields."""

    for key, value in _walk_mapping(payload):
        if key in LOW_LEVEL_ROBOT_FIELDS:
            raise PlannerValidationError(
                f"Low-level robot field '{key}' is outside planner scope."
            )
        if isinstance(value, str) and value.strip().lower() in LOW_LEVEL_ROBOT_FIELDS:
            raise PlannerValidationError(
                f"Low-level robot value '{value}' is outside planner scope."
            )


def _sample_stroke_points(stroke: Stroke) -> list[Point2D]:
    if isinstance(stroke, LineStroke):
        return [stroke.start, stroke.end]
    if isinstance(stroke, ArcStroke):
        points = []
        steps = 96
        span = stroke.end_angle_rad - stroke.start_angle_rad
        if stroke.clockwise and span > 0:
            span -= 2.0 * math.pi
        for index in range(steps + 1):
            angle = stroke.start_angle_rad + span * (index / steps)
            points.append(stroke.point_at(angle))
        return points
    raise TypeError(f"Unknown stroke type: {type(stroke).__name__}")


def _walk_mapping(payload: Any) -> list[tuple[str, Any]]:
    items: list[tuple[str, Any]] = []
    if isinstance(payload, Mapping):
        for raw_key, value in payload.items():
            key = str(raw_key).strip().lower()
            items.append((key, value))
            items.extend(_walk_mapping(value))
    elif isinstance(payload, list):
        for value in payload:
            items.extend(_walk_mapping(value))
    return items

