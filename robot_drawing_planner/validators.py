"""Validation helpers for planner inputs and generated strokes."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from robot_drawing_planner.geometry import stroke_start_point
from robot_drawing_planner.schemas import (
    ArcStroke,
    Board,
    LineStroke,
    Measurement,
    NormalizedGoal,
    ParsedGoal,
    Point2D,
    Stroke,
)
from robot_drawing_planner.units import measurement_to_meters

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


def normalize_goal(parsed: ParsedGoal) -> NormalizedGoal:
    """Validate support and convert a ParsedGoal to board-frame meters."""

    center = parsed.center
    assumptions: list[str] = []
    warnings: list[str] = []
    if center is None:
        center = Point2D(x=0.0, y=0.0)
        assumptions.append("center defaulted to board origin (0, 0) meters")
    if parsed.position_hint:
        assumptions.append(f"position_hint preserved: {parsed.position_hint}")

    radius_m = _measurement_to_meters(parsed.radius)
    side_length_m = _measurement_to_meters(parsed.side_length)
    size_m = _measurement_to_meters(parsed.size)
    letter = parsed.letter

    if parsed.shape_type == "circle":
        if radius_m is None:
            if size_m is None:
                raise PlannerValidationError("Circle goals require radius or size.")
            radius_m = size_m / 2.0
            assumptions.append("circle size interpreted as diameter")
        elif size_m is None:
            size_m = radius_m * 2.0
            assumptions.append("circle size inferred as diameter from radius")
    elif parsed.shape_type in {"square", "triangle"}:
        if side_length_m is None:
            if size_m is None:
                raise PlannerValidationError(
                    f"{parsed.shape_type.capitalize()} goals require side_length or size."
                )
            side_length_m = size_m
            assumptions.append(f"{parsed.shape_type} size interpreted as side length")
        elif size_m is None:
            size_m = side_length_m
            assumptions.append(f"{parsed.shape_type} size inferred from side length")
    elif parsed.shape_type == "letter":
        if not letter:
            raise PlannerValidationError("Letter goals require a letter value.")
        letter = letter.upper()
        if letter not in SUPPORTED_LETTERS:
            raise PlannerValidationError(
                f"Unsupported letter '{letter}'. Supported letters: "
                f"{', '.join(sorted(SUPPORTED_LETTERS))}."
            )
        if size_m is None:
            raise PlannerValidationError("Letter goals require size.")
        if letter == "O":
            radius_m = size_m / 2.0

    if letter is not None and parsed.shape_type != "letter":
        warnings.append("letter ignored because shape_type is not letter")
        letter = None

    return NormalizedGoal(
        shape_type=parsed.shape_type,
        center=center,
        radius_m=radius_m,
        side_length_m=side_length_m,
        size_m=size_m,
        orientation_rad=math.radians(parsed.orientation_deg),
        letter=letter,
        assumptions=assumptions,
        warnings=warnings,
    )


def validate_strokes_within_board(strokes: list[Stroke], board: Board) -> None:
    """Check that sampled stroke geometry stays inside board boundaries."""

    for stroke in strokes:
        for point in _sample_stroke_points(stroke):
            if not point_is_inside_board(point, board):
                raise PlannerValidationError(
                    "Drawing geometry exceeds board boundaries "
                    f"({board.width_m}m x {board.height_m}m, frame=board)."
                )


def point_is_inside_board(point: Point2D, board: Board, tolerance: float = 1e-9) -> bool:
    """Return whether a point lies within the known drawing board."""

    half_width = board.width_m / 2.0
    half_height = board.height_m / 2.0
    return (
        -half_width - tolerance <= point.x <= half_width + tolerance
        and -half_height - tolerance <= point.y <= half_height + tolerance
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


def _measurement_to_meters(measurement: Measurement | None) -> float | None:
    if measurement is None:
        return None
    try:
        return measurement_to_meters(measurement)
    except ValueError as exc:
        raise PlannerValidationError(str(exc)) from exc


def _sample_stroke_points(stroke: Stroke) -> list[Point2D]:
    if isinstance(stroke, LineStroke):
        return [stroke.start, stroke.end]
    if isinstance(stroke, ArcStroke):
        points = []
        steps = 96
        span = stroke.end_angle_rad - stroke.start_angle_rad
        if stroke.direction == "cw" and span > 0:
            span -= 2.0 * math.pi
        for index in range(steps + 1):
            angle = stroke.start_angle_rad + span * (index / steps)
            points.append(
                Point2D(
                    x=stroke.center.x + stroke.radius_m * math.cos(angle),
                    y=stroke.center.y + stroke.radius_m * math.sin(angle),
                )
            )
        points.append(stroke_start_point(stroke))
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
