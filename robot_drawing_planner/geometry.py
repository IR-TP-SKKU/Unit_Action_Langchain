"""Deterministic stroke geometry generation."""

from __future__ import annotations

import math

from robot_drawing_planner.schemas import ArcStroke, LineStroke, NormalizedGoal, Point2D, Stroke


def generate_strokes(goal: NormalizedGoal) -> list[Stroke]:
    """Generate deterministic board-frame strokes for a validated goal."""

    if goal.shape_type == "circle":
        return _circle(goal)
    if goal.shape_type == "square":
        return _square(goal)
    if goal.shape_type == "triangle":
        return _triangle(goal)
    if goal.shape_type == "letter":
        return _letter(goal)
    raise ValueError(f"Unsupported drawing shape: {goal.shape_type}")


def stroke_start_point(stroke: Stroke) -> Point2D:
    """Return the first point of a line or arc stroke."""

    if isinstance(stroke, LineStroke):
        return stroke.start
    return _arc_point(stroke, stroke.start_angle_rad)


def stroke_end_point(stroke: Stroke) -> Point2D:
    """Return the last point of a line or arc stroke."""

    if isinstance(stroke, LineStroke):
        return stroke.end
    if math.isclose(
        abs(stroke.end_angle_rad - stroke.start_angle_rad),
        2.0 * math.pi,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        return stroke_start_point(stroke)
    return _arc_point(stroke, stroke.end_angle_rad)


def _circle(goal: NormalizedGoal) -> list[Stroke]:
    if goal.radius_m is None:
        raise ValueError("Circle goals require radius_m.")
    return [
        ArcStroke(
            stroke_id="stroke_001",
            center=goal.center,
            radius_m=goal.radius_m,
            start_angle_rad=goal.orientation_rad,
            end_angle_rad=goal.orientation_rad + 2.0 * math.pi,
            direction="ccw",
        )
    ]


def _square(goal: NormalizedGoal) -> list[Stroke]:
    if goal.side_length_m is None:
        raise ValueError("Square goals require side_length_m.")
    half = goal.side_length_m / 2.0
    points = [
        _rotated_point(goal.center, -half, half, goal.orientation_rad),
        _rotated_point(goal.center, half, half, goal.orientation_rad),
        _rotated_point(goal.center, half, -half, goal.orientation_rad),
        _rotated_point(goal.center, -half, -half, goal.orientation_rad),
    ]
    return _closed_polyline(points)


def _triangle(goal: NormalizedGoal) -> list[Stroke]:
    if goal.side_length_m is None:
        raise ValueError("Triangle goals require side_length_m.")
    side = goal.side_length_m
    height = math.sqrt(3.0) * side / 2.0
    points = [
        _rotated_point(goal.center, 0.0, 2.0 * height / 3.0, goal.orientation_rad),
        _rotated_point(goal.center, side / 2.0, -height / 3.0, goal.orientation_rad),
        _rotated_point(goal.center, -side / 2.0, -height / 3.0, goal.orientation_rad),
    ]
    return _closed_polyline(points)


def _letter(goal: NormalizedGoal) -> list[Stroke]:
    letter = (goal.letter or "").upper()
    if letter == "A":
        return _letter_a(goal)
    if letter == "H":
        return _letter_h(goal)
    if letter == "L":
        return _letter_l(goal)
    if letter == "T":
        return _letter_t(goal)
    if letter == "O":
        return _circle(goal)
    raise ValueError(f"Unsupported letter: {goal.letter}")


def _letter_box(goal: NormalizedGoal) -> tuple[float, float, float, float]:
    if goal.size_m is None:
        raise ValueError("Letter goals require size_m.")
    height = goal.size_m
    width = height * 0.60
    left = -width / 2.0
    right = width / 2.0
    bottom = -height / 2.0
    top = height / 2.0
    return left, right, bottom, top


def _letter_point(goal: NormalizedGoal, x_offset: float, y_offset: float) -> Point2D:
    return _rotated_point(goal.center, x_offset, y_offset, goal.orientation_rad)


def _letter_a(goal: NormalizedGoal) -> list[Stroke]:
    left, right, bottom, top = _letter_box(goal)
    width = right - left
    return [
        _line("stroke_001", _letter_point(goal, left, bottom), _letter_point(goal, 0.0, top)),
        _line("stroke_002", _letter_point(goal, 0.0, top), _letter_point(goal, right, bottom)),
        _line(
            "stroke_003",
            _letter_point(goal, -width * 0.20, 0.0),
            _letter_point(goal, width * 0.20, 0.0),
        ),
    ]


def _letter_h(goal: NormalizedGoal) -> list[Stroke]:
    left, right, bottom, top = _letter_box(goal)
    return [
        _line("stroke_001", _letter_point(goal, left, bottom), _letter_point(goal, left, top)),
        _line("stroke_002", _letter_point(goal, right, bottom), _letter_point(goal, right, top)),
        _line("stroke_003", _letter_point(goal, left, 0.0), _letter_point(goal, right, 0.0)),
    ]


def _letter_l(goal: NormalizedGoal) -> list[Stroke]:
    left, right, bottom, top = _letter_box(goal)
    return [
        _line("stroke_001", _letter_point(goal, left, top), _letter_point(goal, left, bottom)),
        _line("stroke_002", _letter_point(goal, left, bottom), _letter_point(goal, right, bottom)),
    ]


def _letter_t(goal: NormalizedGoal) -> list[Stroke]:
    left, right, bottom, top = _letter_box(goal)
    return [
        _line("stroke_001", _letter_point(goal, left, top), _letter_point(goal, right, top)),
        _line("stroke_002", _letter_point(goal, 0.0, top), _letter_point(goal, 0.0, bottom)),
    ]


def _closed_polyline(points: list[Point2D]) -> list[Stroke]:
    strokes: list[Stroke] = []
    for index, start in enumerate(points):
        end = points[(index + 1) % len(points)]
        strokes.append(_line(f"stroke_{index + 1:03d}", start, end))
    return strokes


def _line(stroke_id: str, start: Point2D, end: Point2D) -> LineStroke:
    return LineStroke(stroke_id=stroke_id, start=start, end=end)


def _rotated_point(center: Point2D, x_offset: float, y_offset: float, angle_rad: float) -> Point2D:
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    return Point2D(
        x=center.x + x_offset * cos_a - y_offset * sin_a,
        y=center.y + x_offset * sin_a + y_offset * cos_a,
    )


def _arc_point(stroke: ArcStroke, angle_rad: float) -> Point2D:
    return Point2D(
        x=stroke.center.x + stroke.radius_m * math.cos(angle_rad),
        y=stroke.center.y + stroke.radius_m * math.sin(angle_rad),
    )

