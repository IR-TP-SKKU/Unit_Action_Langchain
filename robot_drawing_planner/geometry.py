"""Deterministic stroke geometry generation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from robot_drawing_planner.schemas import DrawingGoal, Point2D


@dataclass(frozen=True)
class LineStroke:
    """A straight board-plane stroke."""

    start: Point2D
    end: Point2D
    kind: Literal["line"] = "line"

    def start_point(self) -> Point2D:
        return self.start

    def end_point(self) -> Point2D:
        return self.end


@dataclass(frozen=True)
class ArcStroke:
    """A circular board-plane arc stroke."""

    center: Point2D
    radius_m: float
    start_angle_rad: float
    end_angle_rad: float
    clockwise: bool = False
    kind: Literal["arc"] = "arc"

    def point_at(self, angle_rad: float) -> Point2D:
        return Point2D(
            x_m=self.center.x_m + self.radius_m * math.cos(angle_rad),
            y_m=self.center.y_m + self.radius_m * math.sin(angle_rad),
        )

    def start_point(self) -> Point2D:
        return self.point_at(self.start_angle_rad)

    def end_point(self) -> Point2D:
        if math.isclose(
            abs(self.end_angle_rad - self.start_angle_rad),
            2.0 * math.pi,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            return self.start_point()
        return self.point_at(self.end_angle_rad)


Stroke = LineStroke | ArcStroke


def generate_strokes(goal: DrawingGoal) -> list[Stroke]:
    """Generate deterministic board-frame strokes for a validated goal."""

    if goal.kind == "circle":
        return _circle(goal)
    if goal.kind == "square":
        return _square(goal)
    if goal.kind == "triangle":
        return _triangle(goal)
    if goal.kind == "letter":
        return _letter(goal)
    raise ValueError(f"Unsupported drawing kind: {goal.kind}")


def _circle(goal: DrawingGoal) -> list[Stroke]:
    return [
        ArcStroke(
            center=goal.center,
            radius_m=goal.size_m / 2.0,
            start_angle_rad=0.0,
            end_angle_rad=2.0 * math.pi,
            clockwise=False,
        )
    ]


def _square(goal: DrawingGoal) -> list[Stroke]:
    half = goal.size_m / 2.0
    cx, cy = goal.center.x_m, goal.center.y_m
    top_left = Point2D(x_m=cx - half, y_m=cy + half)
    top_right = Point2D(x_m=cx + half, y_m=cy + half)
    bottom_right = Point2D(x_m=cx + half, y_m=cy - half)
    bottom_left = Point2D(x_m=cx - half, y_m=cy - half)
    return [
        LineStroke(top_left, top_right),
        LineStroke(top_right, bottom_right),
        LineStroke(bottom_right, bottom_left),
        LineStroke(bottom_left, top_left),
    ]


def _triangle(goal: DrawingGoal) -> list[Stroke]:
    side = goal.size_m
    height = math.sqrt(3.0) * side / 2.0
    cx, cy = goal.center.x_m, goal.center.y_m
    top = Point2D(x_m=cx, y_m=cy + (2.0 * height / 3.0))
    bottom_right = Point2D(x_m=cx + side / 2.0, y_m=cy - (height / 3.0))
    bottom_left = Point2D(x_m=cx - side / 2.0, y_m=cy - (height / 3.0))
    return [
        LineStroke(top, bottom_right),
        LineStroke(bottom_right, bottom_left),
        LineStroke(bottom_left, top),
    ]


def _letter(goal: DrawingGoal) -> list[Stroke]:
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


def _letter_box(goal: DrawingGoal) -> tuple[float, float, float, float]:
    height = goal.size_m
    width = height * 0.60
    left = goal.center.x_m - width / 2.0
    right = goal.center.x_m + width / 2.0
    bottom = goal.center.y_m - height / 2.0
    top = goal.center.y_m + height / 2.0
    return left, right, bottom, top


def _letter_a(goal: DrawingGoal) -> list[Stroke]:
    left, right, bottom, top = _letter_box(goal)
    cx = goal.center.x_m
    mid_y = goal.center.y_m
    width = right - left
    return [
        LineStroke(Point2D(x_m=left, y_m=bottom), Point2D(x_m=cx, y_m=top)),
        LineStroke(Point2D(x_m=cx, y_m=top), Point2D(x_m=right, y_m=bottom)),
        LineStroke(
            Point2D(x_m=cx - width * 0.20, y_m=mid_y),
            Point2D(x_m=cx + width * 0.20, y_m=mid_y),
        ),
    ]


def _letter_h(goal: DrawingGoal) -> list[Stroke]:
    left, right, bottom, top = _letter_box(goal)
    mid_y = goal.center.y_m
    return [
        LineStroke(Point2D(x_m=left, y_m=bottom), Point2D(x_m=left, y_m=top)),
        LineStroke(Point2D(x_m=right, y_m=bottom), Point2D(x_m=right, y_m=top)),
        LineStroke(Point2D(x_m=left, y_m=mid_y), Point2D(x_m=right, y_m=mid_y)),
    ]


def _letter_l(goal: DrawingGoal) -> list[Stroke]:
    left, right, bottom, top = _letter_box(goal)
    return [
        LineStroke(Point2D(x_m=left, y_m=top), Point2D(x_m=left, y_m=bottom)),
        LineStroke(Point2D(x_m=left, y_m=bottom), Point2D(x_m=right, y_m=bottom)),
    ]


def _letter_t(goal: DrawingGoal) -> list[Stroke]:
    left, right, bottom, top = _letter_box(goal)
    cx = goal.center.x_m
    return [
        LineStroke(Point2D(x_m=left, y_m=top), Point2D(x_m=right, y_m=top)),
        LineStroke(Point2D(x_m=cx, y_m=top), Point2D(x_m=cx, y_m=bottom)),
    ]
