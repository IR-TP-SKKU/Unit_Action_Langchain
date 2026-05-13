"""Stateful symbolic plan builder for agentic unit-action planning."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

from robot_drawing_planner.schemas import (
    ArcStroke,
    Direction,
    LineStroke,
    Point2D,
    PrimitiveAction,
    Stroke,
)

PenState = Literal["up", "down"]


@dataclass
class PlanBuilder:
    """Mutable state that LangChain unit-action tools append to.

    This builder records symbolic primitive actions only. It does not compute
    IK, FK, Jacobians, joint commands, trajectory samples, or Isaac Sim commands.
    """

    source_command: str
    current_position: Point2D | None = None
    pen_state: PenState = "up"
    actions: list[PrimitiveAction] = field(default_factory=list)
    strokes: list[Stroke] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    finished: bool = False
    frame: Literal["board"] = "board"

    @classmethod
    def begin_plan(cls, source_command: str) -> "PlanBuilder":
        """Create a fresh plan-builder state for a user command."""

        return cls(source_command=source_command)

    def move_to_start(self, x: float, y: float) -> dict[str, Any]:
        """Append a symbolic move_to_start primitive if the pen is up."""

        if self.pen_state != "up":
            return self._record_error("move_to_start requires pen_state == 'up'.")
        target = Point2D(x=x, y=y)
        self.current_position = target
        self.actions.append(
            PrimitiveAction(
                name="move_to_start",
                frame=self.frame,
                stroke_id=None,
                params={
                    "target": target.model_dump(mode="json"),
                    "note": "symbolic board-frame move only; no robot motion is computed",
                },
            )
        )
        return self.check_plan()

    def align_pen_orientation(self) -> dict[str, Any]:
        """Append a symbolic board-normal orientation primitive."""

        self.actions.append(
            PrimitiveAction(
                name="align_pen_orientation",
                frame=self.frame,
                stroke_id=None,
                params={
                    "mode": "normal_to_board",
                    "board_normal_axis": "+z",
                    "pen_axis_target": "-z",
                    "note": "orientation constraint only; no IK computed here",
                },
            )
        )
        return self.check_plan()

    def pen_down(self) -> dict[str, Any]:
        """Append a pen_down primitive at the current board position."""

        if self.current_position is None:
            return self._record_error("pen_down requires current_position is not None.")
        if self.pen_state == "down":
            return self._record_error("pen_down requires pen_state == 'up'.")
        self.pen_state = "down"
        self.actions.append(
            PrimitiveAction(
                name="pen_down",
                frame=self.frame,
                stroke_id=None,
                params={
                    "target": self.current_position.model_dump(mode="json"),
                    "note": "symbolic contact request only; no force control is computed",
                },
            )
        )
        return self.check_plan()

    def draw_line_to(self, x: float, y: float) -> dict[str, Any]:
        """Append a draw_line primitive from current_position to the given point."""

        if self.pen_state != "down":
            return self._record_error("draw_line_to requires pen_state == 'down'.")
        if self.current_position is None:
            return self._record_error("draw_line_to requires current_position is not None.")
        start = self.current_position
        end = Point2D(x=x, y=y)
        stroke_id = self._next_stroke_id()
        stroke = LineStroke(stroke_id=stroke_id, start=start, end=end)
        self.strokes.append(stroke)
        self.actions.append(
            PrimitiveAction(
                name="draw_line",
                frame=self.frame,
                stroke_id=stroke_id,
                params={
                    "start": start.model_dump(mode="json"),
                    "end": end.model_dump(mode="json"),
                    "sampling_hint": "kinematics module should interpolate Cartesian waypoints",
                },
            )
        )
        self.current_position = end
        return self.check_plan()

    def draw_arc(
        self,
        center_x: float,
        center_y: float,
        radius_m: float,
        start_angle_rad: float,
        end_angle_rad: float,
        direction: Direction,
    ) -> dict[str, Any]:
        """Append a draw_arc primitive.

        The arc is symbolic geometry for the downstream kinematics module; this
        method does not sample trajectories.
        """

        if self.pen_state != "down":
            return self._record_error("draw_arc requires pen_state == 'down'.")
        if radius_m <= 0:
            return self._record_error("draw_arc requires radius_m > 0.")
        center = Point2D(x=center_x, y=center_y)
        stroke_id = self._next_stroke_id()
        stroke = ArcStroke(
            stroke_id=stroke_id,
            center=center,
            radius_m=radius_m,
            start_angle_rad=start_angle_rad,
            end_angle_rad=end_angle_rad,
            direction=direction,
        )
        self.strokes.append(stroke)
        self.actions.append(
            PrimitiveAction(
                name="draw_arc",
                frame=self.frame,
                stroke_id=stroke_id,
                params={
                    "center": center.model_dump(mode="json"),
                    "radius_m": radius_m,
                    "start_angle_rad": start_angle_rad,
                    "end_angle_rad": end_angle_rad,
                    "direction": direction,
                    "sampling_hint": "kinematics module should sample circular Cartesian waypoints",
                },
            )
        )
        self.current_position = self._arc_end_point(stroke)
        return self.check_plan()

    def pen_up(self) -> dict[str, Any]:
        """Append a pen_up primitive."""

        if self.pen_state != "down":
            return self._record_error("pen_up requires pen_state == 'down'.")
        self.pen_state = "up"
        self.actions.append(
            PrimitiveAction(
                name="pen_up",
                frame=self.frame,
                stroke_id=None,
                params={"note": "symbolic lift request only; no robot motion is computed"},
            )
        )
        return self.check_plan()

    def check_plan(self) -> dict[str, Any]:
        """Return a machine-readable summary of builder state."""

        return {
            "current_position": (
                self.current_position.model_dump(mode="json")
                if self.current_position is not None
                else None
            ),
            "pen_state": self.pen_state,
            "number_of_actions": len(self.actions),
            "number_of_strokes": len(self.strokes),
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "finished": self.finished,
        }

    def finish_plan(self) -> dict[str, Any]:
        """Mark the plan complete if at least one drawable stroke exists."""

        if not self.strokes:
            return self._record_error("finish_plan requires at least one drawable stroke.")
        if self.pen_state == "down":
            self.warnings.append("finish_plan called while pen is still down.")
        self.finished = True
        return self.check_plan()

    def _record_error(self, message: str) -> dict[str, Any]:
        self.errors.append(message)
        return self.check_plan()

    def _next_stroke_id(self) -> str:
        return f"stroke_{len(self.strokes) + 1:03d}"

    @staticmethod
    def _arc_end_point(stroke: ArcStroke) -> Point2D:
        if math.isclose(
            abs(stroke.end_angle_rad - stroke.start_angle_rad),
            2.0 * math.pi,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            return Point2D(
                x=stroke.center.x + stroke.radius_m * math.cos(stroke.start_angle_rad),
                y=stroke.center.y + stroke.radius_m * math.sin(stroke.start_angle_rad),
            )
        return Point2D(
            x=stroke.center.x + stroke.radius_m * math.cos(stroke.end_angle_rad),
            y=stroke.center.y + stroke.radius_m * math.sin(stroke.end_angle_rad),
        )

