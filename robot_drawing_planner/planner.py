"""High-level conversion from natural language to primitive action JSON."""

from __future__ import annotations

from robot_drawing_planner.geometry import generate_strokes, stroke_end_point, stroke_start_point
from robot_drawing_planner.llm_client import Invokable, parse_goal
from robot_drawing_planner.schemas import (
    ArcStroke,
    Board,
    DrawingPlan,
    LineStroke,
    ParsedGoal,
    PrimitiveAction,
    Stroke,
)
from robot_drawing_planner.validators import (
    normalize_goal,
    validate_no_low_level_robot_fields,
    validate_strokes_within_board,
)


def plan_from_text(
    command: str,
    parser: Invokable | None = None,
    board: Board | None = None,
) -> DrawingPlan:
    """Parse natural language and return a deterministic primitive plan."""

    parsed = parse_goal(command, parser=parser)
    return plan_from_goal(parsed, board=board)


def plan_from_goal(parsed: ParsedGoal, board: Board | None = None) -> DrawingPlan:
    """Return a primitive action plan from an already parsed drawing goal."""

    drawing_board = board or Board()
    goal = normalize_goal(parsed)
    strokes = generate_strokes(goal)
    validate_strokes_within_board(strokes, drawing_board)
    actions = strokes_to_actions(strokes)
    plan = DrawingPlan(
        source_command=parsed.raw_command,
        goal=goal,
        strokes=strokes,
        actions=actions,
        diagnostics={"board": drawing_board.model_dump(mode="json")},
    )
    validate_no_low_level_robot_fields(plan.model_dump(mode="json"))
    return plan


def strokes_to_actions(strokes: list[Stroke]) -> list[PrimitiveAction]:
    """Convert deterministic strokes into robot-level primitive actions."""

    actions: list[PrimitiveAction] = []
    for stroke in strokes:
        start = stroke_start_point(stroke)
        actions.extend(
            [
                PrimitiveAction(
                    name="move_to_start",
                    stroke_id=stroke.stroke_id,
                    params={"target": start.model_dump(mode="json")},
                ),
                PrimitiveAction(
                    name="align_pen_orientation",
                    stroke_id=stroke.stroke_id,
                    params={"orientation": "board_normal"},
                ),
                PrimitiveAction(
                    name="pen_down",
                    stroke_id=stroke.stroke_id,
                    params={"contact": "drawing_board"},
                ),
            ]
        )
        if isinstance(stroke, LineStroke):
            actions.append(
                PrimitiveAction(
                    name="draw_line",
                    stroke_id=stroke.stroke_id,
                    params={
                        "start": stroke.start.model_dump(mode="json"),
                        "end": stroke.end.model_dump(mode="json"),
                    },
                )
            )
        elif isinstance(stroke, ArcStroke):
            actions.append(
                PrimitiveAction(
                    name="draw_arc",
                    stroke_id=stroke.stroke_id,
                    params={
                        "start": stroke_start_point(stroke).model_dump(mode="json"),
                        "end": stroke_end_point(stroke).model_dump(mode="json"),
                        "center": stroke.center.model_dump(mode="json"),
                        "radius_m": stroke.radius_m,
                        "start_angle_rad": stroke.start_angle_rad,
                        "end_angle_rad": stroke.end_angle_rad,
                        "direction": stroke.direction,
                    },
                )
            )
        else:
            raise TypeError(f"Unknown stroke type: {type(stroke).__name__}")
        actions.append(
            PrimitiveAction(name="pen_up", stroke_id=stroke.stroke_id, params={})
        )
    return actions

