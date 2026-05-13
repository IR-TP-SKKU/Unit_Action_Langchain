"""High-level conversion from natural language to primitive action JSON."""

from __future__ import annotations

from robot_drawing_planner.geometry import ArcStroke, LineStroke, Stroke, generate_strokes
from robot_drawing_planner.llm_client import Invokable, parse_goal
from robot_drawing_planner.schemas import (
    AlignPenOrientationAction,
    Board,
    DrawArcAction,
    DrawLineAction,
    DrawingGoal,
    MoveToStartAction,
    ParsedGoal,
    PenDownAction,
    PenUpAction,
    PrimitivePlan,
    RobotPrimitiveAction,
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
) -> PrimitivePlan:
    """Parse natural language and return a deterministic primitive plan."""

    parsed = parse_goal(command, parser=parser)
    return plan_from_goal(parsed, board=board)


def plan_from_goal(parsed: ParsedGoal, board: Board | None = None) -> PrimitivePlan:
    """Return a primitive action plan from an already parsed drawing goal."""

    drawing_board = board or Board()
    goal = normalize_goal(parsed)
    strokes = generate_strokes(goal)
    validate_strokes_within_board(strokes, drawing_board)
    actions = strokes_to_actions(strokes)
    plan = PrimitivePlan(board=drawing_board, goal=goal, actions=actions)
    validate_no_low_level_robot_fields(plan.model_dump(mode="json"))
    return plan


def strokes_to_actions(strokes: list[Stroke]) -> list[RobotPrimitiveAction]:
    """Convert deterministic strokes into robot-level primitive actions."""

    actions: list[RobotPrimitiveAction] = []
    for stroke in strokes:
        start = stroke.start_point()
        actions.extend(
            [
                MoveToStartAction(target=start),
                AlignPenOrientationAction(),
                PenDownAction(),
            ]
        )
        if isinstance(stroke, LineStroke):
            actions.append(DrawLineAction(start=stroke.start, end=stroke.end))
        elif isinstance(stroke, ArcStroke):
            actions.append(
                DrawArcAction(
                    start=stroke.start_point(),
                    end=stroke.end_point(),
                    center=stroke.center,
                    radius_m=stroke.radius_m,
                    start_angle_rad=stroke.start_angle_rad,
                    end_angle_rad=stroke.end_angle_rad,
                    clockwise=stroke.clockwise,
                )
            )
        else:
            raise TypeError(f"Unknown stroke type: {type(stroke).__name__}")
        actions.append(PenUpAction())
    return actions

