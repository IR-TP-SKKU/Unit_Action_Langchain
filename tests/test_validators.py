import pytest

from robot_drawing_planner.geometry import generate_strokes
from robot_drawing_planner.schemas import Board, ParsedGoal
from robot_drawing_planner.validators import (
    PlannerValidationError,
    normalize_goal,
    validate_no_low_level_robot_fields,
    validate_strokes_within_board,
)


def test_normalize_goal_rejects_unsupported_shape():
    with pytest.raises(PlannerValidationError, match="Unsupported shape"):
        normalize_goal(ParsedGoal(object_type="star", size=5, unit="cm"))


def test_normalize_goal_rejects_unsupported_letter():
    with pytest.raises(PlannerValidationError, match="Unsupported letter"):
        normalize_goal(ParsedGoal(object_type="letter", letter="B", size=5, unit="cm"))


def test_normalize_goal_converts_units():
    goal = normalize_goal(
        ParsedGoal(
            object_type="circle",
            size=50,
            unit="mm",
            center_x=10,
            center_y=-5,
            center_unit="cm",
        )
    )
    assert goal.size_m == pytest.approx(0.05)
    assert goal.center.x_m == pytest.approx(0.10)
    assert goal.center.y_m == pytest.approx(-0.05)


def test_board_boundary_check_rejects_large_square():
    goal = normalize_goal(ParsedGoal(object_type="square", size=50, unit="cm"))
    strokes = generate_strokes(goal)
    with pytest.raises(PlannerValidationError, match="board boundaries"):
        validate_strokes_within_board(strokes, Board(width_m=0.40, height_m=0.30))


def test_no_low_level_robot_fields_rejects_ik_claims():
    with pytest.raises(PlannerValidationError, match="outside planner scope"):
        validate_no_low_level_robot_fields({"actions": [{"ik": "solved"}]})

