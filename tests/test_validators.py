import math

import pytest
from pydantic import ValidationError

from robot_drawing_planner.geometry import generate_strokes
from robot_drawing_planner.schemas import (
    ArcStroke,
    Board,
    DrawingPlan,
    LineStroke,
    Measurement,
    NormalizedGoal,
    ParsedGoal,
    Point2D,
    Point3D,
    PrimitiveAction,
    ValidationErrorReport,
)
from robot_drawing_planner.validators import (
    PlannerValidationError,
    normalize_goal,
    validate_no_low_level_robot_fields,
    validate_strokes_within_board,
)


def test_instantiates_every_schema_model():
    point_2d = Point2D(x=0.0, y=0.0)
    point_3d = Point3D(x=0.0, y=0.0, z=0.1)
    measurement = Measurement(value=5.0, unit="cm")
    parsed = ParsedGoal(
        shape_type="circle",
        center=point_2d,
        radius=measurement,
        raw_command="draw a circle",
    )
    normalized = NormalizedGoal(
        shape_type="circle",
        center=point_2d,
        radius_m=0.05,
        side_length_m=None,
        size_m=0.10,
        orientation_rad=0.0,
        letter=None,
        assumptions=[],
        warnings=[],
    )
    line = LineStroke(stroke_id="stroke_line", start=point_2d, end=Point2D(x=0.1, y=0.0))
    arc = ArcStroke(
        stroke_id="stroke_arc",
        center=point_2d,
        radius_m=0.05,
        start_angle_rad=0.0,
        end_angle_rad=math.pi,
        direction="ccw",
    )
    action = PrimitiveAction(
        name="draw_line",
        stroke_id=line.stroke_id,
        params={"start": line.start.model_dump(), "end": line.end.model_dump()},
    )
    plan = DrawingPlan(
        source_command=parsed.raw_command,
        goal=normalized,
        strokes=[line, arc],
        actions=[action],
        diagnostics={"point_3d_unit": point_3d.unit},
    )
    report = ValidationErrorReport(ok=True, errors=[], warnings=[])

    assert plan.schema_version == "1.0"
    assert report.ok is True


def test_measurement_rejects_non_positive_values():
    with pytest.raises(ValidationError):
        Measurement(value=0, unit="cm")


def test_point2d_rejects_extra_fields():
    with pytest.raises(ValidationError):
        Point2D(x=0.0, y=0.0, z=0.0)


def test_primitive_action_rejects_invalid_name():
    with pytest.raises(ValidationError):
        PrimitiveAction(name="solve_ik", params={})


def test_parsed_goal_rejects_unsupported_shape():
    with pytest.raises(ValidationError):
        ParsedGoal(shape_type="star", size=Measurement(value=5, unit="cm"), raw_command="draw a star")


def test_normalize_goal_rejects_unsupported_letter():
    with pytest.raises(PlannerValidationError, match="Unsupported letter"):
        normalize_goal(
            ParsedGoal(
                shape_type="letter",
                letter="B",
                size=Measurement(value=5, unit="cm"),
                raw_command="draw B",
            )
        )


def test_normalize_goal_converts_units():
    goal = normalize_goal(
        ParsedGoal(
            shape_type="circle",
            center=Point2D(x=0.10, y=-0.05),
            radius=Measurement(value=50, unit="mm"),
            raw_command="draw a circle",
        )
    )
    assert goal.radius_m == pytest.approx(0.05)
    assert goal.size_m == pytest.approx(0.10)
    assert goal.center.x == pytest.approx(0.10)
    assert goal.center.y == pytest.approx(-0.05)


def test_board_boundary_check_rejects_large_square():
    goal = normalize_goal(
        ParsedGoal(
            shape_type="square",
            side_length=Measurement(value=50, unit="cm"),
            raw_command="draw a square",
        )
    )
    strokes = generate_strokes(goal)
    with pytest.raises(PlannerValidationError, match="board boundaries"):
        validate_strokes_within_board(strokes, Board(width_m=0.40, height_m=0.30))


def test_no_low_level_robot_fields_rejects_ik_claims():
    with pytest.raises(PlannerValidationError, match="outside planner scope"):
        validate_no_low_level_robot_fields({"actions": [{"ik": "solved"}]})

