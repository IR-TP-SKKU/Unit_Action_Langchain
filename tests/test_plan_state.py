from robot_drawing_planner.plan_state import PlanBuilder
from robot_drawing_planner.schemas import PrimitiveAction


ALLOWED_PRIMITIVES = {
    "move_to_start",
    "align_pen_orientation",
    "pen_down",
    "draw_line",
    "draw_arc",
    "pen_up",
}


def build_square_plan() -> PlanBuilder:
    builder = PlanBuilder.begin_plan("draw a square with unit action tools")
    builder.move_to_start(-0.05, 0.05)
    builder.align_pen_orientation()
    builder.pen_down()
    builder.draw_line_to(0.05, 0.05)
    builder.draw_line_to(0.05, -0.05)
    builder.draw_line_to(-0.05, -0.05)
    builder.draw_line_to(-0.05, 0.05)
    builder.pen_up()
    return builder


def test_valid_square_sequence_creates_eight_primitive_actions():
    builder = build_square_plan()
    summary = builder.finish_plan()

    assert summary["finished"] is True
    assert summary["errors"] == []
    assert len(builder.actions) == 8
    assert [action.name for action in builder.actions] == [
        "move_to_start",
        "align_pen_orientation",
        "pen_down",
        "draw_line",
        "draw_line",
        "draw_line",
        "draw_line",
        "pen_up",
    ]
    assert len(builder.strokes) == 4


def test_draw_line_to_before_pen_down_produces_error():
    builder = PlanBuilder.begin_plan("bad line")
    builder.move_to_start(0.0, 0.0)
    summary = builder.draw_line_to(0.1, 0.0)

    assert "draw_line_to requires pen_state == 'down'." in summary["errors"]
    assert len(builder.strokes) == 0
    assert [action.name for action in builder.actions] == ["move_to_start"]


def test_move_to_start_while_pen_is_down_produces_error():
    builder = PlanBuilder.begin_plan("bad move")
    builder.move_to_start(0.0, 0.0)
    builder.pen_down()
    summary = builder.move_to_start(0.1, 0.1)

    assert "move_to_start requires pen_state == 'up'." in summary["errors"]
    assert [action.name for action in builder.actions] == ["move_to_start", "pen_down"]
    assert builder.current_position.x == 0.0
    assert builder.current_position.y == 0.0


def test_finish_plan_without_strokes_fails():
    builder = PlanBuilder.begin_plan("empty plan")
    summary = builder.finish_plan()

    assert summary["finished"] is False
    assert "finish_plan requires at least one drawable stroke." in summary["errors"]
    assert builder.actions == []


def test_pen_up_after_drawing_succeeds():
    builder = PlanBuilder.begin_plan("line")
    builder.move_to_start(0.0, 0.0)
    builder.pen_down()
    builder.draw_line_to(0.1, 0.0)
    summary = builder.pen_up()

    assert summary["pen_state"] == "up"
    assert summary["errors"] == []
    assert builder.actions[-1].name == "pen_up"


def test_no_action_name_outside_allowed_robot_primitives():
    builder = build_square_plan()

    assert all(isinstance(action, PrimitiveAction) for action in builder.actions)
    assert {action.name for action in builder.actions}.issubset(ALLOWED_PRIMITIVES)

