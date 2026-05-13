import json

from robot_drawing_planner.action_tools import UnitActionToolset, create_unit_action_tools


def invoke(toolset: UnitActionToolset, name: str, payload: dict | None = None):
    return toolset.tool_by_name(name).invoke(payload or {})


def run_square_sequence(toolset: UnitActionToolset):
    assert invoke(toolset, "begin_plan", {"source_command": "draw a square"})["ok"] is True
    assert invoke(toolset, "move_to_start", {"x": -0.05, "y": 0.05})["ok"] is True
    assert invoke(toolset, "align_pen_orientation")["ok"] is True
    assert invoke(toolset, "pen_down")["ok"] is True
    assert invoke(toolset, "draw_line_to", {"x": 0.05, "y": 0.05})["ok"] is True
    assert invoke(toolset, "draw_line_to", {"x": 0.05, "y": -0.05})["ok"] is True
    assert invoke(toolset, "draw_line_to", {"x": -0.05, "y": -0.05})["ok"] is True
    assert invoke(toolset, "draw_line_to", {"x": -0.05, "y": 0.05})["ok"] is True
    assert invoke(toolset, "pen_up")["ok"] is True


def test_begin_plan_then_square_sequence_direct_tool_invocation():
    toolset, tools = create_unit_action_tools()
    assert {tool.name for tool in tools} == {
        "begin_plan",
        "move_to_start",
        "align_pen_orientation",
        "pen_down",
        "draw_line_to",
        "draw_arc",
        "pen_up",
        "check_plan",
        "finish_plan",
    }

    run_square_sequence(toolset)

    assert toolset.builder is not None
    assert len(toolset.builder.actions) == 8
    assert len(toolset.builder.strokes) == 4


def test_check_plan_returns_ok_true_for_valid_square():
    toolset = UnitActionToolset()
    run_square_sequence(toolset)

    feedback = invoke(toolset, "check_plan")

    assert feedback["ok"] is True
    assert feedback["message"] == "Plan checked."
    assert feedback["current_position"] == {"x": -0.05, "y": 0.05, "unit": "m"}
    assert feedback["pen_state"] == "up"
    assert feedback["action_count"] == 8
    assert feedback["warnings"] == []
    assert feedback["errors"] == []


def test_finish_plan_returns_serializable_plan_payload():
    toolset = UnitActionToolset()
    run_square_sequence(toolset)
    invoke(toolset, "check_plan")

    feedback = invoke(toolset, "finish_plan")

    assert feedback["ok"] is True
    assert feedback["message"] == "Plan finished."
    assert feedback["action_count"] == 8
    assert feedback["plan"]["source_command"] == "draw a square"
    assert feedback["plan"]["finished"] is True
    assert len(feedback["plan"]["actions"]) == 8
    assert len(feedback["plan"]["strokes"]) == 4
    json.dumps(feedback["plan"])


def test_finish_plan_requires_check_plan_before_finish():
    toolset = UnitActionToolset()
    run_square_sequence(toolset)

    feedback = invoke(toolset, "finish_plan")

    assert feedback["ok"] is False
    assert "requires check_plan" in feedback["message"]


def test_invalid_pen_state_returns_ok_false():
    toolset = UnitActionToolset()
    invoke(toolset, "begin_plan", {"source_command": "bad plan"})
    invoke(toolset, "move_to_start", {"x": 0.0, "y": 0.0})

    feedback = invoke(toolset, "draw_line_to", {"x": 0.1, "y": 0.0})

    assert feedback["ok"] is False
    assert "draw_line_to requires pen_state == 'down'." == feedback["message"]
    assert feedback["action_count"] == 1
    assert feedback["errors"] == ["draw_line_to requires pen_state == 'down'."]


def test_tool_descriptions_are_explicit_about_scope():
    toolset = UnitActionToolset()
    for tool in toolset.tools():
        description = tool.description
        assert "symbolic robot primitive actions" in description
        assert "does not move the robot" in description
        assert "does not compute IK, FK, Jacobians" in description
        assert "Isaac Sim commands" in description
    assert "must call check_plan before finish_plan" in toolset.tool_by_name(
        "finish_plan"
    ).description

