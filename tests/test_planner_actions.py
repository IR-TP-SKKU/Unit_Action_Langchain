import json

import pytest

from robot_drawing_planner.llm_client import parse_command_with_llm
from robot_drawing_planner.planner import plan_from_goal, plan_from_text
from robot_drawing_planner.schemas import Measurement, ParsedGoal


class FakeParser:
    def __init__(self, goal):
        self.goal = goal
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        return self.goal


class FakeLLM:
    def __init__(self, result):
        self.result = result
        self.structured_calls = []
        self.invoke_calls = []

    def with_structured_output(self, schema, method):
        self.structured_calls.append((schema, method))
        return self

    def invoke(self, messages):
        self.invoke_calls.append(messages)
        return self.result


def test_plan_from_goal_square_action_sequence():
    plan = plan_from_goal(
        ParsedGoal(
            shape_type="square",
            side_length=Measurement(value=10, unit="cm"),
            raw_command="draw a square",
        )
    )
    action_names = [action.name for action in plan.actions]
    assert action_names[:5] == [
        "move_to_start",
        "align_pen_orientation",
        "pen_down",
        "draw_line",
        "pen_up",
    ]
    assert action_names.count("draw_line") == 4
    assert action_names.count("draw_arc") == 0


def test_circle_uses_draw_arc():
    plan = plan_from_goal(
        ParsedGoal(
            shape_type="circle",
            size=Measurement(value=100, unit="mm"),
            raw_command="draw a circle",
        )
    )
    action_names = [action.name for action in plan.actions]
    assert action_names.count("draw_arc") == 1
    assert action_names.count("draw_line") == 0


def test_plan_from_text_uses_fake_parser_without_live_api():
    fake = FakeParser(
        {
            "shape_type": "letter",
            "letter": "A",
            "size": {"value": 8, "unit": "cm"},
            "raw_command": "draw a capital A",
        }
    )
    plan = plan_from_text("draw a capital A", parser=fake)
    assert fake.calls
    assert plan.goal.shape_type == "letter"
    assert plan.goal.letter == "A"
    assert [action.name for action in plan.actions].count("draw_line") == 3


def test_korean_square_command_parses_with_fake_llm():
    command = "중앙에 한 변 10cm 네모"
    fake = FakeLLM(
        ParsedGoal(
            shape_type="square",
            center=None,
            side_length=Measurement(value=10, unit="cm"),
            position_hint="중앙",
            raw_command=command,
        )
    )
    parsed = parse_command_with_llm(command, llm=fake)

    assert parsed.shape_type == "square"
    assert parsed.side_length == Measurement(value=10, unit="cm")
    assert parsed.center is None
    assert parsed.position_hint == "중앙"
    assert parsed.raw_command == command
    assert fake.structured_calls == [(ParsedGoal, "json_schema")]


def test_english_circle_radius_command_parses_with_fake_llm_dict_result():
    command = "Draw a circle with radius 5 cm"
    fake = FakeLLM(
        {
            "shape_type": "circle",
            "center": None,
            "radius": {"value": 5, "unit": "cm"},
            "raw_command": command,
        }
    )
    parsed = parse_command_with_llm(command, llm=fake)

    assert parsed.shape_type == "circle"
    assert parsed.radius == Measurement(value=5, unit="cm")
    assert parsed.raw_command == command


def test_missing_api_key_raises_only_when_live_llm_requested(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    command = "Draw a circle with radius 5 cm"
    fake = FakeLLM(
        ParsedGoal(
            shape_type="circle",
            radius=Measurement(value=5, unit="cm"),
            raw_command=command,
        )
    )

    parsed = parse_command_with_llm(command, llm=fake)
    assert parsed.shape_type == "circle"

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY is not set"):
        parse_command_with_llm(command)


def test_plan_json_contains_no_low_level_robot_fields():
    plan = plan_from_goal(
        ParsedGoal(
            shape_type="triangle",
            side_length=Measurement(value=8, unit="cm"),
            raw_command="draw a triangle",
        )
    )
    payload = json.dumps(plan.model_dump(mode="json"))
    forbidden = ["joint_angles", "ik", "fk", "jacobian", "isaac_command", "trajectory"]
    assert not any(word in payload for word in forbidden)


def test_empty_command_rejected_before_parser_call():
    fake = FakeParser(
        {
            "shape_type": "circle",
            "size": {"value": 5, "unit": "cm"},
            "raw_command": "draw a circle",
        }
    )
    with pytest.raises(ValueError, match="must not be empty"):
        plan_from_text("   ", parser=fake)
    assert fake.calls == []
