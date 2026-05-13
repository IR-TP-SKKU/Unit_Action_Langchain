import json

from langchain_core.messages import AIMessage

from robot_drawing_planner.agent_events import AgentRunResult
from robot_drawing_planner.agent_planner import plan_drawing_agentic
from robot_drawing_planner.schemas import DrawingPlan


class FakeToolCallingLLM:
    def __init__(self, tool_call_batches):
        self.tool_call_batches = list(tool_call_batches)
        self.bound_tools = None

    def bind_tools(self, tools):
        self.bound_tools = tools
        return self

    def invoke(self, messages):
        if self.tool_call_batches:
            batch = self.tool_call_batches.pop(0)
        else:
            batch = []
        return AIMessage(content="", tool_calls=batch)


def tc(name, args=None, call_id=None):
    return {
        "name": name,
        "args": args or {},
        "id": call_id or f"{name}_id",
    }


def square_batches():
    return [
        [tc("begin_plan", {"source_command": "draw square"})],
        [tc("move_to_start", {"x": -0.05, "y": 0.05})],
        [tc("align_pen_orientation")],
        [tc("pen_down")],
        [tc("draw_line_to", {"x": 0.05, "y": 0.05})],
        [tc("draw_line_to", {"x": 0.05, "y": -0.05})],
        [tc("draw_line_to", {"x": -0.05, "y": -0.05})],
        [tc("draw_line_to", {"x": -0.05, "y": 0.05})],
        [tc("pen_up")],
        [tc("check_plan")],
        [tc("finish_plan")],
    ]


def test_agentic_collect_events_returns_agent_run_result_for_square():
    result = plan_drawing_agentic(
        "draw a square",
        llm=FakeToolCallingLLM(square_batches()),
        collect_events=True,
    )

    assert isinstance(result, AgentRunResult)
    assert result.command == "draw a square"
    assert result.plan.diagnostics["validation_ok"] is True
    assert result.events


def test_agentic_events_include_required_event_types():
    result = plan_drawing_agentic(
        "draw a square",
        llm=FakeToolCallingLLM(square_batches()),
        collect_events=True,
    )

    event_types = [event.event_type for event in result.events]
    assert "user_request" in event_types
    assert any(
        event.event_type == "tool_call" and event.tool_name == "move_to_start"
        for event in result.events
    )
    assert "tool_result" in event_types
    assert "plan_finished" in event_types


def test_agentic_tool_call_events_preserve_fake_llm_order():
    result = plan_drawing_agentic(
        "draw a square",
        llm=FakeToolCallingLLM(square_batches()),
        collect_events=True,
    )

    tool_names = [
        event.tool_name for event in result.events if event.event_type == "tool_call"
    ]

    assert tool_names == [
        "begin_plan",
        "move_to_start",
        "align_pen_orientation",
        "pen_down",
        "draw_line_to",
        "draw_line_to",
        "draw_line_to",
        "draw_line_to",
        "pen_up",
        "check_plan",
        "finish_plan",
    ]


def test_agentic_collect_events_false_still_returns_drawing_plan():
    plan = plan_drawing_agentic(
        "draw a square",
        llm=FakeToolCallingLLM(square_batches()),
        collect_events=False,
    )

    assert isinstance(plan, DrawingPlan)
    assert plan.diagnostics["validation_ok"] is True


def test_agentic_event_callback_receives_events_without_changing_return_type():
    seen = []

    plan = plan_drawing_agentic(
        "draw a square",
        llm=FakeToolCallingLLM(square_batches()),
        event_callback=seen.append,
    )

    assert isinstance(plan, DrawingPlan)
    assert seen
    assert seen[0].event_type == "user_request"
    assert any(event.event_type == "plan_finished" for event in seen)


def test_agentic_events_do_not_contain_openai_api_key_label():
    result = plan_drawing_agentic(
        "draw a square",
        llm=FakeToolCallingLLM(square_batches()),
        collect_events=True,
    )

    payload = json.dumps(
        [event.model_dump(mode="json") for event in result.events],
        ensure_ascii=False,
    )
    assert "OPENAI_API_KEY" not in payload
