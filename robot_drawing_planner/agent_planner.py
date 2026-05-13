"""Agentic LangChain planner that builds plans through unit-action tools."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from robot_drawing_planner.agent_events import AgentRunEvent, AgentRunResult, make_event
from robot_drawing_planner.action_tools import UnitActionToolset
from robot_drawing_planner.config import DEFAULT_CONFIG, PlannerConfig
from robot_drawing_planner.llm_client import get_llm
from robot_drawing_planner.schemas import DrawingPlan, NormalizedGoal, Point2D


def build_agent_system_prompt(config: PlannerConfig) -> str:
    """Build the system prompt with the active board bounds."""

    x_min, x_max, y_min, y_max = _board_ranges(config)
    return f"""You are a robot drawing action planner.

You must plan by calling the available Unit Action tools.
You must not output a final plan directly.
You must call tools step by step.
Use only line and arc primitives.

Board frame convention:
- origin is board center.
- coordinates are meters.
- current board x range is [{_fmt(x_min)}, {_fmt(x_max)}].
- current board y range is [{_fmt(y_min)}, {_fmt(y_max)}].
- default board x range is [-0.25, 0.25].
- default board y range is [-0.175, 0.175].

Scale:
- Use small drawing coordinates, usually within about 0.05 to 0.15 m.
- Do not use pixel-like or arbitrary coordinates such as 1, 2, 3.
- Keep every point and arc bounding box inside the board range.

Required Unit Action sequence:
- begin_plan first.
- move_to_start before pen_down.
- align_pen_orientation before drawing.
- pen_down before draw_line_to or draw_arc.
- pen_up before moving to another stroke.
- pen_up before finish_plan.
- check_plan before finish_plan.

Arc rules:
- Before draw_arc, the arc start point must match current pen position.
- For a circle centered at (cx, cy) with radius r and start_angle 0, first call move_to_start(cx+r, cy), then pen_down, then draw_arc.

Open-ended shapes:
- For house/star/smiley/letters, approximate with multiple line/arc strokes inside the board.

The tools append symbolic robot primitive actions only. They do not move the robot.
Do not compute IK, FK, Jacobians, joint commands, joint velocities, trajectory
samples, or Isaac Sim commands.
"""


def _board_ranges(config: PlannerConfig) -> tuple[float, float, float, float]:
    half_width = config.board_width_m / 2.0
    half_height = config.board_height_m / 2.0
    return -half_width, half_width, -half_height, half_height


def _fmt(value: float) -> str:
    return f"{value:g}"


AGENT_SYSTEM_PROMPT = build_agent_system_prompt(DEFAULT_CONFIG)

PLANNER_SCOPE_NOTE = (
    "This agentic planner outputs primitive action JSON only; it does not compute "
    "IK, FK, Jacobians, joint commands, trajectory samples, or Isaac Sim commands."
)


def plan_drawing_agentic(
    command: str,
    config: PlannerConfig | None = None,
    llm: Any | None = None,
    max_steps: int = 30,
    event_callback: Callable[[AgentRunEvent], None] | None = None,
    collect_events: bool = False,
) -> DrawingPlan | AgentRunResult:
    return _plan_drawing_agentic_impl(
        command=command,
        config=config,
        llm=llm,
        max_steps=max_steps,
        event_callback=event_callback,
        collect_events=collect_events,
    )


def _plan_drawing_agentic_impl(
    command: str,
    config: PlannerConfig | None = None,
    llm: Any | None = None,
    max_steps: int = 30,
    event_callback: Callable[[AgentRunEvent], None] | None = None,
    collect_events: bool = False,
) -> DrawingPlan | AgentRunResult:
    """Plan by letting an LLM sequentially call symbolic unit-action tools."""

    planner_config = config or DEFAULT_CONFIG
    toolset = UnitActionToolset(config=planner_config)
    tools = toolset.tools()
    model = llm if llm is not None else get_llm()
    bound_model = model.bind_tools(tools) if hasattr(model, "bind_tools") else model

    messages: list[Any] = [
        SystemMessage(content=build_agent_system_prompt(planner_config)),
        HumanMessage(content=command),
    ]
    last_finish_feedback: dict[str, Any] | None = None
    errors: list[str] = []
    warnings: list[str] = []
    events: list[AgentRunEvent] = []
    should_emit_events = collect_events or event_callback is not None

    def emit(
        event_type: str,
        message: str,
        *,
        step_index: int | None = None,
        tool_name: str | None = None,
        tool_args: dict[str, Any] | None = None,
        tool_result: dict[str, Any] | None = None,
        ok: bool | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not should_emit_events:
            return
        event = make_event(
            event_index=len(events),
            step_index=step_index,
            event_type=event_type,  # type: ignore[arg-type]
            tool_name=tool_name,
            tool_args=_json_safe_dict(tool_args),
            tool_result=_json_safe_dict(tool_result),
            message=message,
            ok=ok,
            metadata=_json_safe_dict(metadata) or {},
        )
        events.append(event)
        if event_callback is not None:
            event_callback(event)

    def result(plan: DrawingPlan) -> DrawingPlan | AgentRunResult:
        if collect_events:
            return AgentRunResult(command=command, plan=plan, events=events)
        return plan

    emit(
        "user_request",
        command,
        ok=True,
        metadata={"mode": "agentic_unit_action_tools"},
    )

    for _step in range(max_steps):
        response = bound_model.invoke(messages)
        messages.append(response)
        tool_calls = _extract_tool_calls(response)
        emit(
            "llm_message",
            f"LLM response requested {len(tool_calls)} tool call(s).",
            step_index=_step,
            ok=True,
            metadata={"tool_call_count": len(tool_calls)},
        )
        if not tool_calls:
            feedback = {
                "ok": False,
                "message": (
                    "No tool call was returned. Continue by calling the next Unit Action tool; "
                    "do not write the final plan directly."
                ),
            }
            messages.append(HumanMessage(content=json.dumps(feedback)))
            continue

        for tool_call in tool_calls:
            name = tool_call["name"]
            args = tool_call.get("args") or {}
            tool_call_id = tool_call.get("id") or f"tool_call_{len(messages)}"
            emit(
                "tool_call",
                f"Calling unit-action tool '{name}'.",
                step_index=_step,
                tool_name=name,
                tool_args=args,
            )
            feedback = _execute_tool_call(toolset, name, args)
            emit(
                "tool_result",
                str(feedback.get("message", "")),
                step_index=_step,
                tool_name=name,
                tool_result=_tool_result_event_payload(feedback),
                ok=bool(feedback.get("ok", False)),
            )
            messages.append(
                ToolMessage(
                    content=json.dumps(feedback, ensure_ascii=False),
                    tool_call_id=tool_call_id,
                )
            )
            if not feedback.get("ok", False):
                errors = list(feedback.get("errors") or [])
            if name == "finish_plan":
                last_finish_feedback = feedback
                if feedback.get("ok") is True:
                    plan = _drawing_plan_from_toolset(
                        command=command,
                        toolset=toolset,
                        config=planner_config,
                        validation_ok=True,
                        extra_errors=[],
                        extra_warnings=[],
                    )
                    emit(
                        "plan_finished",
                        "finish_plan succeeded.",
                        step_index=_step,
                        ok=True,
                        metadata={
                            "action_count": len(plan.actions),
                            "stroke_count": len(plan.strokes),
                        },
                    )
                    return result(plan)
                errors = [feedback.get("message", "finish_plan failed")]

    errors.append(f"Agentic planning failed: max_steps {max_steps} exceeded.")
    if last_finish_feedback and last_finish_feedback.get("message"):
        warnings.append(f"Last finish_plan feedback: {last_finish_feedback['message']}")
    plan = _drawing_plan_from_toolset(
        command=command,
        toolset=toolset,
        config=planner_config,
        validation_ok=False,
        extra_errors=errors,
        extra_warnings=warnings,
        force_empty_actions=True,
    )
    emit(
        "error",
        f"Agentic planning failed: max_steps {max_steps} exceeded.",
        ok=False,
        metadata={"max_steps": max_steps},
    )
    return result(plan)


def _tool_result_event_payload(feedback: dict[str, Any]) -> dict[str, Any]:
    """Keep tool-result events small and free of raw provider objects."""

    keys = [
        "ok",
        "message",
        "current_position",
        "pen_state",
        "action_count",
        "warnings",
        "errors",
    ]
    return {key: feedback.get(key) for key in keys if key in feedback}


def _json_safe_dict(value: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return a JSON-serializable copy suitable for AgentRunEvent fields."""

    if value is None:
        return None
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def _execute_tool_call(
    toolset: UnitActionToolset,
    name: str,
    args: dict[str, Any],
) -> dict[str, Any]:
    try:
        tool = toolset.tool_by_name(name)
    except KeyError:
        return {
            "ok": False,
            "message": f"Unknown unit-action tool '{name}'.",
            "current_position": None,
            "pen_state": "up",
            "action_count": 0,
            "warnings": [],
            "errors": [f"Unknown unit-action tool '{name}'."],
            "failed_calls": [f"Unknown unit-action tool '{name}'."],
        }
    try:
        return tool.invoke(args)
    except Exception as exc:
        return {
            "ok": False,
            "message": f"Tool '{name}' failed: {exc}",
            "current_position": None,
            "pen_state": "up",
            "action_count": 0,
            "warnings": [],
            "errors": [str(exc)],
            "failed_calls": [str(exc)],
        }


def _extract_tool_calls(response: Any) -> list[dict[str, Any]]:
    tool_calls = getattr(response, "tool_calls", None)
    if tool_calls:
        return [
            {
                "name": call.get("name"),
                "args": call.get("args") or {},
                "id": call.get("id"),
            }
            for call in tool_calls
        ]
    raw_calls = getattr(response, "additional_kwargs", {}).get("tool_calls", [])
    normalized = []
    for call in raw_calls:
        function = call.get("function", {})
        arguments = function.get("arguments") or "{}"
        try:
            args = json.loads(arguments)
        except json.JSONDecodeError:
            args = {}
        normalized.append(
            {
                "name": function.get("name"),
                "args": args,
                "id": call.get("id"),
            }
        )
    return normalized


def _drawing_plan_from_toolset(
    command: str,
    toolset: UnitActionToolset,
    config: PlannerConfig,
    validation_ok: bool,
    extra_errors: list[str],
    extra_warnings: list[str],
    force_empty_actions: bool = False,
) -> DrawingPlan:
    builder = toolset.builder
    if builder is None:
        strokes = []
        actions = []
        builder_errors: list[str] = []
        builder_warnings: list[str] = []
        builder_failed_calls: list[str] = []
    else:
        strokes = list(builder.strokes)
        actions = [] if force_empty_actions else list(builder.actions)
        builder_errors = list(builder.errors)
        builder_warnings = list(builder.warnings)
        builder_failed_calls = list(builder.failed_calls)

    errors = [*builder_errors, *extra_errors]
    warnings = [*builder_warnings, *extra_warnings]
    return DrawingPlan(
        source_command=command,
        goal=_agentic_goal(config, warnings),
        strokes=strokes,
        actions=actions,
        diagnostics={
            "mode": "agentic_unit_action_tools",
            "validation_ok": validation_ok and not errors,
            "assumptions": [
                "The LLM selected the unit-action sequence through tool calls.",
            ],
            "warnings": warnings,
            "errors": errors,
            "failed_calls": builder_failed_calls,
            "requires_robot_feasibility_check": True,
            "note": PLANNER_SCOPE_NOTE,
        },
    )


def _agentic_goal(config: PlannerConfig, warnings: list[str]) -> NormalizedGoal:
    return NormalizedGoal(
        shape_type="custom",
        center=config.default_center,
        radius_m=None,
        side_length_m=None,
        size_m=config.default_shape_size_m,
        orientation_rad=0.0,
        letter=None,
        assumptions=["Agentic mode does not use deterministic ParsedGoal templates."],
        warnings=warnings,
    )
