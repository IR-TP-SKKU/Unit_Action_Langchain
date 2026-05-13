"""Agentic LangChain planner that builds plans through unit-action tools."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from robot_drawing_planner.action_tools import UnitActionToolset
from robot_drawing_planner.config import DEFAULT_CONFIG, PlannerConfig
from robot_drawing_planner.llm_client import get_llm
from robot_drawing_planner.schemas import DrawingPlan, NormalizedGoal, Point2D

AGENT_SYSTEM_PROMPT = """You are a robot drawing action planner.

You must plan by calling the available Unit Action tools.
You must not output a final plan directly.
You must call tools step by step.
You must call check_plan before finish_plan.
Use only line and arc primitives.
For complex shapes, approximate using lines and arcs.

The tools append symbolic robot primitive actions only. They do not move the robot.
Do not compute IK, FK, Jacobians, joint commands, joint velocities, trajectory
samples, or Isaac Sim commands.
"""

PLANNER_SCOPE_NOTE = (
    "This agentic planner outputs primitive action JSON only; it does not compute "
    "IK, FK, Jacobians, joint commands, trajectory samples, or Isaac Sim commands."
)


def plan_drawing_agentic(
    command: str,
    config: PlannerConfig | None = None,
    llm: Any | None = None,
    max_steps: int = 30,
) -> DrawingPlan:
    """Plan by letting an LLM sequentially call symbolic unit-action tools."""

    planner_config = config or DEFAULT_CONFIG
    toolset = UnitActionToolset(config=planner_config)
    tools = toolset.tools()
    model = llm if llm is not None else get_llm()
    bound_model = model.bind_tools(tools) if hasattr(model, "bind_tools") else model

    messages: list[Any] = [
        SystemMessage(content=AGENT_SYSTEM_PROMPT),
        HumanMessage(content=command),
    ]
    last_finish_feedback: dict[str, Any] | None = None
    errors: list[str] = []
    warnings: list[str] = []

    for _step in range(max_steps):
        response = bound_model.invoke(messages)
        messages.append(response)
        tool_calls = _extract_tool_calls(response)
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
            feedback = _execute_tool_call(toolset, name, args)
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
                    return _drawing_plan_from_toolset(
                        command=command,
                        toolset=toolset,
                        config=planner_config,
                        validation_ok=True,
                        extra_errors=[],
                        extra_warnings=[],
                    )
                errors = [feedback.get("message", "finish_plan failed")]

    errors.append(f"Agentic planning failed: max_steps {max_steps} exceeded.")
    if last_finish_feedback and last_finish_feedback.get("message"):
        warnings.append(f"Last finish_plan feedback: {last_finish_feedback['message']}")
    return _drawing_plan_from_toolset(
        command=command,
        toolset=toolset,
        config=planner_config,
        validation_ok=False,
        extra_errors=errors,
        extra_warnings=warnings,
        force_empty_actions=True,
    )


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
