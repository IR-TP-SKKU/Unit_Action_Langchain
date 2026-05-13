"""Streamlit chatbot demo for the agentic robot drawing planner.

The app visualizes planned board-frame primitive paths only. It does not
execute a robot or compute IK, FK, Jacobians, joint commands, torque, dynamics,
or Isaac Sim execution.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import streamlit as st

from robot_drawing_planner.agent_events import (
    AgentRunEvent,
    event_to_chat_markdown,
    make_event,
)
from robot_drawing_planner.demo_core import run_demo_request

MISSING_API_KEY_MESSAGE = (
    "OPENAI_API_KEY is not set. Make sure it is exported in ~/.zshrc."
)


def load_bytes(path: str | Path) -> bytes:
    """Read a file as bytes for Streamlit downloads and images."""

    return Path(path).read_bytes()


def event_to_streamlit_message(event: AgentRunEvent) -> tuple[str, str]:
    """Return a Streamlit chat role and markdown body for an event."""

    role = "user" if event.event_type == "user_request" else "assistant"
    return role, event_to_chat_markdown(event)


def render_event(event: AgentRunEvent) -> None:
    """Render one sanitized agent event as a chat/timeline message."""

    role, markdown = event_to_streamlit_message(event)
    with st.chat_message(role):
        if event.event_type == "tool_result" and event.ok is False:
            st.error(markdown)
        else:
            st.markdown(markdown)


def main() -> None:
    """Run the Streamlit chatbot demo."""

    st.set_page_config(page_title="LLM Robot Drawing Planner Demo", layout="wide")
    st.title("LLM Robot Drawing Planner Demo")

    with st.sidebar:
        mode = st.selectbox("Mode", ["agentic", "template", "no-api"], index=0)
        max_steps = st.number_input("Max steps", min_value=1, max_value=100, value=30)
        show_pen_up_moves = st.checkbox("Show pen-up moves", value=False)
        show_raw_json = st.checkbox("Show raw JSON", value=False)
        out_dir = st.text_input("Output directory", value="outputs/demo")
        st.info(
            "This demo visualizes planned primitive paths only. "
            "It does not execute a robot."
        )

    history = st.session_state.setdefault("demo_chat_history", [])
    for record in history:
        _render_history_record(record, show_raw_json=show_raw_json)

    command = st.chat_input("Describe what the robot should draw")
    if not command:
        return

    user_record = {"kind": "user", "content": command}
    history.append(user_record)
    _render_history_record(user_record, show_raw_json=show_raw_json)

    if mode == "agentic" and not os.getenv("OPENAI_API_KEY"):
        error_event = make_event(
            event_index=0,
            event_type="error",
            message=MISSING_API_KEY_MESSAGE,
            ok=False,
            metadata={"mode": mode},
        )
        event_record = {"kind": "event", "event": error_event.model_dump(mode="json")}
        history.append(event_record)
        _render_history_record(event_record, show_raw_json=show_raw_json)
        return

    try:
        with st.spinner("Planning symbolic unit actions..."):
            result = run_demo_request(
                command,
                mode=mode,
                out_dir=out_dir,
                max_steps=int(max_steps),
                create_plot=True,
                show_pen_up_moves=show_pen_up_moves,
            )
    except Exception as exc:
        error_event = make_event(
            event_index=0,
            event_type="error",
            message=str(exc),
            ok=False,
            metadata={"mode": mode},
        )
        event_record = {"kind": "event", "event": error_event.model_dump(mode="json")}
        history.append(event_record)
        _render_history_record(event_record, show_raw_json=show_raw_json)
        return

    for event in result.events:
        event_record = {"kind": "event", "event": event.model_dump(mode="json")}
        history.append(event_record)
        _render_history_record(event_record, show_raw_json=show_raw_json)

    result_record = {
        "kind": "result",
        "plan_json_path": result.plan_json_path,
        "plot_png_path": result.plot_png_path,
        "plan": result.plan.model_dump(mode="json"),
    }
    history.append(result_record)
    _render_history_record(result_record, show_raw_json=show_raw_json)


def _render_history_record(record: dict[str, Any], show_raw_json: bool) -> None:
    kind = record.get("kind")
    if kind == "user":
        with st.chat_message("user"):
            st.markdown(str(record.get("content", "")))
    elif kind == "event":
        render_event(AgentRunEvent.model_validate(record["event"]))
    elif kind == "result":
        _render_result_record(record, show_raw_json=show_raw_json)


def _render_result_record(record: dict[str, Any], show_raw_json: bool) -> None:
    plan_json_path = record.get("plan_json_path")
    plot_png_path = record.get("plot_png_path")
    plan_payload = record.get("plan") or {}

    with st.chat_message("assistant"):
        if plot_png_path and Path(plot_png_path).exists():
            st.image(load_bytes(plot_png_path), caption="Planned board-frame path")
        if plan_json_path and Path(plan_json_path).exists():
            st.download_button(
                "Download plan JSON",
                data=load_bytes(plan_json_path),
                file_name=Path(plan_json_path).name,
                mime="application/json",
            )
        if plot_png_path and Path(plot_png_path).exists():
            st.download_button(
                "Download plot PNG",
                data=load_bytes(plot_png_path),
                file_name=Path(plot_png_path).name,
                mime="image/png",
            )
        if show_raw_json:
            with st.expander("DrawingPlan JSON"):
                st.json(plan_payload)
        st.caption(
            "Planned primitive path only; no robot execution, IK, FK, Jacobians, "
            "joint commands, torque, dynamics, or Isaac Sim execution."
        )


if __name__ == "__main__":
    main()
