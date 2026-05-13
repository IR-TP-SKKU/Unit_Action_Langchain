import importlib
import sys

from robot_drawing_planner.agent_events import make_event


def import_demo_app_fresh(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    sys.modules.pop("robot_drawing_planner.demo_app", None)
    return importlib.import_module("robot_drawing_planner.demo_app")


def test_import_demo_app_without_openai_api_key(monkeypatch):
    module = import_demo_app_fresh(monkeypatch)

    assert module.MISSING_API_KEY_MESSAGE == (
        "OPENAI_API_KEY is not set. Make sure it is exported in ~/.zshrc."
    )


def test_event_to_streamlit_message_formats_tool_call(monkeypatch):
    module = import_demo_app_fresh(monkeypatch)
    event = make_event(
        event_index=0,
        event_type="tool_call",
        message="Calling move_to_start",
        tool_name="move_to_start",
        tool_args={"x": 0.0, "y": 0.0},
    )

    role, markdown = module.event_to_streamlit_message(event)

    assert role == "assistant"
    assert "move_to_start" in markdown
    assert '"x": 0.0' in markdown


def test_event_to_streamlit_message_formats_user_request(monkeypatch):
    module = import_demo_app_fresh(monkeypatch)
    event = make_event(
        event_index=0,
        event_type="user_request",
        message="draw a square",
    )

    role, markdown = module.event_to_streamlit_message(event)

    assert role == "user"
    assert "draw a square" in markdown


def test_load_bytes_reads_temp_file(tmp_path, monkeypatch):
    module = import_demo_app_fresh(monkeypatch)
    path = tmp_path / "payload.bin"
    path.write_bytes(b"demo-bytes")

    assert module.load_bytes(path) == b"demo-bytes"


def test_main_function_exists(monkeypatch):
    module = import_demo_app_fresh(monkeypatch)

    assert callable(module.main)


def test_demo_app_import_does_not_execute_planner(monkeypatch):
    import robot_drawing_planner.demo_core as demo_core

    def fail_if_called(*args, **kwargs):
        raise AssertionError("planner should not run during demo_app import")

    monkeypatch.setattr(demo_core, "run_demo_request", fail_if_called)
    module = import_demo_app_fresh(monkeypatch)

    assert module.run_demo_request is fail_if_called
