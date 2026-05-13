import json

import pytest

from robot_drawing_planner import cli


def test_cli_no_api_square_exits_zero_and_prints_valid_json(capsys):
    result = cli.main(["중앙에 한 변 10cm짜리 네모를 그려줘", "--no-api"])
    captured = capsys.readouterr()

    assert result == 0
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload["goal"]["shape_type"] == "square"
    assert payload["goal"]["side_length_m"] == 0.1
    assert payload["actions"][0]["name"] == "move_to_start"


def test_cli_no_api_out_writes_file_and_prints_stdout(tmp_path, capsys):
    output = tmp_path / "outputs" / "circle.json"
    result = cli.main(
        [
            "Draw a circle with radius 5 cm",
            "--no-api",
            "--out",
            str(output),
            "--pretty",
        ]
    )
    captured = capsys.readouterr()

    assert result == 0
    assert output.exists()
    file_payload = json.loads(output.read_text(encoding="utf-8"))
    stdout_payload = json.loads(captured.out)
    assert file_payload == stdout_payload
    assert file_payload["goal"]["shape_type"] == "circle"
    assert file_payload["actions"][3]["name"] == "draw_arc"


def test_cli_no_api_out_only_suppresses_stdout(tmp_path, capsys):
    output = tmp_path / "plan.json"
    result = cli.main(["세모 변 길이 10cm", "--no-api", "--out", str(output), "--out-only"])
    captured = capsys.readouterr()

    assert result == 0
    assert captured.out == ""
    assert output.exists()
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["goal"]["shape_type"] == "triangle"


def test_cli_no_api_korean_letter_a(tmp_path, capsys):
    output = tmp_path / "letter_A.json"
    result = cli.main(
        ["중앙에 크기 10cm인 글자 A를 써줘", "--no-api", "--out", str(output), "--out-only"]
    )
    captured = capsys.readouterr()

    assert result == 0
    assert captured.out == ""
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["goal"]["shape_type"] == "letter"
    assert payload["goal"]["letter"] == "A"
    assert payload["actions"]


def test_cli_mode_template_with_no_api_still_uses_demo_fallback(capsys):
    result = cli.main(["Draw a circle with radius 5 cm", "--mode", "template", "--no-api"])
    captured = capsys.readouterr()

    assert result == 0
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload["goal"]["shape_type"] == "circle"
    assert payload["actions"][3]["name"] == "draw_arc"


def test_cli_default_mode_missing_live_key_is_nonzero_and_no_secret(monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = cli.main(["Draw a circle with radius 5 cm"])
    captured = capsys.readouterr()

    assert result != 0
    assert "OPENAI_API_KEY is not set" in captured.err
    assert "sk-" not in captured.err
    assert captured.out == ""


def test_cli_help_mentions_agentic_unit_action_tool_planner(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])
    captured = capsys.readouterr()

    assert exc.value.code == 0
    help_text = " ".join(captured.out.split())
    assert "Default mode uses LLM agentic planning over Unit Action tools" in help_text
    assert "template" in help_text
    assert "ParsedGoal/template compiler baseline" in help_text
    assert "development/testing only" in help_text
