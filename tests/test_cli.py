import json

from robot_drawing_planner import cli
from robot_drawing_planner.planner import plan_from_goal
from robot_drawing_planner.schemas import ParsedGoal


def test_cli_writes_json_plan_with_mocked_planner(tmp_path, monkeypatch):
    expected_plan = plan_from_goal(ParsedGoal(object_type="square", size=6, unit="cm"))

    def fake_plan_from_text(command, board=None):
        assert command == "draw a square"
        assert board.width_m == 0.5
        return expected_plan

    monkeypatch.setattr(cli, "plan_from_text", fake_plan_from_text)
    output = tmp_path / "plan.json"
    result = cli.main(
        [
            "draw",
            "a",
            "square",
            "-o",
            str(output),
            "--board-width-m",
            "0.5",
        ]
    )

    assert result == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["goal"]["kind"] == "square"
    assert payload["actions"][0]["action"] == "move_to_start"

