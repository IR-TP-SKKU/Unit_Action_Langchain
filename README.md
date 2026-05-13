# robot_drawing_planner

LangChain / LLM planner for **LLM-Assisted Robot Shape Drawing in Isaac Sim**.

## Project Scope

This package implements only the LangChain/LLM planner layer:

```text
natural language command -> ParsedGoal -> deterministic strokes -> primitive action JSON
```

It outputs robot-level primitive action JSON. It does **not** compute robot
motion, joint angles, IK, FK, Jacobians, Isaac Sim commands, trajectories, or
fake robot execution results.

The downstream robot/kinematics module owns frame transforms, trajectory
sampling, IK/Jacobian control, and Isaac Sim execution.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## API Key Setup

The live parser reads the API key from `OPENAI_API_KEY`. Do not put raw API keys
in this repository.

```bash
export OPENAI_API_KEY="..."
export OPENAI_MODEL="gpt-5-nano"
```

If `OPENAI_MODEL` is omitted, the default model is `gpt-5-nano`.

If the key is stored in `~/.zshrc`, run the CLI through zsh when needed:

```bash
zsh -ic 'robot-drawing-plan "Draw a circle with radius 5 cm" --pretty'
```

## CLI Examples

Live LangChain/OpenAI mode:

```bash
zsh -ic 'robot-drawing-plan "중앙에 반지름 5cm짜리 원을 그려줘" --pretty --out outputs/circle_plan.json'
```

Development/demo mode without OpenAI:

```bash
python -m robot_drawing_planner.cli "중앙에 한 변 10cm짜리 네모를 그려줘" --no-api --pretty --out examples/square_plan.json
```

Use `--out-only` to write only to a file without printing JSON to stdout.

## Output Schema

Shortened output example:

```json
{
  "schema_version": "1.0",
  "source_command": "Draw a circle with radius 5 cm",
  "goal": {
    "shape_type": "circle",
    "center": {"x": 0.0, "y": 0.0, "unit": "m"},
    "radius_m": 0.05,
    "side_length_m": null,
    "size_m": 0.1,
    "orientation_rad": 0.0,
    "letter": null,
    "frame": "board",
    "assumptions": [],
    "warnings": [
      "robot reachability and IK feasibility are not checked by the LangChain planner"
    ]
  },
  "strokes": [
    {
      "type": "arc",
      "stroke_id": "stroke_001",
      "center": {"x": 0.0, "y": 0.0, "unit": "m"},
      "radius_m": 0.05,
      "start_angle_rad": 0.0,
      "end_angle_rad": 6.283185307179586,
      "direction": "ccw"
    }
  ],
  "actions": [
    {
      "name": "move_to_start",
      "frame": "board",
      "stroke_id": "stroke_001",
      "params": {
        "target": {"x": 0.05, "y": 0.0, "z": 0.03, "unit": "m"},
        "hover_height_m": 0.03,
        "note": "free-space move; kinematics module converts board frame to base frame"
      }
    }
  ],
  "diagnostics": {
    "validation_ok": true,
    "assumptions": [],
    "warnings": [],
    "errors": [],
    "requires_robot_feasibility_check": true,
    "note": "This planner does not compute IK, joint angles, Jacobians, or Isaac Sim commands."
  }
}
```

Primitive action names are:

- `move_to_start`: free-space move to a hover point above the first drawing point.
- `align_pen_orientation`: symbolic orientation constraint normal to the board.
- `pen_down`: approach the board drawing surface.
- `draw_line`: line segment geometry in board-frame meters.
- `draw_arc`: circular arc geometry in board-frame meters/radians.
- `pen_up`: lift the pen after a stroke or contour.

`params` contains only planner-level geometric parameters and hints. It does not
contain IK solutions, joint commands, robot trajectories, or Isaac Sim commands.

## Handoff To Kinematics Teammate

- Board frame coordinates are in meters.
- Board frame origin is at the board center.
- The kinematics module must convert board frame coordinates to robot base frame.
- The kinematics module must handle the pen tip offset from the end-effector.
- The kinematics module must interpolate Cartesian waypoints for line and arc actions.
- The kinematics module must compute IK/Jacobian control and any robot feasibility checks.
- This planner only provides symbolic primitive actions and geometric parameters.

## Supported Shapes

- `circle`
- `square`
- `triangle`
- Letters: `A`, `H`, `L`, `T`, `O`

## Tests

Unit tests use fakes/mocks and do not require the live OpenAI API:

```bash
pytest -q
```

## Live Smoke Test

Run through zsh when `OPENAI_API_KEY` is exported in `~/.zshrc`:

```bash
zsh -ic 'robot-drawing-plan "중앙에 반지름 5cm짜리 원을 그려줘" --pretty'
```

