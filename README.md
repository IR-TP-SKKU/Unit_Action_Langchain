# robot_drawing_planner

LangChain / LLM planner for **LLM-Assisted Robot Shape Drawing in Isaac Sim**.

## Project Scope

This package implements only the LangChain/LLM planner layer. The project
contribution is the **agentic planner over robot primitive actions**:

```text
natural language command
-> ChatOpenAI bound to Unit Action tools
-> multi-step tool calls
-> PlanBuilder state
-> primitive action JSON
```

It outputs robot-level primitive action JSON. It does **not** compute robot
motion, joint angles, IK, FK, Jacobians, Isaac Sim commands, trajectories,
trajectory samples, or fake robot execution results.

This module outputs primitive action JSON only; it does not compute IK, Jacobians, joint commands, FK, or Isaac Sim execution.

The downstream robot/kinematics module owns frame transforms, trajectory
sampling, IK/Jacobian control, and Isaac Sim execution.

## Planner Modes

There are two planning paths:

- **Agentic LLM planner**: the production/default mode. The LLM is bound to
  Unit Action tools such as `move_to_start`, `pen_down`, `draw_line_to`,
  `draw_arc`, `check_plan`, and `finish_plan`. The LLM must call these tools
  step by step, and each call appends one symbolic primitive action to
  `PlanBuilder`.
- **Template baseline**: the old ParsedGoal/template compiler. It is kept for
  baseline comparisons, fallback behavior, tests, and simple demos. It extracts
  shape parameters and deterministic Python code creates the template strokes.

Open-ended shapes such as a house, smiley face, or star approximation are
examples for the agentic LLM planner. They are **not** hard-coded as deterministic
geometry templates.

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

Live LangChain/OpenAI agentic Unit Action tool mode:

```bash
zsh -ic 'robot-drawing-plan "중앙에 반지름 5cm짜리 원을 그려줘" --pretty --out outputs/circle_plan.json'
```

Template baseline mode:

```bash
zsh -ic 'robot-drawing-plan "Draw a circle with radius 5 cm" --mode template --pretty'
```

Development/demo mode without OpenAI:

```bash
python -m robot_drawing_planner.cli "중앙에 한 변 10cm짜리 네모를 그려줘" --no-api --pretty --out examples/square_plan.json
```

Use `--out-only` to write only to a file without printing JSON to stdout.

## Open-Ended Agentic Examples

Prompt/tool-call examples are stored in `examples/agentic_tool_calls/`:

- `square_tool_plan.json`
- `circle_tool_plan.json`
- `letter_A_tool_plan.json`
- `house_tool_plan.json`
- `smiley_tool_plan.json`
- `star_approx_tool_plan.json`

For example, a "house" can be decomposed by the LLM into:

- body: a square-like closed line contour
- roof: a triangle-like line contour
- door: a smaller line contour on the body

That decomposition is represented as Unit Action tool calls. It is not a
deterministic compiler template in `geometry.py`.

## Output Schema

Shortened output example:

```json
{
  "schema_version": "1.0",
  "source_command": "Draw a circle with radius 5 cm",
  "goal": {
    "shape_type": "custom",
    "center": {"x": 0.0, "y": 0.0, "unit": "m"},
    "radius_m": 0.05,
    "side_length_m": null,
    "size_m": 0.1,
    "orientation_rad": 0.0,
    "letter": null,
    "frame": "board",
    "assumptions": [
      "Agentic mode does not use deterministic ParsedGoal templates."
    ],
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
    "mode": "agentic_unit_action_tools",
    "note": "This agentic planner outputs primitive action JSON only; it does not compute IK, FK, Jacobians, joint commands, trajectory samples, or Isaac Sim commands."
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
contain IK solutions, FK results, Jacobians, joint commands, robot trajectories,
trajectory samples, or Isaac Sim commands.

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
