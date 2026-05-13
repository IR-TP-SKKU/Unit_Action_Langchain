# robot_drawing_planner

LangChain / LLM planner for **LLM-Assisted Robot Shape Drawing in Isaac Sim**.

This package stops at robot-level primitive JSON. It does not generate joint
angles, IK/FK/Jacobian commands, Isaac Sim commands, trajectories, or robot
execution code.

## Scope

Natural language drawing command:

```text
Draw a 10 cm square at the center of the board.
```

Becomes:

```text
structured drawing goal -> deterministic stroke geometry -> primitive action JSON
```

The downstream robot/kinematics module owns frame transforms, trajectory
sampling, IK, Jacobian control, and Isaac Sim execution.

## Install

```bash
python -m pip install -e ".[dev]"
```

The live parser reads the API key only from `OPENAI_API_KEY`. Do not place the
key in this repository.

```bash
export OPENAI_API_KEY=...
export OPENAI_MODEL=gpt-5-nano
```

If `OPENAI_MODEL` is omitted, the default is `gpt-5-nano`.

## CLI

Use `zsh -ic` for live API calls so `~/.zshrc` can load the key:

```bash
zsh -ic 'robot-drawing-plan "draw a 10 cm square at the center" --out plan.json'
```

For development and tests, `--no-api` uses a small deterministic demo parser and
does not call OpenAI:

```bash
python -m robot_drawing_planner.cli "Draw a circle with radius 5 cm" --no-api --out outputs/circle.json --pretty
```

The CLI writes a JSON plan and does not execute the robot.

## Supported Goals

- Shapes: `circle`, `square`, `triangle`
- Letters: `A`, `H`, `L`, `T`, `O`
- Units: `m`, `cm`, `mm`
- Board frame: 2D board coordinates in meters, origin at board center

## Primitive Actions

The handoff JSON contains only these primitive actions:

- `move_to_start`
- `align_pen_orientation`
- `pen_down`
- `draw_line`
- `draw_arc`
- `pen_up`

Example shape of the handoff:

```json
{
  "schema_version": "1.0",
  "coordinate_frame": "drawing_board_2d_m",
  "board": {
    "width_m": 0.4,
    "height_m": 0.3,
    "origin": "center"
  },
  "goal": {
    "kind": "square",
    "letter": null,
    "size_m": 0.1,
    "center": {
      "x_m": 0.0,
      "y_m": 0.0
    }
  },
  "actions": []
}
```

## Validation

The planner validates:

- supported shape
- positive size
- supported unit conversion
- board boundary containment
- unsupported letters
- absence of low-level robot fields such as IK, FK, Jacobian, joint angles,
  trajectories, or Isaac Sim commands

## Tests

Unit tests use fakes/mocks and do not call the live OpenAI API:

```bash
pytest -q
```

Live API smoke testing is intentionally separate and skipped by default. Run it
only through zsh when `OPENAI_API_KEY` is exported in the user's zsh
environment:

```bash
zsh -ic 'RUN_LIVE_OPENAI_SMOKE=1 pytest -q -m live'
```
