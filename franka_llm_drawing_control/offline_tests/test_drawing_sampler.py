import json
from pathlib import Path

import numpy as np
import pytest

from franka_llm_drawing.llm_bridge import load_plan_json
from franka_llm_drawing.trajectory import sample_drawing_plan

ROOT = Path(__file__).resolve().parents[1]


def test_action_order_produces_samples_for_circle() -> None:
    plan = load_plan_json(ROOT / "examples" / "sample_plan_circle.json")

    samples = sample_drawing_plan(
        plan,
        dt=0.01,
        default_speed_m_s=0.03,
        hover_height_m=0.03,
        draw_height_m=0.0,
        default_normal_force_n=1.0,
    )

    assert len(samples) > 10
    assert np.allclose(samples[0].position, np.array([0.05, 0.0, 0.03]))
    assert np.allclose(samples[-1].position[2], 0.03)
    assert any(sample.source_action == "draw_arc" for sample in samples)


def test_pen_down_and_pen_up_contact_flags() -> None:
    plan = load_plan_json(ROOT / "examples" / "sample_plan_circle.json")

    samples = sample_drawing_plan(plan, 0.01, 0.03, 0.03, 0.0, 1.0)
    pen_down_samples = [sample for sample in samples if sample.source_action == "pen_down"]
    pen_up_samples = [sample for sample in samples if sample.source_action == "pen_up"]

    assert pen_down_samples[-1].pen_contact_desired is True
    assert pen_up_samples[0].pen_contact_desired is True
    assert pen_up_samples[-1].pen_contact_desired is False


def test_invalid_action_order_raises() -> None:
    bad_plan = {
        "actions": [
            {
                "name": "draw_line",
                "frame": "board",
                "params": {
                    "start": {"x": 0.0, "y": 0.0, "z": 0.0, "unit": "m"},
                    "end": {"x": 0.1, "y": 0.0, "z": 0.0, "unit": "m"},
                },
            }
        ]
    }

    with pytest.raises(ValueError, match="current position"):
        sample_drawing_plan(bad_plan, 0.01, 0.03, 0.03, 0.0)


def test_square_plan_closes_path() -> None:
    with (ROOT / "examples" / "sample_plan_line_square.json").open(encoding="utf-8") as f:
        plan = json.load(f)

    samples = sample_drawing_plan(plan, 0.02, 0.03, 0.03, 0.0, 1.0)
    draw_samples = [sample for sample in samples if sample.source_action == "draw_line"]

    assert np.allclose(draw_samples[0].position, np.array([-0.05, -0.05, 0.0]))
    assert np.allclose(draw_samples[-1].position, np.array([-0.05, -0.05, 0.0]))
