import numpy as np

from franka_llm_drawing.trajectory.time_scaling import (
    cubic_time_scaling,
    cubic_time_scaling_derivative,
)


def test_cubic_scaling_endpoints() -> None:
    assert cubic_time_scaling(0.0, 2.0) == 0.0
    assert cubic_time_scaling(2.0, 2.0) == 1.0


def test_cubic_scaling_is_monotonic() -> None:
    values = [cubic_time_scaling(t, 1.0) for t in np.linspace(0.0, 1.0, 51)]

    assert all(a <= b for a, b in zip(values, values[1:]))


def test_cubic_derivative_zero_at_endpoints() -> None:
    assert np.isclose(cubic_time_scaling_derivative(0.0, 1.0), 0.0)
    assert np.isclose(cubic_time_scaling_derivative(1.0, 1.0), 0.0)
