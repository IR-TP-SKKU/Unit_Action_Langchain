"""Smooth scalar time scaling functions."""

from __future__ import annotations


def _validate_duration(duration_s: float) -> float:
    duration = float(duration_s)
    if duration <= 0.0:
        raise ValueError("duration_s must be positive.")
    return duration


def _clamped_u(t: float, duration_s: float) -> float:
    duration = _validate_duration(duration_s)
    return min(max(float(t) / duration, 0.0), 1.0)


def cubic_time_scaling(t: float, duration_s: float) -> float:
    """Return ``s(t) = 3u^2 - 2u^3`` for ``u = t / duration_s``."""

    u = _clamped_u(t, duration_s)
    return 3.0 * u**2 - 2.0 * u**3


def cubic_time_scaling_derivative(t: float, duration_s: float) -> float:
    """Return ``ds/dt`` for cubic time scaling."""

    duration = _validate_duration(duration_s)
    u = _clamped_u(t, duration)
    return (6.0 * u - 6.0 * u**2) / duration


def cubic_time_scaling_second_derivative(t: float, duration_s: float) -> float:
    """Return ``d2s/dt2`` for cubic time scaling."""

    duration = _validate_duration(duration_s)
    u = _clamped_u(t, duration)
    return (6.0 - 12.0 * u) / (duration**2)
