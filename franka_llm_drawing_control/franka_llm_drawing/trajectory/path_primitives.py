"""Sampling for line and circular arc drawing primitives."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from franka_llm_drawing.trajectory.time_scaling import (
    cubic_time_scaling,
    cubic_time_scaling_derivative,
    cubic_time_scaling_second_derivative,
)


@dataclass(frozen=True)
class PoseSample:
    """A desired pen-tip pose sample.

    Positions and velocities are expressed in the local path frame used by the
    caller. For samples returned by the drawing sampler, that frame is the board
    frame. Rotation maps from tip frame to the same parent frame.
    """

    t: float
    position: np.ndarray
    rotation: np.ndarray
    linear_velocity: np.ndarray | None = None
    angular_velocity: np.ndarray | None = None
    linear_acceleration: np.ndarray | None = None
    pen_contact_desired: bool = False
    desired_normal_force_n: float | None = None
    source_action: str | None = None
    stroke_id: str | None = None


def sample_line(
    start: np.ndarray,
    end: np.ndarray,
    speed_m_s: float,
    dt: float,
    start_time: float = 0.0,
    rotation: np.ndarray | None = None,
    pen_contact_desired: bool = True,
    desired_normal_force_n: float | None = None,
    source_action: str | None = None,
    stroke_id: str | None = None,
) -> list[PoseSample]:
    """Sample a straight-line path using cubic time scaling."""

    start_arr = _vector3(start, "start")
    end_arr = _vector3(end, "end")
    speed = _positive(speed_m_s, "speed_m_s")
    dt_s = _positive(dt, "dt")
    rotation_arr = _rotation(rotation)
    delta = end_arr - start_arr
    length = float(np.linalg.norm(delta))
    if length < 1e-12:
        return [
            PoseSample(
                t=float(start_time),
                position=start_arr.copy(),
                rotation=rotation_arr.copy(),
                linear_velocity=np.zeros(3),
                linear_acceleration=np.zeros(3),
                pen_contact_desired=pen_contact_desired,
                desired_normal_force_n=desired_normal_force_n,
                source_action=source_action,
                stroke_id=stroke_id,
            )
        ]

    duration = max(length / speed, dt_s)
    samples: list[PoseSample] = []
    for tau in _sample_relative_times(duration, dt_s):
        s = cubic_time_scaling(tau, duration)
        sd = cubic_time_scaling_derivative(tau, duration)
        sdd = cubic_time_scaling_second_derivative(tau, duration)
        samples.append(
            PoseSample(
                t=float(start_time + tau),
                position=start_arr + s * delta,
                rotation=rotation_arr.copy(),
                linear_velocity=sd * delta,
                linear_acceleration=sdd * delta,
                pen_contact_desired=pen_contact_desired,
                desired_normal_force_n=desired_normal_force_n,
                source_action=source_action,
                stroke_id=stroke_id,
            )
        )
    return samples


def sample_arc(
    center: np.ndarray,
    radius_m: float,
    start_angle_rad: float,
    end_angle_rad: float,
    direction: str,
    speed_m_s: float,
    dt: float,
    start_time: float = 0.0,
    rotation: np.ndarray | None = None,
    pen_contact_desired: bool = True,
    desired_normal_force_n: float | None = None,
    source_action: str | None = None,
    stroke_id: str | None = None,
) -> list[PoseSample]:
    """Sample a circular arc in the x-y plane around ``center``."""

    center_arr = _vector3(center, "center")
    radius = _positive(radius_m, "radius_m")
    speed = _positive(speed_m_s, "speed_m_s")
    dt_s = _positive(dt, "dt")
    rotation_arr = _rotation(rotation)
    delta_theta = normalized_arc_delta(start_angle_rad, end_angle_rad, direction)
    arc_length = abs(radius * delta_theta)
    if arc_length < 1e-12:
        theta = float(start_angle_rad)
        position = center_arr + np.array([radius * math.cos(theta), radius * math.sin(theta), 0.0])
        return [
            PoseSample(
                t=float(start_time),
                position=position,
                rotation=rotation_arr.copy(),
                linear_velocity=np.zeros(3),
                linear_acceleration=np.zeros(3),
                pen_contact_desired=pen_contact_desired,
                desired_normal_force_n=desired_normal_force_n,
                source_action=source_action,
                stroke_id=stroke_id,
            )
        ]

    duration = max(arc_length / speed, dt_s)
    samples: list[PoseSample] = []
    for tau in _sample_relative_times(duration, dt_s):
        s = cubic_time_scaling(tau, duration)
        sd = cubic_time_scaling_derivative(tau, duration)
        sdd = cubic_time_scaling_second_derivative(tau, duration)
        theta = float(start_angle_rad) + s * delta_theta
        theta_dot = sd * delta_theta
        theta_ddot = sdd * delta_theta
        radial = np.array([math.cos(theta), math.sin(theta), 0.0])
        tangent = np.array([-math.sin(theta), math.cos(theta), 0.0])
        position = center_arr + radius * radial
        velocity = radius * theta_dot * tangent
        acceleration = radius * theta_ddot * tangent - radius * theta_dot**2 * radial
        samples.append(
            PoseSample(
                t=float(start_time + tau),
                position=position,
                rotation=rotation_arr.copy(),
                linear_velocity=velocity,
                linear_acceleration=acceleration,
                pen_contact_desired=pen_contact_desired,
                desired_normal_force_n=desired_normal_force_n,
                source_action=source_action,
                stroke_id=stroke_id,
            )
        )
    return samples


def normalized_arc_delta(
    start_angle_rad: float,
    end_angle_rad: float,
    direction: str,
) -> float:
    """Return signed angular travel for the requested arc direction."""

    theta0 = float(start_angle_rad)
    theta1 = float(end_angle_rad)
    raw_delta = theta1 - theta0
    if direction == "ccw":
        if math.isclose(raw_delta, 0.0, abs_tol=1e-12):
            return 2.0 * math.pi
        while raw_delta < 0.0:
            raw_delta += 2.0 * math.pi
        return raw_delta
    if direction == "cw":
        if math.isclose(raw_delta, 0.0, abs_tol=1e-12):
            return -2.0 * math.pi
        while raw_delta > 0.0:
            raw_delta -= 2.0 * math.pi
        return raw_delta
    raise ValueError("direction must be 'cw' or 'ccw'.")


def _sample_relative_times(duration: float, dt: float) -> np.ndarray:
    count = max(2, int(math.ceil(duration / dt)) + 1)
    return np.linspace(0.0, duration, count)


def _vector3(value: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {arr.shape}.")
    return arr


def _rotation(value: np.ndarray | None) -> np.ndarray:
    if value is None:
        return np.eye(3)
    arr = np.asarray(value, dtype=float)
    if arr.shape != (3, 3):
        raise ValueError(f"rotation must have shape (3, 3), got {arr.shape}.")
    return arr


def _positive(value: float, name: str) -> float:
    value_f = float(value)
    if value_f <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return value_f
