"""Pure math hybrid position-force controller."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from franka_llm_drawing.controllers.nullspace import compute_nullspace_torque
from franka_llm_drawing.controllers.pose_error import orientation_error_axis_angle


@dataclass(frozen=True)
class HybridPositionForceConfig:
    kp_pos_xy: float = 150.0
    kd_pos_xy: float = 20.0
    kp_z_when_not_in_contact: float = 100.0
    kd_z_when_not_in_contact: float = 10.0
    desired_normal_force_n: float = 1.0
    kp_force: float = 5.0
    ki_force: float = 0.0
    kd_force: float = 0.0
    max_force_cmd_n: float = 20.0
    max_torque_nm: float = 30.0
    orientation_kp: float = 20.0
    orientation_kd: float = 3.0
    nullspace_kp: float = 5.0
    nullspace_kd: float = 1.0
    use_gravity_compensation: bool = True
    normal_force_positive_when_pressing: bool = True
    contact_threshold_n: float = 0.1
    nullspace_damping: float = 0.05


@dataclass(frozen=True)
class HybridControllerDiagnostics:
    task_wrench: np.ndarray
    motion_wrench: np.ndarray
    force_wrench: np.ndarray
    orientation_wrench: np.ndarray
    tau_task: np.ndarray
    tau_null: np.ndarray
    tau_cmd_raw: np.ndarray
    tau_cmd_clipped: np.ndarray
    normal_force_error: float
    contact_active: bool


class HybridPositionForceController:
    """Task-space controller that splits tangent motion and normal force."""

    def __init__(self, config: HybridPositionForceConfig | None = None) -> None:
        self.config = config or HybridPositionForceConfig()
        self._force_error_integral = 0.0
        self._previous_force_error: float | None = None

    def reset(self) -> None:
        """Clear integral and derivative history."""

        self._force_error_integral = 0.0
        self._previous_force_error = None

    def compute_torque(
        self,
        q: np.ndarray,
        qd: np.ndarray,
        jacobian_6xn: np.ndarray,
        p_des_base: np.ndarray,
        R_des_base: np.ndarray,
        p_cur_base: np.ndarray,
        R_cur_base: np.ndarray,
        v_cur_base: np.ndarray | None,
        w_cur_base: np.ndarray | None,
        board_normal_base: np.ndarray,
        measured_normal_force_n: float | None,
        gravity_torque: np.ndarray | None = None,
        q_nominal: np.ndarray | None = None,
        dt: float = 0.01,
        contact_active: bool = False,
    ) -> tuple[np.ndarray, HybridControllerDiagnostics]:
        """Compute a joint torque command from task-space errors."""

        q_arr = _vector(q, "q")
        qd_arr = _vector(qd, "qd")
        if q_arr.shape != qd_arr.shape:
            raise ValueError("q and qd must have the same shape.")
        J = np.asarray(jacobian_6xn, dtype=float)
        if J.shape != (6, q_arr.size):
            raise ValueError(f"jacobian_6xn must have shape (6, {q_arr.size}).")

        p_des = _vector3(p_des_base, "p_des_base")
        p_cur = _vector3(p_cur_base, "p_cur_base")
        R_des = _rotation(R_des_base, "R_des_base")
        R_cur = _rotation(R_cur_base, "R_cur_base")
        normal = _normalized(board_normal_base, "board_normal_base")
        tangent_projector = np.eye(3) - np.outer(normal, normal)
        v_cur = np.zeros(3) if v_cur_base is None else _vector3(v_cur_base, "v_cur_base")
        w_cur = np.zeros(3) if w_cur_base is None else _vector3(w_cur_base, "w_cur_base")

        measured_force = 0.0 if measured_normal_force_n is None else float(measured_normal_force_n)
        effective_contact = bool(
            contact_active or measured_force >= self.config.contact_threshold_n
        )

        position_error = p_des - p_cur
        tangent_error = tangent_projector @ position_error
        tangent_velocity = tangent_projector @ v_cur
        motion_linear = (
            self.config.kp_pos_xy * tangent_error - self.config.kd_pos_xy * tangent_velocity
        )

        normal_motion = np.zeros(3)
        force_linear = np.zeros(3)
        force_error = self.config.desired_normal_force_n - measured_force

        if effective_contact:
            self._force_error_integral += force_error * max(float(dt), 0.0)
            force_derivative = 0.0
            if self._previous_force_error is not None and dt > 0.0:
                force_derivative = (force_error - self._previous_force_error) / float(dt)
            self._previous_force_error = force_error
            force_mag = (
                self.config.desired_normal_force_n
                + self.config.kp_force * force_error
                + self.config.ki_force * self._force_error_integral
                + self.config.kd_force * force_derivative
            )
            force_mag = float(
                np.clip(force_mag, -self.config.max_force_cmd_n, self.config.max_force_cmd_n)
            )
            press_direction = -normal if self.config.normal_force_positive_when_pressing else normal
            force_linear = force_mag * press_direction
        else:
            self._previous_force_error = None
            normal_error = float(position_error @ normal)
            normal_velocity = float(v_cur @ normal)
            normal_motion = (
                self.config.kp_z_when_not_in_contact * normal_error
                - self.config.kd_z_when_not_in_contact * normal_velocity
            ) * normal

        orientation_error = orientation_error_axis_angle(R_des, R_cur)
        orientation_linear = (
            self.config.orientation_kp * orientation_error - self.config.orientation_kd * w_cur
        )

        motion_wrench = np.concatenate([motion_linear + normal_motion, np.zeros(3)])
        force_wrench = np.concatenate([force_linear, np.zeros(3)])
        orientation_wrench = np.concatenate([np.zeros(3), orientation_linear])
        task_wrench = motion_wrench + force_wrench + orientation_wrench
        tau_task = J.T @ task_wrench

        if q_nominal is None:
            tau_null = np.zeros_like(q_arr)
        else:
            tau_null, _ = compute_nullspace_torque(
                q_arr,
                qd_arr,
                J,
                _vector(q_nominal, "q_nominal"),
                kp=self.config.nullspace_kp,
                kd=self.config.nullspace_kd,
                damping=self.config.nullspace_damping,
                max_torque_nm=self.config.max_torque_nm,
            )

        tau_cmd_raw = tau_task + tau_null
        if self.config.use_gravity_compensation and gravity_torque is not None:
            gravity = _vector(gravity_torque, "gravity_torque")
            if gravity.shape != q_arr.shape:
                raise ValueError("gravity_torque must have the same shape as q.")
            tau_cmd_raw = tau_cmd_raw + gravity

        limit = abs(float(self.config.max_torque_nm))
        tau_cmd_clipped = np.clip(tau_cmd_raw, -limit, limit)
        diagnostics = HybridControllerDiagnostics(
            task_wrench=task_wrench,
            motion_wrench=motion_wrench,
            force_wrench=force_wrench,
            orientation_wrench=orientation_wrench,
            tau_task=tau_task,
            tau_null=tau_null,
            tau_cmd_raw=tau_cmd_raw,
            tau_cmd_clipped=tau_cmd_clipped,
            normal_force_error=float(force_error),
            contact_active=effective_contact,
        )
        return tau_cmd_clipped, diagnostics


def _vector(value: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a vector, got shape {arr.shape}.")
    return arr


def _vector3(value: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {arr.shape}.")
    return arr


def _rotation(value: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.shape != (3, 3):
        raise ValueError(f"{name} must have shape (3, 3), got {arr.shape}.")
    return arr


def _normalized(value: np.ndarray, name: str) -> np.ndarray:
    arr = _vector3(value, name)
    norm = float(np.linalg.norm(arr))
    if norm < 1e-12:
        raise ValueError(f"{name} must be nonzero.")
    return arr / norm
