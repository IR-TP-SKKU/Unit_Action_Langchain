import numpy as np

from franka_llm_drawing.controllers import (
    HybridPositionForceConfig,
    HybridPositionForceController,
)


def _jacobian() -> np.ndarray:
    J = np.zeros((6, 7))
    J[:6, :6] = np.eye(6)
    return J


def test_hybrid_torque_shape() -> None:
    controller = HybridPositionForceController()

    tau, diagnostics = controller.compute_torque(
        q=np.zeros(7),
        qd=np.zeros(7),
        jacobian_6xn=_jacobian(),
        p_des_base=np.array([0.1, 0.0, 0.0]),
        R_des_base=np.eye(3),
        p_cur_base=np.zeros(3),
        R_cur_base=np.eye(3),
        v_cur_base=None,
        w_cur_base=None,
        board_normal_base=np.array([0.0, 0.0, 1.0]),
        measured_normal_force_n=None,
    )

    assert tau.shape == (7,)
    assert diagnostics.task_wrench.shape == (6,)


def test_no_contact_mode_uses_z_position_behavior() -> None:
    controller = HybridPositionForceController()

    _, diagnostics = controller.compute_torque(
        q=np.zeros(7),
        qd=np.zeros(7),
        jacobian_6xn=_jacobian(),
        p_des_base=np.array([0.0, 0.0, -0.01]),
        R_des_base=np.eye(3),
        p_cur_base=np.zeros(3),
        R_cur_base=np.eye(3),
        v_cur_base=np.zeros(3),
        w_cur_base=np.zeros(3),
        board_normal_base=np.array([0.0, 0.0, 1.0]),
        measured_normal_force_n=0.0,
        contact_active=False,
    )

    assert diagnostics.contact_active is False
    assert diagnostics.motion_wrench[2] < 0.0
    assert np.allclose(diagnostics.force_wrench, np.zeros(6))


def test_contact_mode_uses_normal_force_error() -> None:
    controller = HybridPositionForceController()

    _, diagnostics = controller.compute_torque(
        q=np.zeros(7),
        qd=np.zeros(7),
        jacobian_6xn=_jacobian(),
        p_des_base=np.zeros(3),
        R_des_base=np.eye(3),
        p_cur_base=np.zeros(3),
        R_cur_base=np.eye(3),
        v_cur_base=np.zeros(3),
        w_cur_base=np.zeros(3),
        board_normal_base=np.array([0.0, 0.0, 1.0]),
        measured_normal_force_n=0.2,
        contact_active=True,
    )

    assert diagnostics.contact_active is True
    assert diagnostics.normal_force_error > 0.0
    assert diagnostics.force_wrench[2] < 0.0


def test_torque_clipping() -> None:
    controller = HybridPositionForceController(
        HybridPositionForceConfig(kp_pos_xy=1e6, max_torque_nm=0.5)
    )

    tau, diagnostics = controller.compute_torque(
        q=np.zeros(7),
        qd=np.zeros(7),
        jacobian_6xn=_jacobian(),
        p_des_base=np.array([1.0, 0.0, 0.0]),
        R_des_base=np.eye(3),
        p_cur_base=np.zeros(3),
        R_cur_base=np.eye(3),
        v_cur_base=np.zeros(3),
        w_cur_base=np.zeros(3),
        board_normal_base=np.array([0.0, 0.0, 1.0]),
        measured_normal_force_n=0.0,
    )

    assert np.max(np.abs(tau)) <= 0.5
    assert np.max(np.abs(diagnostics.tau_cmd_raw)) > 0.5
