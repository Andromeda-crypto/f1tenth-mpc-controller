"""Offline C3 validation; run with: python test_mpc_qp.py"""

from __future__ import annotations

from statistics import median

import numpy as np

from f1tenth_mpc.mpc_qp import MPCConfig, LinearMPCQP, linearize_discrete_kbm


def nonlinear_step(z: np.ndarray, u: np.ndarray, dt: float, wheelbase: float) -> np.ndarray:
    x, y, speed, yaw = z
    accel, steer = u
    return z + dt * np.array(
        [speed * np.cos(yaw), speed * np.sin(yaw), accel, speed * np.tan(steer) / wheelbase]
    )


def main() -> None:
    cfg = MPCConfig(horizon=8, dt=0.1)
    wheelbase = 0.3302
    nominal_z = np.array([0.0, 0.0, 2.0, 0.0])
    nominal_u = np.array([0.0, 0.0])
    a, b, c = linearize_discrete_kbm(nominal_z, nominal_u, dt=cfg.dt, wheelbase=wheelbase)

    # C2 numerical check: affine model equals the nonlinear model at its operating point.
    affine_next = a @ nominal_z + b @ nominal_u + c
    exact_next = nonlinear_step(nominal_z, nominal_u, cfg.dt, wheelbase)
    np.testing.assert_allclose(affine_next, exact_next, atol=1e-12)

    n = cfg.horizon
    a_seq = np.repeat(a[None, :, :], n, axis=0)
    b_seq = np.repeat(b[None, :, :], n, axis=0)
    c_seq = np.repeat(c[None, :], n, axis=0)
    z_ref = np.zeros((4, n + 1))
    z_ref[0] = np.arange(n + 1) * cfg.dt * nominal_z[2]
    z_ref[2] = nominal_z[2]
    z0 = np.array([0.0, 0.6, 2.0, 0.0])

    mpc = LinearMPCQP(cfg)
    results = [mpc.solve(z0, z_ref, a_seq, b_seq, c_seq) for _ in range(12)]
    result = results[-1]

    assert result.states.shape == (4, n + 1)
    assert result.controls.shape == (2, n)
    np.testing.assert_allclose(result.states[:, 0], z0, atol=2e-5)
    residuals = [
        result.states[:, k + 1]
        - (a_seq[k] @ result.states[:, k] + b_seq[k] @ result.controls[:, k] + c_seq[k])
        for k in range(n)
    ]
    assert np.max(np.abs(residuals)) < 2e-5
    assert np.all(result.controls[0] >= cfg.min_accel - 2e-5)
    assert np.all(result.controls[0] <= cfg.max_accel + 2e-5)
    assert np.max(np.abs(result.controls[1])) <= cfg.max_steer + 2e-5
    steering_moves = np.diff(np.r_[0.0, result.controls[1]])
    assert np.max(np.abs(steering_moves)) <= cfg.max_steer_rate * cfg.dt + 2e-5
    assert result.first_control[1] < 0.0, "positive lateral offset should command corrective right steer"

    warm_times = [item.wall_time_ms for item in results[2:]]
    print(f"status: {result.status}")
    print(f"objective: {result.objective:.6f}")
    print(f"first control [a, delta]: {result.first_control}")
    print(f"max dynamics residual: {np.max(np.abs(residuals)):.3e}")
    print(f"warm solve median: {median(warm_times):.3f} ms")
    print(f"warm solve maximum: {max(warm_times):.3f} ms")
    assert median(warm_times) < 50.0, "C3 timing target not met on this machine"
    print("C3 validation passed")


if __name__ == "__main__":
    main()

