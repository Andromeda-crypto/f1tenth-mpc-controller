"""C7 equivalence and latency validation for the persistent native OSQP backend.

Run from the repository root:

    python -m experiments.c7_native_osqp_benchmark
"""

from __future__ import annotations

import csv
from pathlib import Path
from statistics import median

import numpy as np

from experiments.c4_forward_sim import IterativeLinearMPC, make_curve_path, nonlinear_kbm_step
from f1tenth_mpc.mpc_qp import LinearMPCQP, MPCConfig, linearize_discrete_kbm
from f1tenth_mpc.native_osqp_mpc import NativeOSQPLinearMPCQP


def make_problem(config: MPCConfig, wheelbase: float):
    path = make_curve_path(radius=8.0, speed=2.0)
    state = np.array([0.0, 0.75, 2.0, 0.0])
    reference, _ = path.reference_horizon(state, config.horizon, config.dt, 0.0)
    controls = np.zeros((2, config.horizon))
    nominal = np.zeros((4, config.horizon + 1)); nominal[:, 0] = state
    a = np.zeros((config.horizon, 4, 4)); b = np.zeros((config.horizon, 4, 2)); c = np.zeros((config.horizon, 4))
    for k in range(config.horizon):
        nominal[:, k + 1] = nonlinear_kbm_step(
            nominal[:, k], controls[:, k], dt=config.dt, wheelbase=wheelbase
        )
        a[k], b[k], c[k] = linearize_discrete_kbm(
            nominal[:, k], controls[:, k], dt=config.dt, wheelbase=wheelbase
        )
    return state, reference, a, b, c


def main() -> None:
    wheelbase = 0.3302
    config = MPCConfig(
        horizon=8, dt=0.1, max_speed=8.0,
        q=(4.0, 4.0, 1.0, 4.0), q_terminal=(8.0, 8.0, 2.0, 8.0),
        r=(0.1, 0.2), r_delta=(0.1, 1.0),
    )
    args = make_problem(config, wheelbase)
    previous = np.array([0.15, -0.03])
    reference_solver = LinearMPCQP(config)
    native_solver = NativeOSQPLinearMPCQP(config)
    cvx = reference_solver.solve(*args, u_previous=previous, reuse_solver_cache=False)
    native = native_solver.solve(*args, u_previous=previous)
    control_difference = float(np.max(np.abs(cvx.controls - native.controls)))
    state_difference = float(np.max(np.abs(cvx.states - native.states)))
    relative_objective_difference = abs(cvx.objective - native.objective) / max(1.0, abs(cvx.objective))

    # Perturb the numeric data without changing sparsity and measure warm updates.
    warm_times = []
    for sample in range(102):
        state, reference, a, b, c = make_problem(config, wheelbase)
        state[1] += 0.1 * np.sin(0.1 * sample)
        reference = reference.copy(); reference[1] += 0.02 * np.cos(0.07 * sample)
        result = native_solver.solve(state, reference, a, b, c, u_previous=previous)
        if sample >= 2: warm_times.append(result.wall_time_ms)

    # Benchmark the complete successive-linearization controller, not just OSQP.
    controller = IterativeLinearMPC(config, wheelbase, solver_backend="native")
    path = make_curve_path(radius=8.0, speed=2.0)
    state = np.array([0.0, 0.75, 2.0, 0.0]); progress = 0.0
    update_times = []
    for sample in range(82):
        reference, progress = path.reference_horizon(state, config.horizon, config.dt, progress)
        control, update_ms, status, _ = controller.command(state, reference)
        if status not in ("optimal", "optimal_inaccurate"): raise AssertionError(status)
        state = nonlinear_kbm_step(state, control, dt=config.dt, wheelbase=wheelbase)
        if sample >= 2: update_times.append(update_ms)

    rows = [{
        "max_control_difference": control_difference,
        "max_state_difference": state_difference,
        "relative_objective_difference": relative_objective_difference,
        "median_native_qp_ms": median(warm_times),
        "max_native_qp_ms": max(warm_times),
        "median_native_controller_ms": median(update_times),
        "max_native_controller_ms": max(update_times),
        "controller_deadline_misses_100ms": sum(t > 100.0 for t in update_times),
    }]
    output = Path(__file__).resolve().parents[1] / "results" / "c7"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "c7_benchmark.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    for key, value in rows[0].items(): print(f"{key}: {value}")

    if control_difference > 2e-3 or state_difference > 2e-3 or relative_objective_difference > 2e-4:
        raise AssertionError("C7 failed: native OSQP is not numerically equivalent to CVXPY")
    if median(warm_times) >= 5.0:
        raise AssertionError("C7 failed: median native QP latency is not below 5 ms")
    if any(t > 100.0 for t in update_times):
        raise AssertionError("C7 failed: complete controller missed its 100 ms deadline")
    print(f"C7 validation passed; results written to {output}")


if __name__ == "__main__":
    main()
