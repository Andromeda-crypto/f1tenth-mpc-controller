"""C4: closed-loop MPC validation in a standalone nonlinear forward simulator.

Run from this directory with:

    python -m experiments.c4_forward_sim

State order: z = [x, y, v, psi]
Input order: u = [acceleration, steering_angle]
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import numpy as np

from f1tenth_mpc.mpc_qp import MPCConfig, LinearMPCQP, linearize_discrete_kbm


def wrap_angle(angle: np.ndarray | float) -> np.ndarray | float:
    """Wrap angle(s) to [-pi, pi)."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def nonlinear_kbm_step(
    state: np.ndarray,
    control: np.ndarray,
    *,
    dt: float,
    wheelbase: float,
) -> np.ndarray:
    """Advance the nonlinear kinematic bicycle plant by one Euler step."""
    x, y, speed, yaw = np.asarray(state, dtype=float)
    accel, steer = np.asarray(control, dtype=float)
    derivative = np.array(
        [
            speed * np.cos(yaw),
            speed * np.sin(yaw),
            accel,
            speed * np.tan(steer) / wheelbase,
        ]
    )
    return np.asarray(state, dtype=float) + dt * derivative


@dataclass(frozen=True)
class ReferencePath:
    name: str
    x: np.ndarray
    y: np.ndarray
    yaw: np.ndarray
    speed: np.ndarray
    s: np.ndarray

    def __post_init__(self) -> None:
        arrays = [self.x, self.y, self.yaw, self.speed, self.s]
        if any(np.asarray(item).ndim != 1 for item in arrays):
            raise ValueError("reference path arrays must be one-dimensional")
        if len({len(item) for item in arrays}) != 1 or len(self.x) < 2:
            raise ValueError("reference path arrays must have equal length >= 2")
        if np.any(np.diff(self.s) <= 0.0):
            raise ValueError("path distance s must be strictly increasing")

    @property
    def length(self) -> float:
        return float(self.s[-1])

    def nearest_index(self, state: np.ndarray) -> int:
        distance_squared = (self.x - state[0]) ** 2 + (self.y - state[1]) ** 2
        return int(np.argmin(distance_squared))

    def reference_horizon(
        self,
        state: np.ndarray,
        horizon: int,
        dt: float,
        minimum_progress: float,
    ) -> tuple[np.ndarray, float]:
        """Interpolate a time-indexed horizon beginning at nearest progress."""
        nearest_s = float(self.s[self.nearest_index(state)])
        progress = max(minimum_progress, nearest_s)
        reference_speed = float(np.interp(progress, self.s, self.speed))
        sample_s = np.minimum(
            progress + np.arange(horizon + 1) * reference_speed * dt,
            self.s[-1],
        )
        yaw = np.interp(sample_s, self.s, self.yaw)
        # Align the unwrapped path heading with the current continuous yaw.
        yaw += 2.0 * np.pi * np.round((state[3] - yaw[0]) / (2.0 * np.pi))
        z_ref = np.vstack(
            [
                np.interp(sample_s, self.s, self.x),
                np.interp(sample_s, self.s, self.y),
                np.interp(sample_s, self.s, self.speed),
                yaw,
            ]
        )
        return z_ref, progress


def _path_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    ds = np.hypot(np.diff(x), np.diff(y))
    return np.r_[0.0, np.cumsum(ds)]


def make_straight_path(length: float = 30.0, speed: float = 2.0) -> ReferencePath:
    x = np.linspace(0.0, length, 1501)
    y = np.zeros_like(x)
    yaw = np.zeros_like(x)
    return ReferencePath("straight", x, y, yaw, np.full_like(x, speed), _path_distance(x, y))


def make_curve_path(radius: float = 8.0, speed: float = 2.0) -> ReferencePath:
    angle = np.linspace(0.0, np.pi, 1601)
    x = radius * np.sin(angle)
    y = radius * (1.0 - np.cos(angle))
    yaw = angle.copy()
    return ReferencePath("curve", x, y, yaw, np.full_like(x, speed), _path_distance(x, y))


@dataclass(frozen=True)
class SimulationResult:
    controller: str
    path_name: str
    time: np.ndarray
    states: np.ndarray
    controls: np.ndarray
    solve_times_ms: np.ndarray
    statuses: tuple[str, ...]
    iterations: np.ndarray


class IterativeLinearMPC:
    """Successive-linearization wrapper around the verified C3 QP."""

    def __init__(
        self,
        config: MPCConfig,
        wheelbase: float,
        *,
        max_iterations: int = 3,
        convergence_tolerance: float = 1e-2,
    ) -> None:
        self.config = config
        self.wheelbase = wheelbase
        self.max_iterations = max_iterations
        self.convergence_tolerance = convergence_tolerance
        self.qp = LinearMPCQP(config)
        self.control_guess = np.zeros((2, config.horizon))
        self.previous_applied_control = np.zeros(2)

    def _nominal_rollout(self, state: np.ndarray, controls: np.ndarray) -> np.ndarray:
        nominal = np.zeros((4, self.config.horizon + 1))
        nominal[:, 0] = state
        for k in range(self.config.horizon):
            nominal[:, k + 1] = nonlinear_kbm_step(
                nominal[:, k],
                controls[:, k],
                dt=self.config.dt,
                wheelbase=self.wheelbase,
            )
        return nominal

    def command(self, state: np.ndarray, z_ref: np.ndarray) -> tuple[np.ndarray, float, str, int]:
        controls = self.control_guess.copy()
        total_wall_time_ms = 0.0
        result = None

        for iteration in range(1, self.max_iterations + 1):
            nominal = self._nominal_rollout(state, controls)
            a_seq = np.zeros((self.config.horizon, 4, 4))
            b_seq = np.zeros((self.config.horizon, 4, 2))
            c_seq = np.zeros((self.config.horizon, 4))
            for k in range(self.config.horizon):
                a_seq[k], b_seq[k], c_seq[k] = linearize_discrete_kbm(
                    nominal[:, k],
                    controls[:, k],
                    dt=self.config.dt,
                    wheelbase=self.wheelbase,
                )

            result = self.qp.solve(
                state,
                z_ref,
                a_seq,
                b_seq,
                c_seq,
                u_previous=self.previous_applied_control,
                reuse_solver_cache=False,
            )
            total_wall_time_ms += result.wall_time_ms
            change = float(np.max(np.abs(result.controls - controls)))
            controls = result.controls
            if change < self.convergence_tolerance:
                break

        if result is None:
            raise RuntimeError("MPC iteration produced no result")

        # OSQP may return values a few solver-tolerance units beyond a bound.
        # Apply the same hard actuator saturation used by the baseline before
        # advancing the nonlinear plant.
        command = result.first_control
        command[0] = np.clip(command[0], self.config.min_accel, self.config.max_accel)
        maximum_move = self.config.max_steer_rate * self.config.dt
        command[1] = np.clip(
            command[1],
            self.previous_applied_control[1] - maximum_move,
            self.previous_applied_control[1] + maximum_move,
        )
        command[1] = np.clip(command[1], -self.config.max_steer, self.config.max_steer)
        self.previous_applied_control = command.copy()
        self.control_guess = np.column_stack((controls[:, 1:], controls[:, -1]))
        return command, total_wall_time_ms, result.status, iteration


class PurePursuit:
    """Pure Pursuit baseline adapted to the same [acceleration, steer] plant input."""

    def __init__(
        self,
        wheelbase: float,
        dt: float,
        *,
        lookahead: float = 0.8,
        speed_gain: float = 2.0,
        min_accel: float = -8.0,
        max_accel: float = 8.0,
        max_steer: float = np.deg2rad(24.0),
        max_steer_rate: float = np.deg2rad(180.0),
    ) -> None:
        self.wheelbase = wheelbase
        self.dt = dt
        self.lookahead = lookahead
        self.speed_gain = speed_gain
        self.min_accel = min_accel
        self.max_accel = max_accel
        self.max_steer = max_steer
        self.max_steer_rate = max_steer_rate
        self.previous_steer = 0.0

    def command(self, state: np.ndarray, path: ReferencePath, progress: float) -> np.ndarray:
        target_s = min(progress + self.lookahead, path.length)
        target_x = float(np.interp(target_s, path.s, path.x))
        target_y = float(np.interp(target_s, path.s, path.y))
        target_speed = float(np.interp(progress, path.s, path.speed))
        alpha = wrap_angle(np.arctan2(target_y - state[1], target_x - state[0]) - state[3])
        steer = np.arctan2(2.0 * self.wheelbase * np.sin(alpha), self.lookahead)
        steer = float(np.clip(steer, -self.max_steer, self.max_steer))
        max_move = self.max_steer_rate * self.dt
        steer = float(np.clip(steer, self.previous_steer - max_move, self.previous_steer + max_move))
        self.previous_steer = steer
        accel = float(np.clip(self.speed_gain * (target_speed - state[2]), self.min_accel, self.max_accel))
        return np.array([accel, steer])


def run_mpc_simulation(
    path: ReferencePath,
    initial_state: np.ndarray,
    config: MPCConfig,
    wheelbase: float,
    steps: int,
) -> SimulationResult:
    controller = IterativeLinearMPC(config, wheelbase)
    states = np.zeros((steps + 1, 4))
    controls = np.zeros((steps, 2))
    states[0] = initial_state
    progress = 0.0
    solve_times: list[float] = []
    statuses: list[str] = []
    iterations: list[int] = []

    for k in range(steps):
        z_ref, progress = path.reference_horizon(states[k], config.horizon, config.dt, progress)
        control, solve_time, status, iteration_count = controller.command(states[k], z_ref)
        controls[k] = control
        states[k + 1] = nonlinear_kbm_step(
            states[k], control, dt=config.dt, wheelbase=wheelbase
        )
        solve_times.append(solve_time)
        statuses.append(status)
        iterations.append(iteration_count)

    return SimulationResult(
        "MPC",
        path.name,
        np.arange(steps + 1) * config.dt,
        states,
        controls,
        np.asarray(solve_times),
        tuple(statuses),
        np.asarray(iterations),
    )


def run_pure_pursuit_simulation(
    path: ReferencePath,
    initial_state: np.ndarray,
    config: MPCConfig,
    wheelbase: float,
    steps: int,
) -> SimulationResult:
    controller = PurePursuit(
        wheelbase,
        config.dt,
        min_accel=config.min_accel,
        max_accel=config.max_accel,
        max_steer=config.max_steer,
        max_steer_rate=config.max_steer_rate,
    )
    states = np.zeros((steps + 1, 4))
    controls = np.zeros((steps, 2))
    states[0] = initial_state
    progress = 0.0

    for k in range(steps):
        nearest_s = float(path.s[path.nearest_index(states[k])])
        progress = max(progress, nearest_s)
        controls[k] = controller.command(states[k], path, progress)
        states[k + 1] = nonlinear_kbm_step(
            states[k], controls[k], dt=config.dt, wheelbase=wheelbase
        )

    return SimulationResult(
        "Pure Pursuit",
        path.name,
        np.arange(steps + 1) * config.dt,
        states,
        controls,
        np.zeros(steps),
        tuple("not_applicable" for _ in range(steps)),
        np.zeros(steps, dtype=int),
    )


def tracking_errors(result: SimulationResult, path: ReferencePath) -> tuple[np.ndarray, np.ndarray]:
    lateral = np.zeros(len(result.states))
    heading = np.zeros(len(result.states))
    for k, state in enumerate(result.states):
        index = path.nearest_index(state)
        lateral[k] = np.hypot(state[0] - path.x[index], state[1] - path.y[index])
        heading[k] = abs(float(wrap_angle(state[3] - path.yaw[index])))
    return lateral, heading


def calculate_metrics(result: SimulationResult, path: ReferencePath) -> dict[str, float | str]:
    lateral, heading = tracking_errors(result, path)
    reference_speed = np.array(
        [path.speed[path.nearest_index(state)] for state in result.states]
    )
    steering = result.controls[:, 1]
    steering_change = np.diff(np.r_[0.0, steering])
    final_index = path.nearest_index(result.states[-1])
    warm_times = result.solve_times_ms[2:] if len(result.solve_times_ms) > 2 else result.solve_times_ms
    return {
        "path": result.path_name,
        "controller": result.controller,
        "cross_track_rmse_m": float(np.sqrt(np.mean(lateral**2))),
        "cross_track_max_m": float(np.max(lateral)),
        "heading_rmse_deg": float(np.rad2deg(np.sqrt(np.mean(heading**2)))),
        "speed_rmse_mps": float(np.sqrt(np.mean((result.states[:, 2] - reference_speed) ** 2))),
        "steering_total_variation_rad": float(np.sum(np.abs(steering_change))),
        "max_abs_steer_deg": float(np.rad2deg(np.max(np.abs(steering)))),
        "progress_m": float(path.s[final_index]),
        "median_control_update_ms": float(median(warm_times)) if len(warm_times) else 0.0,
        "max_control_update_ms": float(np.max(warm_times)) if len(warm_times) else 0.0,
        "mean_sqp_iterations": float(np.mean(result.iterations)) if len(result.iterations) else 0.0,
    }


def validate_result(result: SimulationResult, path: ReferencePath, config: MPCConfig) -> None:
    if not np.all(np.isfinite(result.states)) or not np.all(np.isfinite(result.controls)):
        raise AssertionError(f"{result.controller} {path.name}: non-finite simulation value")
    if np.max(np.abs(result.controls[:, 1])) > config.max_steer + 2e-5:
        raise AssertionError(f"{result.controller} {path.name}: steering bound violated")
    if np.max(result.controls[:, 0]) > config.max_accel + 2e-5:
        raise AssertionError(f"{result.controller} {path.name}: acceleration upper bound violated")
    if np.min(result.controls[:, 0]) < config.min_accel - 2e-5:
        raise AssertionError(f"{result.controller} {path.name}: acceleration lower bound violated")
    steering_moves = np.diff(np.r_[0.0, result.controls[:, 1]])
    if np.max(np.abs(steering_moves)) > config.max_steer_rate * config.dt + 2e-5:
        raise AssertionError(f"{result.controller} {path.name}: steering-rate bound violated")
    if result.controller == "MPC" and any(status not in ("optimal", "optimal_inaccurate") for status in result.statuses):
        raise AssertionError(f"MPC {path.name}: non-optimal QP status")


def write_csvs(
    output_dir: Path,
    paths: dict[str, ReferencePath],
    results: list[SimulationResult],
    metrics: list[dict[str, float | str]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "c4_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0].keys()))
        writer.writeheader()
        writer.writerows(metrics)

    for result in results:
        lateral, heading = tracking_errors(result, paths[result.path_name])
        target = output_dir / f"{result.path_name}_{result.controller.lower().replace(' ', '_')}.csv"
        with target.open("w", newline="", encoding="utf-8") as handle:
            fieldnames = [
                "time_s", "x_m", "y_m", "speed_mps", "yaw_rad", "accel_mps2",
                "steer_rad", "cross_track_error_m", "heading_error_rad", "control_update_ms",
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for k, state in enumerate(result.states):
                control = result.controls[k] if k < len(result.controls) else np.array([np.nan, np.nan])
                solve_time = result.solve_times_ms[k] if k < len(result.solve_times_ms) else np.nan
                writer.writerow(
                    {
                        "time_s": result.time[k],
                        "x_m": state[0],
                        "y_m": state[1],
                        "speed_mps": state[2],
                        "yaw_rad": state[3],
                        "accel_mps2": control[0],
                        "steer_rad": control[1],
                        "cross_track_error_m": lateral[k],
                        "heading_error_rad": heading[k],
                        "control_update_ms": solve_time,
                    }
                )


def write_plots(
    output_dir: Path,
    paths: dict[str, ReferencePath],
    results: list[SimulationResult],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("plots skipped: install matplotlib with 'python -m pip install matplotlib'")
        return

    for path_name, path in paths.items():
        matching = [item for item in results if item.path_name == path_name]
        figure, axes = plt.subplots(1, 3, figsize=(15, 4.2))
        axes[0].plot(path.x, path.y, "k--", linewidth=1.5, label="reference")
        for result in matching:
            axes[0].plot(result.states[:, 0], result.states[:, 1], linewidth=2.0, label=result.controller)
            lateral, _ = tracking_errors(result, path)
            axes[1].plot(result.time, lateral, linewidth=2.0, label=result.controller)
            axes[2].step(result.time[:-1], np.rad2deg(result.controls[:, 1]), where="post", linewidth=1.8, label=result.controller)
        axes[0].set_title(f"{path_name.title()} path tracking")
        axes[0].set_xlabel("x [m]")
        axes[0].set_ylabel("y [m]")
        if path_name == "curve":
            axes[0].axis("equal")
        else:
            trajectory_y = np.concatenate([item.states[:, 1] for item in matching])
            axes[0].set_ylim(float(np.min(trajectory_y)) - 0.15, float(np.max(trajectory_y)) + 0.15)
        axes[1].set_title("Cross-track error")
        axes[1].set_xlabel("time [s]")
        axes[1].set_ylabel("error [m]")
        axes[2].set_title("Steering command")
        axes[2].set_xlabel("time [s]")
        axes[2].set_ylabel("steering [deg]")
        for axis in axes:
            axis.grid(alpha=0.25)
            axis.legend()
        figure.tight_layout()
        figure.savefig(output_dir / f"c4_{path_name}.png", dpi=180)
        plt.close(figure)


def print_metrics(metrics: list[dict[str, float | str]]) -> None:
    print("\nC4 comparison")
    print(
        f"{'path':<10} {'controller':<14} {'CTE RMSE [m]':>13} {'heading [deg]':>14} "
        f"{'steer TV [rad]':>15} {'update [ms]':>12}"
    )
    print("-" * 84)
    for item in metrics:
        print(
            f"{item['path']:<10} {item['controller']:<14} "
            f"{item['cross_track_rmse_m']:>13.4f} {item['heading_rmse_deg']:>14.3f} "
            f"{item['steering_total_variation_rad']:>15.4f} "
            f"{item['median_control_update_ms']:>12.3f}"
        )


def main() -> None:
    wheelbase = 0.3302
    config = MPCConfig(horizon=8, dt=0.1)
    steps = 80
    initial_state = np.array([0.0, 0.75, 2.0, 0.0])
    paths = {
        "straight": make_straight_path(),
        "curve": make_curve_path(),
    }

    results: list[SimulationResult] = []
    for path in paths.values():
        print(f"running {path.name} MPC...")
        results.append(run_mpc_simulation(path, initial_state, config, wheelbase, steps))
        print(f"running {path.name} Pure Pursuit...")
        results.append(run_pure_pursuit_simulation(path, initial_state, config, wheelbase, steps))

    for result in results:
        validate_result(result, paths[result.path_name], config)
    metrics = [calculate_metrics(result, paths[result.path_name]) for result in results]

    # Stability/validity gates, not a predetermined claim that MPC must win.
    for item in metrics:
        if item["cross_track_rmse_m"] >= 1.0:
            raise AssertionError(f"{item['controller']} {item['path']}: tracking RMSE >= 1 m")
        if item["progress_m"] <= 10.0:
            raise AssertionError(f"{item['controller']} {item['path']}: insufficient forward progress")

    output_dir = Path(__file__).resolve().parents[1] / "results" / "c4"
    write_csvs(output_dir, paths, results, metrics)
    write_plots(output_dir, paths, results)
    print_metrics(metrics)
    print(f"\nC4 validation passed; results written to {output_dir}")


if __name__ == "__main__":
    main()
