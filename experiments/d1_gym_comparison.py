"""D1: controlled F1TENTH Gym comparison of MPC and Pure Pursuit."""

from __future__ import annotations

import argparse
import csv
import inspect
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
import yaml

from experiments.c4_forward_sim import IterativeLinearMPC
from experiments.c5_f1tenth_gym import (
    ClosedTrack, ContinuousYaw, collision_detected, find_default_config,
    lap_count, load_track, make_environment, observation_state,
    render_environment, reset_environment, step_environment,
)
from f1tenth_mpc.mpc_qp import MPCConfig
from f1tenth_mpc.pure_pursuit import PurePursuitController


SCRIPT_REVISION = "D1 comparison v3 - validated reference and post-step diagnostics"


@dataclass
class ControllerResult:
    controller: str
    telemetry: list[dict[str, float | int | str]]
    simulated_time: float
    lap_completed: bool
    collision: bool
    termination_reason: str
    terminal_state: dict[str, float | int | str]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare MPC and Pure Pursuit")
    parser.add_argument("--controller", choices=("both", "mpc", "pure-pursuit"), default="both")
    parser.add_argument("--solver", choices=("native", "cvxpy"), default="native")
    parser.add_argument("--config")
    parser.add_argument("--waypoints")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--max-time", type=float, default=60.0)
    parser.add_argument("--wheelbase", type=float, default=0.3302)
    parser.add_argument("--lookahead", type=float, default=0.8246188790)
    return parser.parse_args()


def make_mpc_controller(config: MPCConfig, wheelbase: float, solver: str) -> IterativeLinearMPC:
    parameters = inspect.signature(IterativeLinearMPC).parameters
    for keyword in ("solver_backend", "backend", "solver"):
        if keyword in parameters:
            return IterativeLinearMPC(config, wheelbase, **{keyword: solver})
    if solver != "cvxpy":
        raise RuntimeError("IterativeLinearMPC exposes no solver-backend argument")
    return IterativeLinearMPC(config, wheelbase)


def reference_speed_at_progress(track: ClosedTrack, progress: float) -> float:
    return float(np.interp(progress % track.length, track.s, track.speed))


def make_telemetry_row(
    *, controller: str, elapsed_time: float, state: np.ndarray,
    acceleration: float, steering: float, desired_speed: float,
    cross_track: float, heading_error: float, update_ms: float,
    iterations: int, status: str, progress: float, observation: dict,
) -> dict[str, float | int | str]:
    return {
        "controller": controller, "time_s": elapsed_time,
        "x_m": float(state[0]), "y_m": float(state[1]),
        "speed_mps": float(state[2]), "yaw_rad": float(state[3]),
        "accel_command_mps2": acceleration, "steer_command_rad": steering,
        "speed_command_mps": desired_speed, "cross_track_error_m": cross_track,
        "heading_error_rad": heading_error, "controller_update_ms": update_ms,
        "controller_iterations": int(iterations), "controller_status": status,
        "progress_m": progress, "lap_count": lap_count(observation),
        "collision": int(collision_detected(observation)),
    }


def write_telemetry(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        raise RuntimeError(f"no telemetry available for {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _vehicle_geometry(env) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    current = env
    seen: set[int] = set()
    while hasattr(current, "env") and id(current) not in seen:
        seen.add(id(current))
        current = current.env
    agents = getattr(getattr(current, "sim", None), "agents", None)
    if agents is None or len(agents) == 0:
        raise RuntimeError("cannot access F1TENTH Gym vehicle geometry")
    vehicle = agents[0]
    cosines = np.asarray(vehicle.cosines, dtype=float)
    sides = np.asarray(vehicle.side_distances, dtype=float)
    angles = getattr(vehicle, "scan_angles", None)
    return cosines, sides, None if angles is None else np.asarray(angles, dtype=float)


def collision_diagnostics(
    observation: dict, cosines: np.ndarray, sides: np.ndarray,
    angles: np.ndarray | None, pre_step_speed: float,
) -> dict[str, float | int]:
    scans = observation.get("scans")
    if scans is None or len(scans) == 0:
        return {"minimum_clearance_m": float("nan"), "minimum_ttc_s": float("nan"),
                "critical_beam_index": -1, "critical_beam_angle_rad": float("nan"),
                "critical_scan_range_m": float("nan"),
                "critical_side_distance_m": float("nan")}
    scan = np.asarray(scans[0], dtype=float)
    n = min(scan.size, cosines.size, sides.size)
    scan, beam_cos, boundary = scan[:n], cosines[:n], sides[:n]
    clearance = scan - boundary
    finite = np.isfinite(clearance)
    clearance_index = int(np.argmin(np.where(finite, clearance, np.inf))) if np.any(finite) else -1
    minimum_clearance = float(clearance[clearance_index]) if clearance_index >= 0 else float("nan")
    projected_speed = float(pre_step_speed) * beam_cos
    valid = finite & np.isfinite(scan) & (projected_speed > 0.0)
    ttc = np.full(n, np.inf)
    ttc[valid] = clearance[valid] / projected_speed[valid]
    candidates = valid & (ttc >= 0.0)
    critical = int(np.argmin(np.where(candidates, ttc, np.inf))) if np.any(candidates) else clearance_index
    minimum_ttc = float(ttc[critical]) if critical >= 0 and candidates[critical] else float("nan")
    if critical < 0:
        angle = scan_range = side_distance = float("nan")
    else:
        angle = float(angles[critical]) if angles is not None and critical < angles.size else float(np.arccos(np.clip(beam_cos[critical], -1.0, 1.0)))
        scan_range, side_distance = float(scan[critical]), float(boundary[critical])
    return {"minimum_clearance_m": minimum_clearance, "minimum_ttc_s": minimum_ttc,
            "critical_beam_index": critical, "critical_beam_angle_rad": angle,
            "critical_scan_range_m": scan_range,
            "critical_side_distance_m": side_distance}


def _termination(observation: dict, done: bool, elapsed: float, maximum: float) -> str:
    if collision_detected(observation):
        return "collision"
    if lap_count(observation) >= 1:
        return "lap_complete"
    if elapsed >= maximum:
        return "max_time"
    if done:
        return "unexpected_gym_done"
    return "running"


def run_controller(
    *, controller_name: str, track: ClosedTrack, gym_config: dict,
    config_path: Path, mpc_config: MPCConfig, wheelbase: float,
    lookahead: float, solver: str, physics_dt: float, control_dt: float,
    max_time: float, render: bool,
) -> ControllerResult:
    hold_steps = round(control_dt / physics_dt)
    if not np.isclose(hold_steps * physics_dt, control_dt):
        raise ValueError("control_dt must be an integer multiple of physics_dt")
    env = make_environment(gym_config, config_path, physics_dt)
    cosines, sides, angles = _vehicle_geometry(env)
    # The accepted path may be a different immutable candidate than the
    # requested raceline.  Reset on that validated reference, not on an
    # unrelated pose retained in the source config.
    initial_pose = np.array([[track.x[0], track.y[0], track.yaw[0]]], dtype=float)
    observation, reward, done, _ = reset_environment(env, initial_pose)
    elapsed = float(reward)
    yaw_tracker, progress = ContinuousYaw(), None
    telemetry: list[dict[str, float | int | str]] = []
    action = np.zeros((1, 2), dtype=float)
    updates = 0

    if controller_name == "MPC":
        controller = make_mpc_controller(mpc_config, wheelbase, solver)
    elif controller_name == "Pure Pursuit":
        controller = PurePursuitController(wheelbase=wheelbase, lookahead_distance=lookahead,
                                           max_steering_angle=mpc_config.max_steer)
    else:
        raise ValueError(f"unknown controller: {controller_name}")
    path = {"x": track.x, "y": track.y}

    try:
        if render:
            render_environment(env)
        while _termination(observation, done, elapsed, max_time) == "running":
            state = observation_state(observation, yaw_tracker)
            progress = track.update_progress(state, progress)
            reference = track.reference_horizon(state, progress, mpc_config.horizon, mpc_config.dt)
            if controller_name == "MPC":
                control, update_ms, status, iterations = controller.command(state, reference)
                acceleration, steering = float(control[0]), float(control[1])
            else:
                start = perf_counter()
                pp_state = np.array([state[0], state[1], state[3], state[2]])
                steering, _, _ = controller.compute_steering(pp_state, path)
                target = reference_speed_at_progress(track, progress)
                acceleration = float(np.clip((target - state[2]) / control_dt,
                                             mpc_config.min_accel, mpc_config.max_accel))
                update_ms, status, iterations = (perf_counter() - start) * 1000.0, "not_applicable", 0
                steering = float(steering)
            desired_speed = float(np.clip(state[2] + acceleration * control_dt,
                                          mpc_config.min_speed, mpc_config.max_speed))
            action[0] = [steering, desired_speed]
            pre_step_speed = float(state[2])
            for _ in range(hold_steps):
                observation, step_reward, done, _ = step_environment(env, action.copy())
                elapsed += float(step_reward)
                if render:
                    render_environment(env)
                if _termination(observation, done, elapsed, max_time) != "running":
                    break

            post_state = observation_state(observation, yaw_tracker)
            post_progress = track.update_progress(post_state, progress)
            cross_track, heading_error = track.errors(post_state)
            diagnostics = collision_diagnostics(observation, cosines, sides, angles, pre_step_speed)
            reason = _termination(observation, done, elapsed, max_time)
            row = make_telemetry_row(
                controller=controller_name, elapsed_time=elapsed, state=post_state,
                acceleration=acceleration, steering=steering, desired_speed=desired_speed,
                cross_track=cross_track, heading_error=heading_error, update_ms=update_ms,
                iterations=iterations, status=status, progress=post_progress, observation=observation)
            row.update(diagnostics)
            row["termination_reason"] = reason
            telemetry.append(row)
            progress, updates = post_progress, updates + 1
            if updates % 20 == 0 or reason != "running":
                print(f"{controller_name:<12} t={elapsed:6.2f}s  progress={post_progress:7.2f}m  "
                      f"CTE={cross_track:.3f}m  speed={post_state[2]:.2f}m/s  "
                      f"clearance={diagnostics['minimum_clearance_m']:.4f}m  "
                      f"TTC={diagnostics['minimum_ttc_s']:.6f}s  termination={reason}")
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()

    if not telemetry:
        raise RuntimeError(f"{controller_name} ended before its first control update")
    reason = _termination(observation, done, elapsed, max_time)
    terminal = dict(telemetry[-1])
    print(f"\n{controller_name} termination: {reason}; time={elapsed:.3f}s; "
          f"pose=({terminal['x_m']:.4f}, {terminal['y_m']:.4f}); "
          f"clearance={terminal['minimum_clearance_m']:.6f}m; "
          f"TTC={terminal['minimum_ttc_s']:.6f}s; beam={terminal['critical_beam_index']}")
    return ControllerResult(controller_name, telemetry, elapsed,
                            lap_count(observation) >= 1, collision_detected(observation),
                            reason, terminal)


def calculate_summary(result: ControllerResult, control_dt: float) -> dict[str, float | int | str | bool]:
    def values(key: str) -> np.ndarray:
        return np.asarray([row[key] for row in result.telemetry], dtype=float)
    cross_track, heading, steering = values("cross_track_error_m"), values("heading_error_rad"), values("steer_command_rad")
    speed, update_times = values("speed_mps"), values("controller_update_ms")
    steering_change = np.diff(np.r_[0.0, steering])
    return {
        "controller": result.controller, "lap_completed": result.lap_completed,
        "collision": result.collision, "termination_reason": result.termination_reason,
        "lap_time_s": result.simulated_time, "control_updates": len(result.telemetry),
        "cross_track_rmse_m": float(np.sqrt(np.mean(cross_track ** 2))),
        "cross_track_max_m": float(np.max(np.abs(cross_track))),
        "heading_rmse_deg": float(np.rad2deg(np.sqrt(np.mean(heading ** 2)))),
        "heading_max_deg": float(np.rad2deg(np.max(np.abs(heading)))),
        "mean_speed_mps": float(np.mean(speed)), "max_speed_mps": float(np.max(speed)),
        "steering_total_variation_rad": float(np.sum(np.abs(steering_change))),
        "max_abs_steer_deg": float(np.rad2deg(np.max(np.abs(steering)))),
        "median_controller_update_ms": float(median(update_times)),
        "max_controller_update_ms": float(np.max(update_times)),
        "deadline_misses_100ms": int(np.sum(update_times > control_dt * 1000.0)),
    }


def write_summary(path: Path, rows: list[dict[str, float | int | str | bool]]) -> None:
    if not rows:
        raise RuntimeError("no comparison summary rows available")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def print_summary(rows: list[dict[str, float | int | str | bool]]) -> None:
    print("\nD1 controlled comparison")
    for row in rows:
        print(f"{row['controller']:<14} lap={row['lap_time_s']:.3f}s  "
              f"CTE={row['cross_track_rmse_m']:.4f}m  heading={row['heading_rmse_deg']:.3f}deg  "
              f"steerTV={row['steering_total_variation_rad']:.4f}rad  "
              f"update={row['median_controller_update_ms']:.3f}ms  "
              f"termination={row['termination_reason']}")


def validate_result(result: ControllerResult, summary: dict[str, float | int | str | bool]) -> None:
    if result.collision:
        raise AssertionError(f"D1 failed: {result.controller} collided")
    if not result.lap_completed:
        raise AssertionError(f"D1 failed: {result.controller} did not complete a lap")
    if float(summary["cross_track_rmse_m"]) >= 1.0:
        raise AssertionError(f"D1 failed: {result.controller} CTE RMSE exceeded 1 m")
    if int(summary["deadline_misses_100ms"]) != 0:
        raise AssertionError(f"D1 failed: {result.controller} missed the 100 ms deadline")


def main() -> None:
    args = parse_arguments()
    print(SCRIPT_REVISION)
    config_path = Path(args.config).resolve() if args.config else find_default_config()
    with config_path.open(encoding="utf-8") as handle:
        gym_config = yaml.safe_load(handle)
    track = load_track(gym_config, config_path, args.waypoints)
    control_dt, physics_dt = 0.1, 0.01
    mpc_config = MPCConfig(horizon=8, dt=control_dt, max_speed=8.0,
                           q=(4.0, 4.0, 1.0, 4.0), q_terminal=(8.0, 8.0, 2.0, 8.0),
                           r=(0.1, 0.2), r_delta=(0.1, 1.0))
    names = {"both": ("MPC", "Pure Pursuit"), "mpc": ("MPC",),
             "pure-pursuit": ("Pure Pursuit",)}[args.controller]
    output = Path(__file__).resolve().parents[1] / "results" / "d1"
    summaries = []
    for name in names:
        print(f"\nrunning {name}...")
        result = run_controller(controller_name=name, track=track, gym_config=gym_config,
                                config_path=config_path, mpc_config=mpc_config,
                                wheelbase=args.wheelbase, lookahead=args.lookahead,
                                solver=args.solver, physics_dt=physics_dt,
                                control_dt=control_dt, max_time=args.max_time, render=args.render)
        summary = calculate_summary(result, control_dt)
        filename = "mpc_telemetry.csv" if name == "MPC" else "pure_pursuit_telemetry.csv"
        write_telemetry(output / filename, result.telemetry)
        summaries.append(summary)
        if not result.collision:
            validate_result(result, summary)
    write_summary(output / "comparison_summary.csv", summaries)
    print_summary(summaries)
    print(f"\nD1 results written to {output}")


if __name__ == "__main__":
    main()
