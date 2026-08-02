"""C6: compare the selected MPC against a fairly tuned Pure Pursuit baseline.

The comparison fixes the simulator, track, speed profile, initial pose, control
period, vehicle limits, and metrics.  MPC uses the heading_2x weights selected
by ``c6_tuning_sweep.py``.  Pure Pursuit uses the repository's existing
``PurePursuitController`` and sweeps only its lookahead distance.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import median
from time import perf_counter

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from c5_f1tenth_gym import (
    ContinuousYaw,
    collision_detected,
    find_default_config,
    lap_count,
    load_track,
    make_environment,
    observation_state,
    reset_environment,
    step_environment,
)
from c6_tuning_sweep import TuneCase, is_valid, run_case
from pure_pursuit import PurePursuitController


FINAL_MPC = TuneCase(
    "mpc_heading_2x",
    q=(4.0, 4.0, 1.0, 4.0),
    q_terminal=(8.0, 8.0, 2.0, 8.0),
    r=(0.1, 0.2),
    r_delta=(0.1, 1.0),
)

LOOKAHEADS_M = (0.8, 1.0, 1.2, 1.5, 1.8, 2.2)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="C6 MPC versus Pure Pursuit comparison")
    parser.add_argument("--config", help="path to config_example_map.yaml")
    parser.add_argument("--waypoints", help="optional waypoint CSV override")
    parser.add_argument("--max-time", type=float, default=60.0)
    parser.add_argument("--wheelbase", type=float, default=0.3302)
    return parser.parse_args()


def run_pure_pursuit(
    lookahead: float,
    *,
    track,
    gym_config: dict,
    config_path: Path,
    wheelbase: float,
    max_time: float,
) -> dict[str, float | int | str | bool]:
    physics_dt = 0.01
    control_dt = 0.1
    action_hold_steps = round(control_dt / physics_dt)
    maximum_speed = 8.0
    minimum_accel = -8.0
    maximum_accel = 8.0
    maximum_steer = float(np.deg2rad(24.0))
    maximum_steer_rate = float(np.deg2rad(180.0))

    controller = PurePursuitController(
        wheelbase=wheelbase,
        lookahead_distance=lookahead,
        max_steering_angle=maximum_steer,
    )
    path = {"x": track.x, "y": track.y}
    env = make_environment(gym_config, config_path, physics_dt)
    initial_pose = np.array(
        [[float(gym_config["sx"]), float(gym_config["sy"]), float(gym_config["stheta"])]]
    )
    observation, reward, done, _ = reset_environment(env, initial_pose)
    elapsed_time = float(reward)
    yaw_tracker = ContinuousYaw()
    progress: float | None = None
    action = np.zeros((1, 2), dtype=float)
    previous_steering = 0.0

    cte_values: list[float] = []
    heading_values: list[float] = []
    steering_commands: list[float] = []
    update_times: list[float] = []

    try:
        while not done and elapsed_time < max_time and lap_count(observation) < 1:
            state = observation_state(observation, yaw_tracker)
            progress = track.update_progress(state, progress)
            reference_speed = float(
                np.interp(progress % track.length, track.s, track.speed)
            )

            update_start = perf_counter()
            pp_state = np.array([state[0], state[1], state[3], state[2]])
            raw_steering, _, _ = controller.compute_steering(pp_state, path)

            maximum_move = maximum_steer_rate * control_dt
            steering = float(
                np.clip(
                    raw_steering,
                    previous_steering - maximum_move,
                    previous_steering + maximum_move,
                )
            )
            steering = float(np.clip(steering, -maximum_steer, maximum_steer))
            previous_steering = steering

            # Use the same acceleration-limited speed-command update used by MPC.
            acceleration = float(
                np.clip(2.0 * (reference_speed - state[2]), minimum_accel, maximum_accel)
            )
            desired_speed = float(
                np.clip(state[2] + acceleration * control_dt, 0.0, maximum_speed)
            )
            update_times.append(1000.0 * (perf_counter() - update_start))

            action[0] = [steering, desired_speed]
            cross_track, heading_error = track.errors(state)
            cte_values.append(cross_track)
            heading_values.append(heading_error)
            steering_commands.append(steering)

            for _ in range(action_hold_steps):
                observation, step_reward, done, _ = step_environment(env, action.copy())
                elapsed_time += float(step_reward)
                if done or lap_count(observation) >= 1:
                    break
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()

    completed = lap_count(observation) >= 1
    collided = collision_detected(observation)
    cte = np.asarray(cte_values, dtype=float)
    heading = np.asarray(heading_values, dtype=float)
    steering = np.asarray(steering_commands, dtype=float)
    updates = np.asarray(update_times, dtype=float)
    changes = np.diff(np.r_[0.0, steering])

    return {
        "case": f"pure_pursuit_{lookahead:.1f}m",
        "controller": "Pure Pursuit",
        "lookahead_m": lookahead,
        "completed": completed,
        "collision": collided,
        "lap_time_s": elapsed_time,
        "cte_rmse_m": float(np.sqrt(np.mean(cte**2))) if len(cte) else float("nan"),
        "heading_rmse_deg": (
            float(np.rad2deg(np.sqrt(np.mean(heading**2))))
            if len(heading)
            else float("nan")
        ),
        "steering_tv_rad": float(np.sum(np.abs(changes))) if len(changes) else float("nan"),
        "steering_rms_rad": float(np.sqrt(np.mean(steering**2))) if len(steering) else float("nan"),
        "median_update_ms": float(median(update_times)) if update_times else float("nan"),
        "p95_update_ms": float(np.percentile(updates, 95)) if len(updates) else float("nan"),
        "deadline_miss_pct": (
            float(100.0 * np.mean(updates > 100.0)) if len(updates) else float("nan")
        ),
        "mean_iterations": 0.0,
        "solver_failures": 0,
        "q": "not_applicable",
        "q_terminal": "not_applicable",
        "r": "not_applicable",
        "r_delta": "not_applicable",
    }


def pp_dominates(left: dict, right: dict) -> bool:
    fields = ("lap_time_s", "cte_rmse_m", "heading_rmse_deg", "steering_tv_rad")
    return all(float(left[key]) <= float(right[key]) for key in fields) and any(
        float(left[key]) < float(right[key]) for key in fields
    )


def pure_pursuit_pareto(rows: list[dict]) -> list[str]:
    valid = [row for row in rows if is_valid(row)]
    return [
        str(row["case"])
        for row in valid
        if not any(pp_dominates(other, row) for other in valid if other is not row)
    ]


def print_row(row: dict) -> None:
    print(
        f"{row['case']:<22} completed={str(row['completed']):<5} "
        f"collision={str(row['collision']):<5} lap={row['lap_time_s']:6.3f} s  "
        f"CTE={row['cte_rmse_m']:.4f} m  heading={row['heading_rmse_deg']:.3f} deg  "
        f"steerTV={row['steering_tv_rad']:.4f} rad  "
        f"update={row['median_update_ms']:.2f}/{row['p95_update_ms']:.2f} ms med/p95"
    )


def write_summary(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_plot(path: Path, rows: list[dict]) -> None:
    valid = [row for row in rows if is_valid(row)]
    figure, axis = plt.subplots(figsize=(9, 6))
    for row in valid:
        is_mpc = str(row["case"]).startswith("mpc_")
        axis.scatter(
            float(row["cte_rmse_m"]),
            float(row["steering_tv_rad"]),
            s=130 if is_mpc else 65,
            marker="*" if is_mpc else "o",
            color="tab:red" if is_mpc else "tab:blue",
        )
        axis.annotate(
            str(row["case"]),
            (float(row["cte_rmse_m"]), float(row["steering_tv_rad"])),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    axis.set_xlabel("Cross-track RMSE [m] (lower is better)")
    axis.set_ylabel("Steering total variation [rad] (lower is smoother)")
    axis.set_title("C6: selected MPC versus Pure Pursuit lookahead sweep")
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    arguments = parse_arguments()
    config_path = Path(arguments.config).resolve() if arguments.config else find_default_config()
    with config_path.open("r", encoding="utf-8") as handle:
        gym_config = yaml.safe_load(handle)

    print("[1/7] running selected MPC: heading_2x")
    mpc_row = run_case(
        FINAL_MPC,
        gym_config=gym_config,
        config_path=config_path,
        waypoint_override=arguments.waypoints,
        wheelbase=arguments.wheelbase,
        max_time=arguments.max_time,
    )
    mpc_row["controller"] = "MPC"
    mpc_row["lookahead_m"] = "not_applicable"
    print_row(mpc_row)

    track = load_track(gym_config, config_path, arguments.waypoints)
    pp_rows: list[dict] = []
    for index, lookahead in enumerate(LOOKAHEADS_M, start=2):
        print(f"\n[{index}/7] running Pure Pursuit lookahead={lookahead:.1f} m")
        row = run_pure_pursuit(
            lookahead,
            track=track,
            gym_config=gym_config,
            config_path=config_path,
            wheelbase=arguments.wheelbase,
            max_time=arguments.max_time,
        )
        pp_rows.append(row)
        print_row(row)

    rows = [mpc_row, *pp_rows]
    results_directory = Path(__file__).resolve().parent / "c6_results"
    summary_path = results_directory / "c6_controller_comparison.csv"
    plot_path = results_directory / "c6_controller_comparison.png"
    write_summary(summary_path, rows)
    write_plot(plot_path, rows)

    pareto = pure_pursuit_pareto(pp_rows)
    print("\nC6 controller comparison complete")
    print(f"valid Pure Pursuit cases: {sum(is_valid(row) for row in pp_rows)}/{len(pp_rows)}")
    print(f"Pure Pursuit Pareto candidates: {', '.join(pareto) if pareto else 'none'}")
    print(f"summary: {summary_path}")
    print(f"plot: {plot_path}")

    if not is_valid(mpc_row):
        raise AssertionError("selected MPC failed its final comparison lap")
    if not pareto:
        raise AssertionError("no valid Pure Pursuit comparison case completed")


if __name__ == "__main__":
    main()