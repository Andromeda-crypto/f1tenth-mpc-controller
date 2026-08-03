"""C6: tune MPC cost weights on the F1TENTH Gym example track.

The sweep changes only Q, Qf, R, and Rd. Vehicle limits, model horizon,
reference speed, simulator timestep, initial pose, and lap gate remain fixed.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.c4_forward_sim import IterativeLinearMPC
from experiments.c5_f1tenth_gym import (
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
from f1tenth_mpc.mpc_qp import MPCConfig


@dataclass(frozen=True)
class TuneCase:
    name: str
    q: tuple[float, ...]
    q_terminal: tuple[float, ...]
    r: tuple[float, ...]
    r_delta: tuple[float, ...]


CASES = (
    TuneCase(
        "baseline",
        q=(4.0, 4.0, 1.0, 2.0),
        q_terminal=(8.0, 8.0, 2.0, 4.0),
        r=(0.1, 0.2),
        r_delta=(0.1, 1.0),
    ),
    TuneCase(
        "position_2x",
        q=(8.0, 8.0, 1.0, 2.0),
        q_terminal=(16.0, 16.0, 2.0, 4.0),
        r=(0.1, 0.2),
        r_delta=(0.1, 1.0),
    ),
    TuneCase(
        "heading_2x",
        q=(4.0, 4.0, 1.0, 4.0),
        q_terminal=(8.0, 8.0, 2.0, 8.0),
        r=(0.1, 0.2),
        r_delta=(0.1, 1.0),
    ),
    TuneCase(
        "steer_smooth_2x",
        q=(4.0, 4.0, 1.0, 2.0),
        q_terminal=(8.0, 8.0, 2.0, 4.0),
        r=(0.1, 0.2),
        r_delta=(0.1, 2.0),
    ),
    TuneCase(
        "steer_smooth_4x",
        q=(4.0, 4.0, 1.0, 2.0),
        q_terminal=(8.0, 8.0, 2.0, 4.0),
        r=(0.1, 0.2),
        r_delta=(0.1, 4.0),
    ),
    TuneCase(
        "steer_effort_2x",
        q=(4.0, 4.0, 1.0, 2.0),
        q_terminal=(8.0, 8.0, 2.0, 4.0),
        r=(0.1, 0.4),
        r_delta=(0.1, 1.0),
    ),
    TuneCase(
        "balanced",
        q=(8.0, 8.0, 1.0, 4.0),
        q_terminal=(16.0, 16.0, 2.0, 8.0),
        r=(0.1, 0.2),
        r_delta=(0.1, 2.0),
    ),
    TuneCase(
        "balanced_smooth",
        q=(8.0, 8.0, 1.0, 4.0),
        q_terminal=(16.0, 16.0, 2.0, 8.0),
        r=(0.1, 0.4),
        r_delta=(0.1, 4.0),
    ),
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the C6 MPC weight sweep")
    parser.add_argument("--config", help="path to config_example_map.yaml")
    parser.add_argument("--waypoints", help="optional waypoint CSV override")
    parser.add_argument("--max-time", type=float, default=60.0)
    parser.add_argument("--wheelbase", type=float, default=0.3302)
    parser.add_argument(
        "--case",
        action="append",
        dest="case_names",
        help="run only this named case; may be supplied more than once",
    )
    return parser.parse_args()


def run_case(
    case: TuneCase,
    *,
    gym_config: dict,
    config_path: Path,
    waypoint_override: str | None,
    wheelbase: float,
    max_time: float,
) -> dict[str, float | int | str | bool]:
    physics_dt = 0.01
    control_dt = 0.1
    action_hold_steps = round(control_dt / physics_dt)
    mpc_config = MPCConfig(
        horizon=8,
        dt=control_dt,
        max_speed=8.0,
        q=case.q,
        q_terminal=case.q_terminal,
        r=case.r,
        r_delta=case.r_delta,
    )
    track = load_track(gym_config, config_path, waypoint_override)
    controller = IterativeLinearMPC(mpc_config, wheelbase)
    env = make_environment(gym_config, config_path, physics_dt)
    initial_pose = np.array(
        [[float(gym_config["sx"]), float(gym_config["sy"]), float(gym_config["stheta"])]]
    )
    observation, reward, done, _ = reset_environment(env, initial_pose)
    elapsed_time = float(reward)
    yaw_tracker = ContinuousYaw()
    progress: float | None = None
    action = np.zeros((1, 2), dtype=float)

    cte_values: list[float] = []
    heading_values: list[float] = []
    update_times: list[float] = []
    iteration_counts: list[int] = []
    steering_commands: list[float] = []
    solver_failures = 0

    try:
        while not done and elapsed_time < max_time and lap_count(observation) < 1:
            state = observation_state(observation, yaw_tracker)
            progress = track.update_progress(state, progress)
            reference = track.reference_horizon(
                state, progress, mpc_config.horizon, mpc_config.dt
            )
            try:
                control, update_ms, _, iterations = controller.command(state, reference)
            except RuntimeError:
                solver_failures += 1
                break

            acceleration, steering = control
            desired_speed = float(
                np.clip(
                    state[2] + acceleration * control_dt,
                    mpc_config.min_speed,
                    mpc_config.max_speed,
                )
            )
            action[0] = [steering, desired_speed]
            cross_track, heading_error = track.errors(state)
            cte_values.append(cross_track)
            heading_values.append(heading_error)
            update_times.append(update_ms)
            iteration_counts.append(iterations)
            steering_commands.append(float(steering))

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
    changes = np.diff(np.r_[0.0, steering])
    updates = np.asarray(update_times, dtype=float)

    return {
        "case": case.name,
        "completed": completed,
        "collision": collided,
        "lap_time_s": elapsed_time,
        "cte_rmse_m": float(np.sqrt(np.mean(cte**2))) if len(cte) else float("nan"),
        "heading_rmse_deg": (
            float(np.rad2deg(np.sqrt(np.mean(heading**2)))) if len(heading) else float("nan")
        ),
        "steering_tv_rad": float(np.sum(np.abs(changes))) if len(changes) else float("nan"),
        "steering_rms_rad": float(np.sqrt(np.mean(steering**2))) if len(steering) else float("nan"),
        "median_update_ms": float(median(update_times)) if update_times else float("nan"),
        "p95_update_ms": float(np.percentile(updates, 95)) if len(updates) else float("nan"),
        "deadline_miss_pct": (
            float(100.0 * np.mean(updates > 100.0)) if len(updates) else float("nan")
        ),
        "mean_iterations": (
            float(np.mean(iteration_counts)) if iteration_counts else float("nan")
        ),
        "solver_failures": solver_failures,
        "q": repr(case.q),
        "q_terminal": repr(case.q_terminal),
        "r": repr(case.r),
        "r_delta": repr(case.r_delta),
    }


def is_valid(row: dict[str, float | int | str | bool]) -> bool:
    return bool(row["completed"]) and not bool(row["collision"]) and row["solver_failures"] == 0


def dominates(left: dict, right: dict) -> bool:
    """Return True when left is no worse in all control metrics and better in one."""
    fields = ("lap_time_s", "cte_rmse_m", "heading_rmse_deg", "steering_tv_rad")
    no_worse = all(float(left[field]) <= float(right[field]) for field in fields)
    strictly_better = any(float(left[field]) < float(right[field]) for field in fields)
    return no_worse and strictly_better


def pareto_names(rows: list[dict]) -> list[str]:
    valid = [row for row in rows if is_valid(row)]
    return [
        str(row["case"])
        for row in valid
        if not any(dominates(other, row) for other in valid if other is not row)
    ]


def write_summary(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_plot(path: Path, rows: list[dict], pareto: set[str]) -> None:
    valid = [row for row in rows if is_valid(row)]
    if not valid:
        return
    figure, axis = plt.subplots(figsize=(9, 6))
    for row in valid:
        name = str(row["case"])
        axis.scatter(
            float(row["cte_rmse_m"]),
            float(row["steering_tv_rad"]),
            s=90 if name in pareto else 50,
            marker="o" if name in pareto else "x",
        )
        axis.annotate(name, (float(row["cte_rmse_m"]), float(row["steering_tv_rad"])),
                      xytext=(5, 5), textcoords="offset points", fontsize=8)
    axis.set_xlabel("Cross-track RMSE [m] (lower is better)")
    axis.set_ylabel("Steering total variation [rad] (lower is smoother)")
    axis.set_title("C6 MPC weight sweep: tracking versus smoothness")
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def print_row(row: dict) -> None:
    print(
        f"{row['case']:<20} completed={str(row['completed']):<5} "
        f"collision={str(row['collision']):<5} lap={row['lap_time_s']:6.3f} s  "
        f"CTE={row['cte_rmse_m']:.4f} m  heading={row['heading_rmse_deg']:.3f} deg  "
        f"steerTV={row['steering_tv_rad']:.4f} rad  "
        f"update={row['median_update_ms']:.1f}/{row['p95_update_ms']:.1f} ms med/p95  "
        f"miss={row['deadline_miss_pct']:.1f}%"
    )


def main() -> None:
    arguments = parse_arguments()
    config_path = Path(arguments.config).resolve() if arguments.config else find_default_config()
    with config_path.open("r", encoding="utf-8") as handle:
        gym_config = yaml.safe_load(handle)

    selected = CASES
    if arguments.case_names:
        requested = set(arguments.case_names)
        known = {case.name for case in CASES}
        unknown = sorted(requested - known)
        if unknown:
            raise ValueError(f"unknown cases: {unknown}; choose from {sorted(known)}")
        selected = tuple(case for case in CASES if case.name in requested)

    rows: list[dict] = []
    for index, case in enumerate(selected, start=1):
        print(f"\n[{index}/{len(selected)}] running {case.name}")
        row = run_case(
            case,
            gym_config=gym_config,
            config_path=config_path,
            waypoint_override=arguments.waypoints,
            wheelbase=arguments.wheelbase,
            max_time=arguments.max_time,
        )
        rows.append(row)
        print_row(row)

    results_directory = Path(__file__).resolve().parents[1] / "results" / "c6"
    summary_path = results_directory / "c6_tuning_summary.csv"
    plot_path = results_directory / "c6_tracking_vs_smoothness.png"
    write_summary(summary_path, rows)
    pareto = pareto_names(rows)
    write_plot(plot_path, rows, set(pareto))

    print("\nC6 sweep complete")
    print(f"valid cases: {sum(is_valid(row) for row in rows)}/{len(rows)}")
    print(f"Pareto candidates: {', '.join(pareto) if pareto else 'none'}")
    print(f"summary: {summary_path}")
    print(f"plot: {plot_path}")

    if not rows or not is_valid(rows[0]):
        raise AssertionError("C6 baseline case failed; the sweep is not comparable to C5")
    if not pareto:
        raise AssertionError("C6 produced no valid candidate")


if __name__ == "__main__":
    main()