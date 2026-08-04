"""D2: generate comparison plots from the authoritative D1 telemetry.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from experiments.c5_f1tenth_gym import (
    find_default_config,
    load_track,
)


ROOT = Path(__file__).resolve().parents[1]
D1_DIR = ROOT / "results" / "d1"
OUTPUT_DIR = ROOT / "results" / "d2"


def load_csv(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run D1 with both controllers first."
        )

    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise RuntimeError(f"{path} contains no telemetry rows")

    data = {}

    for column in rows[0]:
        if column in ("controller", "controller_status"):
            data[column] = np.array([row[column] for row in rows])
        else:
            data[column] = np.array(
                [float(row[column]) for row in rows],
                dtype=float,
            )

    return data


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 220,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "legend.frameon": False,
            "lines.linewidth": 1.8,
        }
    )


def save_figure(figure: plt.Figure, filename: str) -> None:
    path = OUTPUT_DIR / filename
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    print(f"wrote {path}")


def plot_trajectory(
    mpc: dict[str, np.ndarray],
    pure_pursuit: dict[str, np.ndarray],
) -> None:
    config_path = find_default_config()

    with config_path.open("r", encoding="utf-8") as handle:
        gym_config = yaml.safe_load(handle)

    track = load_track(
        gym_config,
        config_path,
        waypoint_override=None,
    )

    figure, axis = plt.subplots(figsize=(8.0, 6.5))

    axis.plot(
        track.x,
        track.y,
        color="black",
        linestyle="--",
        linewidth=1.2,
        alpha=0.65,
        label="Reference path",
    )
    axis.plot(
        mpc["x_m"],
        mpc["y_m"],
        color="#1565C0",
        label="MPC",
    )
    axis.plot(
        pure_pursuit["x_m"],
        pure_pursuit["y_m"],
        color="#E65100",
        label="Pure Pursuit",
    )

    axis.scatter(
        [mpc["x_m"][0]],
        [mpc["y_m"][0]],
        color="black",
        marker="o",
        s=35,
        zorder=5,
        label="Start",
    )

    axis.set_title("Trajectory comparison")
    axis.set_xlabel("Global x [m]")
    axis.set_ylabel("Global y [m]")
    axis.set_aspect("equal", adjustable="box")
    axis.legend()

    save_figure(figure, "trajectory_overlay.png")


def plot_tracking_error(
    mpc: dict[str, np.ndarray],
    pure_pursuit: dict[str, np.ndarray],
) -> None:
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(9.0, 6.5),
        sharex=True,
    )

    axes[0].plot(
        mpc["time_s"],
        np.abs(mpc["cross_track_error_m"]),
        color="#1565C0",
        label="MPC",
    )
    axes[0].plot(
        pure_pursuit["time_s"],
        np.abs(pure_pursuit["cross_track_error_m"]),
        color="#E65100",
        label="Pure Pursuit",
    )
    axes[0].set_ylabel("|Cross-track error| [m]")
    axes[0].set_title("Tracking error over the lap")
    axes[0].legend()

    axes[1].plot(
        mpc["time_s"],
        np.rad2deg(np.abs(mpc["heading_error_rad"])),
        color="#1565C0",
        label="MPC",
    )
    axes[1].plot(
        pure_pursuit["time_s"],
        np.rad2deg(
            np.abs(pure_pursuit["heading_error_rad"])
        ),
        color="#E65100",
        label="Pure Pursuit",
    )
    axes[1].set_xlabel("Simulated time [s]")
    axes[1].set_ylabel("|Heading error| [deg]")

    save_figure(figure, "tracking_error_vs_time.png")


def plot_control_inputs(
    mpc: dict[str, np.ndarray],
    pure_pursuit: dict[str, np.ndarray],
) -> None:
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(9.0, 6.5),
        sharex=True,
    )

    axes[0].plot(
        mpc["time_s"],
        np.rad2deg(mpc["steer_command_rad"]),
        color="#1565C0",
        label="MPC",
    )
    axes[0].plot(
        pure_pursuit["time_s"],
        np.rad2deg(
            pure_pursuit["steer_command_rad"]
        ),
        color="#E65100",
        label="Pure Pursuit",
    )
    axes[0].set_title("Controller commands")
    axes[0].set_ylabel("Steering command [deg]")
    axes[0].legend()

    axes[1].plot(
        mpc["time_s"],
        mpc["speed_command_mps"],
        color="#1565C0",
        label="MPC",
    )
    axes[1].plot(
        pure_pursuit["time_s"],
        pure_pursuit["speed_command_mps"],
        color="#E65100",
        label="Pure Pursuit",
    )
    axes[1].set_xlabel("Simulated time [s]")
    axes[1].set_ylabel("Speed command [m/s]")

    save_figure(figure, "control_inputs_vs_time.png")


def plot_update_histogram(
    mpc: dict[str, np.ndarray],
    pure_pursuit: dict[str, np.ndarray],
) -> None:
    mpc_times = mpc["controller_update_ms"]
    pp_times = pure_pursuit["controller_update_ms"]

    positive_times = np.concatenate(
        [
            mpc_times[mpc_times > 0.0],
            pp_times[pp_times > 0.0],
        ]
    )

    lower = max(float(np.min(positive_times)) * 0.8, 1.0e-3)
    upper = float(np.max(positive_times)) * 1.2
    bins = np.logspace(
        np.log10(lower),
        np.log10(upper),
        28,
    )

    figure, axis = plt.subplots(figsize=(8.5, 5.5))

    axis.hist(
        mpc_times,
        bins=bins,
        alpha=0.65,
        color="#1565C0",
        label=(
            "MPC "
            f"(median {np.median(mpc_times):.3f} ms)"
        ),
    )
    axis.hist(
        pp_times,
        bins=bins,
        alpha=0.65,
        color="#E65100",
        label=(
            "Pure Pursuit "
            f"(median {np.median(pp_times):.3f} ms)"
        ),
    )

    axis.axvline(
        100.0,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label="100 ms deadline",
    )

    axis.set_xscale("log")
    axis.set_title("Controller update-time distribution")
    axis.set_xlabel("Controller update time [ms, logarithmic scale]")
    axis.set_ylabel("Control updates")
    axis.legend()

    save_figure(figure, "solve_time_histogram.png")


def validate_data(
    mpc: dict[str, np.ndarray],
    pure_pursuit: dict[str, np.ndarray],
) -> None:
    required_columns = {
        "time_s",
        "x_m",
        "y_m",
        "speed_command_mps",
        "steer_command_rad",
        "cross_track_error_m",
        "heading_error_rad",
        "controller_update_ms",
    }

    for name, data in (
        ("MPC", mpc),
        ("Pure Pursuit", pure_pursuit),
    ):
        missing = required_columns.difference(data)
        if missing:
            raise KeyError(
                f"{name} telemetry is missing columns: "
                f"{sorted(missing)}"
            )

        lengths = {
            len(data[column])
            for column in required_columns
        }

        if len(lengths) != 1:
            raise RuntimeError(
                f"{name} telemetry columns have inconsistent lengths"
            )


def main() -> None:
    configure_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    mpc = load_csv(D1_DIR / "mpc_telemetry.csv")
    pure_pursuit = load_csv(
        D1_DIR / "pure_pursuit_telemetry.csv"
    )

    validate_data(mpc, pure_pursuit)

    plot_trajectory(mpc, pure_pursuit)
    plot_tracking_error(mpc, pure_pursuit)
    plot_control_inputs(mpc, pure_pursuit)
    plot_update_histogram(mpc, pure_pursuit)

    print(
        "\nD2 complete: four comparison plots written to "
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()