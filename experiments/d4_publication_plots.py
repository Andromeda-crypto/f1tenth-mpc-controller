"""D4: publication-quality figures from frozen D3 results."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
D3_DIR = ROOT / "results" / "d3"
OUTPUT_DIR = ROOT / "results" / "d4"

TRACKS = (
    "silverstone",
    "spielberg",
    "monza",
    "nuerburgring",
    "zandvoort",
)

TRACK_LABELS = {
    "silverstone": "Silverstone",
    "spielberg": "Spielberg",
    "monza": "Monza",
    "nuerburgring": "Nürburgring",
    "zandvoort": "Zandvoort",
}

COLORS = {
    "MPC": "#1565C0",
    "Pure Pursuit": "#E65100",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "legend.frameon": False,
            "lines.linewidth": 1.7,
        }
    )


def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise RuntimeError(f"{path} contains no rows")

    return rows


def column(rows: list[dict[str, str]], name: str) -> np.ndarray:
    return np.asarray([float(row[name]) for row in rows], dtype=float)


def save_figure(figure: plt.Figure, stem: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()

    for extension in ("png", "svg"):
        path = OUTPUT_DIR / f"{stem}.{extension}"
        figure.savefig(path, bbox_inches="tight")
        print(f"wrote {path}")

    plt.close(figure)


def summary_lookup() -> dict[tuple[str, str], dict[str, str]]:
    rows = load_csv(D3_DIR / "multitrack_summary.csv")
    return {
        (row["track"], row["controller"]): row
        for row in rows
    }


def plot_multitrack_overview() -> None:
    summary = summary_lookup()
    labels = [TRACK_LABELS[track] for track in TRACKS]
    x = np.arange(len(TRACKS))
    width = 0.36

    metrics = (
        ("lap_time_s", "Lap time [s]", "Lap time"),
        (
            "cross_track_rmse_m",
            "Cross-track RMSE [m]",
            "Tracking accuracy",
        ),
        (
            "heading_rmse_deg",
            "Heading RMSE [deg]",
            "Heading accuracy",
        ),
        (
            "steering_total_variation_rad",
            "Steering total variation [rad]",
            "Control smoothness",
        ),
    )

    figure, axes = plt.subplots(2, 2, figsize=(12.0, 8.0))
    axes = axes.ravel()

    for axis, (key, ylabel, title) in zip(axes, metrics):
        mpc = np.asarray(
            [float(summary[(track, "MPC")][key]) for track in TRACKS]
        )
        pure_pursuit = np.asarray(
            [
                float(summary[(track, "Pure Pursuit")][key])
                for track in TRACKS
            ]
        )

        axis.bar(
            x - width / 2,
            mpc,
            width,
            color=COLORS["MPC"],
            label="MPC",
        )
        axis.bar(
            x + width / 2,
            pure_pursuit,
            width,
            color=COLORS["Pure Pursuit"],
            label="Pure Pursuit",
        )

        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_xticks(x)
        axis.set_xticklabels(labels, rotation=20, ha="right")

    axes[0].legend(loc="upper right")

    figure.suptitle(
        "MPC vs Pure Pursuit across five F1TENTH tracks\n"
        "Both controllers: 5/5 laps completed, zero collisions",
        fontsize=14,
        fontweight="bold",
    )
    figure.subplots_adjust(top=0.88)

    save_figure(figure, "multitrack_overview")


def load_track_telemetry(
    track: str,
    filename: str,
) -> list[dict[str, str]]:
    return load_csv(D3_DIR / track / filename)


def plot_trajectory_grid() -> None:
    figure, axes = plt.subplots(2, 3, figsize=(14.0, 8.5))
    axes = axes.ravel()

    for axis, track in zip(axes, TRACKS):
        mpc = load_track_telemetry(track, "mpc_telemetry.csv")
        pure_pursuit = load_track_telemetry(
            track,
            "pure_pursuit_telemetry.csv",
        )

        axis.plot(
            column(mpc, "x_m"),
            column(mpc, "y_m"),
            color=COLORS["MPC"],
            label="MPC",
        )
        axis.plot(
            column(pure_pursuit, "x_m"),
            column(pure_pursuit, "y_m"),
            color=COLORS["Pure Pursuit"],
            label="Pure Pursuit",
            alpha=0.85,
        )

        axis.scatter(
            [float(mpc[0]["x_m"])],
            [float(mpc[0]["y_m"])],
            s=22,
            color="black",
            zorder=4,
        )

        axis.set_title(TRACK_LABELS[track])
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("x [m]")
        axis.set_ylabel("y [m]")

    axes[-1].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[-1].legend(
        handles,
        labels,
        loc="center",
        fontsize=12,
    )

    figure.suptitle(
        "Collision-free closed-loop trajectories",
        fontsize=14,
        fontweight="bold",
    )
    figure.subplots_adjust(top=0.91)

    save_figure(figure, "trajectory_grid")


def plot_steering_case_study(
    track: str = "nuerburgring",
) -> None:
    mpc = load_track_telemetry(track, "mpc_telemetry.csv")
    pure_pursuit = load_track_telemetry(
        track,
        "pure_pursuit_telemetry.csv",
    )

    figure, axes = plt.subplots(
        3,
        1,
        figsize=(11.0, 8.5),
        sharex=True,
    )

    for rows, controller in (
        (mpc, "MPC"),
        (pure_pursuit, "Pure Pursuit"),
    ):
        time = column(rows, "time_s")
        color = COLORS[controller]

        axes[0].plot(
            time,
            column(rows, "cross_track_error_m"),
            color=color,
            label=controller,
        )
        axes[1].plot(
            time,
            np.rad2deg(column(rows, "heading_error_rad")),
            color=color,
            label=controller,
        )
        axes[2].plot(
            time,
            np.rad2deg(column(rows, "steer_command_rad")),
            color=color,
            label=controller,
        )

    axes[0].set_ylabel("Cross-track\nerror [m]")
    axes[0].legend(ncol=2)

    axes[1].set_ylabel("Heading\nerror [deg]")

    axes[2].set_ylabel("Steering\ncommand [deg]")
    axes[2].set_xlabel("Simulated time [s]")

    figure.suptitle(
        f"{TRACK_LABELS[track]} telemetry: tracking and control activity",
        fontsize=14,
        fontweight="bold",
    )
    figure.subplots_adjust(top=0.92)

    save_figure(figure, "steering_case_study")


def main() -> None:
    configure_style()
    plot_multitrack_overview()
    plot_trajectory_grid()
    plot_steering_case_study()

    print(f"\nD4 complete: publication figures written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()