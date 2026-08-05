"""D3: run the frozen D1 comparison across multiple tracks."""

from __future__ import annotations

import argparse
import copy
import csv
from pathlib import Path
import numpy as np 
import yaml

from experiments.c5_f1tenth_gym import find_default_config, load_track
from experiments.d1_gym_comparison import (
    calculate_summary,
    run_controller,
    write_telemetry,
)
from f1tenth_mpc.mpc_qp import MPCConfig


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "d3"

TRACKS = {
    "silverstone": ("silverstone/Silverstone_map", "Silverstone_raceline.csv"),
    "spielberg": ("spielberg/Spielberg_map", "Spielberg_raceline.csv"),
    "monza": ("monza/Monza_map", "Monza_raceline.csv"),
    "nuerburgring": (
        "nuerburgring/Nuerburgring_map",
        "Nuerburgring_raceline.csv",
    ),
    "zandvoort": ("zandvoort/Zandvoort_map", "Zandvoort_raceline.csv"),
}


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", choices=TRACKS)
    parser.add_argument(
        "--controller",
        choices=("both", "mpc", "pure-pursuit"),
        default="both",
    )
    parser.add_argument("--max-time", type=float, default=120.0)
    parser.add_argument("--summarize-existing",action="store_true",help="rebuild the aggregate summary from saved telemetry without simulation")
    return parser.parse_args()

def summarize_telemetry(
    track_name: str,
    controller: str,
    path: Path,
    control_dt: float = 0.1,
) -> dict[str, float | int | str | bool]:
    with path.open(newline="", encoding="utf-8") as handle:
        telemetry = list(csv.DictReader(handle))

    if not telemetry:
        raise RuntimeError(f"empty telemetry file: {path}")

    def values(key: str) -> np.ndarray:
        return np.asarray([float(row[key]) for row in telemetry], dtype=float)

    cross_track = values("cross_track_error_m")
    heading = values("heading_error_rad")
    steering = values("steer_command_rad")
    speed = values("speed_mps")
    update_times = values("controller_update_ms")

    steering_change = np.diff(np.r_[0.0, steering])
    final = telemetry[-1]
    termination_reason = final["termination_reason"]
    collision = any(int(float(row["collision"])) != 0 for row in telemetry)

    return {
        "track": track_name,
        "controller": controller,
        "lap_completed": termination_reason == "lap_complete",
        "collision": collision,
        "termination_reason": termination_reason,
        "lap_time_s": float(final["time_s"]),
        "control_updates": len(telemetry),
        "cross_track_rmse_m": float(np.sqrt(np.mean(cross_track ** 2))),
        "cross_track_max_m": float(np.max(np.abs(cross_track))),
        "heading_rmse_deg": float(
            np.rad2deg(np.sqrt(np.mean(heading ** 2)))
        ),
        "heading_max_deg": float(
            np.rad2deg(np.max(np.abs(heading)))
        ),
        "mean_speed_mps": float(np.mean(speed)),
        "max_speed_mps": float(np.max(speed)),
        "steering_total_variation_rad": float(
            np.sum(np.abs(steering_change))
        ),
        "max_abs_steer_deg": float(
            np.rad2deg(np.max(np.abs(steering)))
        ),
        "median_controller_update_ms": float(np.median(update_times)),
        "max_controller_update_ms": float(np.max(update_times)),
        "deadline_misses_100ms": int(
            np.sum(update_times > control_dt * 1000.0)
        ),
    }


def rebuild_summary() -> list[dict[str, float | int | str | bool]]:
    rows = []

    controller_files = (
        ("MPC", "mpc_telemetry.csv"),
        ("Pure Pursuit", "pure_pursuit_telemetry.csv"),
    )

    for track_name in TRACKS:
        for controller, filename in controller_files:
            telemetry_path = OUT / track_name / filename
            if telemetry_path.exists():
                rows.append(
                    summarize_telemetry(
                        track_name,
                        controller,
                        telemetry_path,
                    )
                )

    if not rows:
        raise RuntimeError(f"no D3 telemetry files found under {OUT}")

    OUT.mkdir(parents=True, exist_ok=True)
    summary_path = OUT / "multitrack_summary.csv"

    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"rebuilt {summary_path} from {len(rows)} telemetry files")
    return rows

def main():
    args = arguments()
    if args.summarize_existing:
        rebuild_summary()
        return 
    
    config_path = find_default_config()

    with config_path.open(encoding="utf-8") as handle:
        base_config = yaml.safe_load(handle)

    selected = (
        {args.track: TRACKS[args.track]}
        if args.track
        else TRACKS
    )

    controllers = {
        "both": ("MPC", "Pure Pursuit"),
        "mpc": ("MPC",),
        "pure-pursuit": ("Pure Pursuit",),
    }[args.controller]

    mpc_config = MPCConfig(
        horizon=8,
        dt=0.1,
        max_speed=8.0,
        q=(4.0, 4.0, 1.0, 4.0),
        q_terminal=(8.0, 8.0, 2.0, 8.0),
        r=(0.1, 0.2),
        r_delta=(0.1, 1.0),
    )

    summaries = []

    for name, (map_path, waypoint_file) in selected.items():
        config = copy.deepcopy(base_config)
        config.update(
            map_path=f"../data/maps/{map_path}",
            map_ext=".png",
            wpt_path=f"../data/waypoints/{waypoint_file}",
            wpt_delim=";",
            wpt_rowskip=3,
        )

        track = load_track(config, config_path, None)

        # Start directly on the raceline, aligned with its tangent.
        config.update(
            sx=float(track.x[0]),
            sy=float(track.y[0]),
            stheta=float(track.yaw[0]),
        )

        for controller in controllers:
            print(f"\n{name}: running {controller}")

            result = run_controller(
                controller_name=controller,
                track=track,
                gym_config=config,
                config_path=config_path,
                mpc_config=mpc_config,
                wheelbase=0.3302,
                lookahead=1.2,
                solver="native",
                physics_dt=0.01,
                control_dt=0.1,
                max_time=args.max_time,
                render=True,
            )

            slug = controller.lower().replace(" ", "_")
            folder = OUT / name
            write_telemetry(
                folder / f"{slug}_telemetry.csv",
                result.telemetry,
            )

            row = {"track": name}
            row.update(calculate_summary(result, 0.1))
            summaries.append(row)

            print(
                f"collision={row['collision']}  "
                f"lap={row['lap_time_s']:.3f}s  "
                f"CTE={row['cross_track_rmse_m']:.4f}m"
            )

    rebuild_summary()
    print(f"\nD3 results written to {OUT}")


if __name__ == "__main__":
    main()