"""D3: run the frozen D1 comparison across multiple tracks."""

from __future__ import annotations

import argparse
import copy
import csv
from pathlib import Path

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
    return parser.parse_args()


def main():
    args = arguments()
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

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "multitrack_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=summaries[0].keys())
        writer.writeheader()
        writer.writerows(summaries)

    print(f"\nD3 results written to {OUT}")


if __name__ == "__main__":
    main()