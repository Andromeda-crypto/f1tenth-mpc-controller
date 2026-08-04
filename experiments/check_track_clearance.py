"""
Valdate clearnace aganist an F1tenth occupancy map.
"""

from __future__ import annotations
import argparse 
import csv
from pathlib import Path
import numpy as np
import yaml
from PIL import Image
from scipy.ndimage import distance_transform_edt

ROOT = Path(__file__).resolve().parents[1]


TRACKS = {
    "silverstone": {
        "map_dir": ROOT / "data" / "maps" / "silverstone",
        "map_yaml": "Silverstone_map.yaml",
        "waypoints": ROOT
        / "data"
        / "waypoints"
        / "Silverstone_raceline.csv",
    },
    "spielberg": {
        "map_dir": ROOT / "data" / "maps" / "spielberg",
        "map_yaml": "Spielberg_map.yaml",
        "waypoints": ROOT
        / "data"
        / "waypoints"
        / "Spielberg_raceline.csv",
    },
    "monza": {
        "map_dir": ROOT / "data" / "maps" / "monza",
        "map_yaml": "Monza_map.yaml",
        "waypoints": ROOT
        / "data"
        / "waypoints"
        / "Monza_raceline.csv",
    },
    "nuerburgring": {
        "map_dir": ROOT / "data" / "maps" / "nuerburgring",
        "map_yaml": "Nuerburgring_map.yaml",
        "waypoints": ROOT
        / "data"
        / "waypoints"
        / "Nuerburgring_raceline.csv",
    },
    "zandvoort": {
        "map_dir": ROOT / "data" / "maps" / "zandvoort",
        "map_yaml": "Zandvoort_map.yaml",
        "waypoints": ROOT
        / "data"
        / "waypoints"
        / "Zandvoort_raceline.csv",
    },
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure raceline distance from occupied map cells."
    )
    parser.add_argument(
        "--track",
        choices=TRACKS,
        default="silverstone"
    )
    return parser.parse_args()


def load_clearance_map(map_dir: Path,yaml_name: str) -> tuple[np.ndarray, float, float, float]:
    yaml_path = map_dir / yaml_name

    with yaml_path.open("r", encoding="utf-8") as handle:
        metadata = yaml.safe_load(handle)

    image_path = map_dir / metadata["image"]
    image = np.asarray(Image.open(image_path).convert("L"), dtype=float)

    resolution = float(metadata["resolution"])
    origin_x, origin_y, _ = map(float, metadata["origin"])
    occupied_threshold = float(metadata["occupied_thresh"])
    negate = int(metadata.get("negate", 0))

    normalized = image / 255.0

    if negate == 0:
        occupancy_probability = 1.0 - normalized
    else:
        occupancy_probability = normalized

    free_mask = occupancy_probability < occupied_threshold
    clearance = distance_transform_edt(free_mask) * resolution

    return clearance, resolution, origin_x, origin_y


def sample_clearance(clearance_map: np.ndarray,resolution: float,origin_x: float,origin_y: float,x: np.ndarray,y: np.ndarray) -> np.ndarray:
    columns = np.floor((x - origin_x) / resolution).astype(int)
    map_rows = np.floor((y - origin_y) / resolution).astype(int)
    rows = clearance_map.shape[0] - 1 - map_rows

    valid = (
        (rows >= 0)
        & (rows < clearance_map.shape[0])
        & (columns >= 0)
        & (columns < clearance_map.shape[1])
    )

    values = np.full(x.shape, np.nan, dtype=float)
    values[valid] = clearance_map[rows[valid], columns[valid]]
    return values


def load_waypoints(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.loadtxt(path,delimiter=";",skiprows=3)

    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError(f"Invalid raceline shape: {data.shape}")

    return data[:, 0], data[:, 1], data[:, 2]


def read_terminal_position(path: Path) -> tuple[float, float] | None:
    if not path.exists():
        return None

    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        return None

    final = rows[-1]
    return float(final["x_m"]), float(final["y_m"])


def print_terminal_clearance(
    label: str,
    telemetry_path: Path,
    clearance_map: np.ndarray,
    resolution: float,
    origin_x: float,
    origin_y: float
) -> None:
    position = read_terminal_position(telemetry_path)

    if position is None:
        return

    x, y = position
    clearance = sample_clearance(
        clearance_map,
        resolution,
        origin_x,
        origin_y,
        np.array([x]),
        np.array([y]),
    )[0]

    print(
        f"{label:22s}: "
        f"x={x:+.3f} m, y={y:+.3f} m, "
        f"centre clearance={clearance:.3f} m"
    )


def main() -> None:
    arguments = parse_arguments()
    track_name = arguments.track
    paths = TRACKS[track_name]

    clearance_map, resolution, origin_x, origin_y = load_clearance_map(paths["map_dir"],paths["map_yaml"])

    progress, x, y = load_waypoints(paths["waypoints"])
    clearances = sample_clearance(clearance_map,resolution,origin_x,origin_y,x,y)

    valid = np.isfinite(clearances)
    if not np.all(valid):
        invalid_count = int(np.count_nonzero(~valid))
        raise ValueError(
            f"{invalid_count} raceline points lie outside the map image"
        )

    minimum_index = int(np.argmin(clearances))
    percentiles = np.percentile(clearances, [1, 5, 50])

    print(f"\nTrack: {track_name}")
    print(f"Raceline points: {len(x)}")
    print(f"Map resolution: {resolution:.4f} m/pixel")
    print(
        "Minimum centre clearance: "
        f"{clearances[minimum_index]:.3f} m "
        f"at progress {progress[minimum_index]:.3f} m"
    )
    print(
        "Minimum-clearance position: "
        f"x={x[minimum_index]:+.3f} m, "
        f"y={y[minimum_index]:+.3f} m"
    )
    print(f"1st percentile clearance: {percentiles[0]:.3f} m")
    print(f"5th percentile clearance: {percentiles[1]:.3f} m")
    print(f"Median clearance: {percentiles[2]:.3f} m")

    for threshold in (0.20, 0.25, 0.30, 0.35):
        count = int(np.count_nonzero(clearances < threshold))
        percentage = 100.0 * count / len(clearances)
        print(
            f"Points below {threshold:.2f} m: "
            f"{count} ({percentage:.2f}%)"
        )

    telemetry_dir = ROOT / "results" / "d3" / track_name

    print("\nTerminal telemetry positions:")
    print_terminal_clearance("MPC",telemetry_dir / "mpc_telemetry.csv",clearance_map,resolution,origin_x,origin_y)
    print_terminal_clearance("Pure Pursuit",telemetry_dir / "pure_pursuit_telemetry.csv",clearance_map,resolution,origin_x,origin_y)


if __name__ == "__main__":
    main()