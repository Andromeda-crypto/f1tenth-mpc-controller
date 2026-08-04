"""Validate and select immutable F1TENTH track-reference candidates.

The validator mirrors the Gym map rasterization, checks an oriented physical
vehicle rectangle on a densely sampled closed path, and never moves individual
waypoints.  If a requested raceline is unsafe, the paired supplied centerline
is evaluated as a separate candidate.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from scipy.ndimage import distance_transform_edt


@dataclass(frozen=True)
class CandidateReport:
    kind: str
    source: str
    points: int
    length_m: float
    minimum_footprint_clearance_m: float
    first_percentile_clearance_m: float
    samples_below_margin: int
    required_margin_m: float
    passed: bool


@dataclass(frozen=True)
class ReferenceSelection:
    x: np.ndarray
    y: np.ndarray
    yaw: np.ndarray
    speed: np.ndarray
    s: np.ndarray
    yaw_per_lap: float
    requested: CandidateReport
    selected: CandidateReport
    selected_kind: str
    selected_source: Path
    start_pose: tuple[float, float, float]


@dataclass(frozen=True)
class _MapGrid:
    clearance: np.ndarray
    resolution: float
    origin_x: float
    origin_y: float
    origin_yaw: float

    def sample(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        cosine, sine = np.cos(self.origin_yaw), np.sin(self.origin_yaw)
        dx, dy = x - self.origin_x, y - self.origin_y
        grid_x = (cosine * dx + sine * dy) / self.resolution
        grid_y = (-sine * dx + cosine * dy) / self.resolution
        columns, rows = grid_x.astype(int), grid_y.astype(int)
        valid = (
            (grid_x >= 0.0)
            & (grid_y >= 0.0)
            & (rows < self.clearance.shape[0])
            & (columns < self.clearance.shape[1])
        )
        values = np.full(x.shape, -np.inf, dtype=float)
        values[valid] = self.clearance[rows[valid], columns[valid]]
        return values


def _resolve(config_path: Path, value: str) -> Path:
    path = Path(value)
    return (config_path.parent / path).resolve() if not path.is_absolute() else path


def _map_paths(config: dict, config_path: Path) -> tuple[Path, Path]:
    base = _resolve(config_path, str(config["map_path"]))
    extension = str(config["map_ext"])
    if not extension.startswith("."):
        extension = "." + extension
    return base.with_suffix(".yaml"), base.with_suffix(extension)


def _load_map(config: dict, config_path: Path) -> _MapGrid:
    yaml_path, image_path = _map_paths(config, config_path)
    with yaml_path.open(encoding="utf-8") as handle:
        metadata = yaml.safe_load(handle)
    image = np.flipud(np.asarray(Image.open(image_path).convert("L"), dtype=float))

    # This deliberately matches ScanSimulator2D.set_map(): pixels <= 128 are
    # occupied and pixels > 128 are free.  YAML occupancy thresholds are not
    # used by the Gym collision scanner.
    free = image > 128.0
    resolution = float(metadata["resolution"])
    clearance = distance_transform_edt(free) * resolution
    origin_x, origin_y, origin_yaw = map(float, metadata["origin"])
    return _MapGrid(clearance, resolution, origin_x, origin_y, origin_yaw)


def _open_loop(x: np.ndarray, y: np.ndarray, *fields: np.ndarray) -> tuple[np.ndarray, ...]:
    arrays = [np.asarray(value, dtype=float).copy() for value in (x, y, *fields)]
    if any(value.ndim != 1 for value in arrays) or len({value.size for value in arrays}) != 1:
        raise ValueError("track fields must be one-dimensional and equal length")
    if arrays[0].size < 3 or not all(np.all(np.isfinite(value)) for value in arrays):
        raise ValueError("track contains too few points or non-finite values")
    if np.hypot(arrays[0][-1] - arrays[0][0], arrays[1][-1] - arrays[1][0]) < 1e-6:
        arrays = [value[:-1] for value in arrays]
    separation = np.hypot(np.diff(arrays[0]), np.diff(arrays[1]))
    keep = np.r_[True, separation > 1e-9]
    arrays = [value[keep] for value in arrays]
    if arrays[0].size < 3:
        raise ValueError("track collapses after duplicate points are removed")
    return tuple(arrays)


def _segments(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    length = np.hypot(np.roll(x, -1) - x, np.roll(y, -1) - y)
    if np.any(length <= 1e-9):
        raise ValueError("closed track contains a zero-length segment")
    return length


def _resample(x: np.ndarray, y: np.ndarray, spacing: float) -> tuple[np.ndarray, np.ndarray]:
    x, y = _open_loop(x, y)
    segment = _segments(x, y)
    nodes = np.r_[0.0, np.cumsum(segment)]
    count = max(3, int(np.ceil(nodes[-1] / spacing)))
    samples = np.linspace(0.0, nodes[-1], count, endpoint=False)
    return (
        np.interp(samples, nodes, np.r_[x, x[0]]),
        np.interp(samples, nodes, np.r_[y, y[0]]),
    )


def _geometry(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    segment = _segments(x, y)
    tangent_x, tangent_y = np.roll(x, -1) - np.roll(x, 1), np.roll(y, -1) - np.roll(y, 1)
    raw_yaw = np.arctan2(tangent_y, tangent_x)
    yaw = np.unwrap(raw_yaw)
    turn = np.sum(np.arctan2(np.sin(np.roll(raw_yaw, -1) - raw_yaw),
                             np.cos(np.roll(raw_yaw, -1) - raw_yaw)))
    yaw_per_lap = 2.0 * np.pi * round(turn / (2.0 * np.pi))
    if abs(yaw_per_lap) < np.pi:
        yaw_per_lap = float(turn)
    s = np.r_[0.0, np.cumsum(segment)]
    return yaw, s, float(yaw_per_lap)


def _curvature(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    outgoing = np.arctan2(np.roll(y, -1) - y, np.roll(x, -1) - x)
    incoming = np.arctan2(y - np.roll(y, 1), x - np.roll(x, 1))
    turn = np.arctan2(np.sin(outgoing - incoming), np.cos(outgoing - incoming))
    distance = 0.5 * (_segments(x, y) + np.roll(_segments(x, y), 1))
    return turn / distance


def _speed_profile(x: np.ndarray, y: np.ndarray, config: dict, maximum: float) -> np.ndarray:
    lateral_accel = float(config.get("reference_max_lateral_accel_mps2", 4.0))
    acceleration = float(config.get("reference_max_accel_mps2", 3.0))
    deceleration = float(config.get("reference_max_decel_mps2", 4.0))
    minimum = float(config.get("reference_min_speed_mps", 1.0))
    curvature = np.abs(_curvature(x, y))
    speed = np.minimum(maximum, np.sqrt(lateral_accel / np.maximum(curvature, 1e-6)))
    speed = np.maximum(speed, minimum)
    segment = _segments(x, y)
    for _ in range(20):
        before = speed.copy()
        for index in range(speed.size):
            following = (index + 1) % speed.size
            limit = np.sqrt(speed[index] ** 2 + 2.0 * acceleration * segment[index])
            speed[following] = min(speed[following], limit)
        for index in range(speed.size - 1, -1, -1):
            following = (index + 1) % speed.size
            limit = np.sqrt(speed[following] ** 2 + 2.0 * deceleration * segment[index])
            speed[index] = min(speed[index], limit)
        if np.max(np.abs(speed - before)) < 1e-6:
            break
    return speed


def _footprint_clearance(
    grid: _MapGrid,
    x: np.ndarray,
    y: np.ndarray,
    yaw: np.ndarray,
    length: float,
    width: float,
) -> np.ndarray:
    sample_step = 0.5 * grid.resolution
    longitudinal = np.linspace(-0.5 * length, 0.5 * length,
                               max(3, int(np.ceil(length / sample_step)) + 1))
    lateral = np.linspace(-0.5 * width, 0.5 * width,
                          max(3, int(np.ceil(width / sample_step)) + 1))
    local_x, local_y = np.meshgrid(longitudinal, lateral, indexing="ij")
    local_x, local_y = local_x.ravel(), local_y.ravel()
    result = np.empty(x.size, dtype=float)
    for start in range(0, x.size, 256):
        stop = min(x.size, start + 256)
        cosine, sine = np.cos(yaw[start:stop])[:, None], np.sin(yaw[start:stop])[:, None]
        world_x = x[start:stop, None] + cosine * local_x - sine * local_y
        world_y = y[start:stop, None] + sine * local_x + cosine * local_y
        result[start:stop] = np.min(grid.sample(world_x, world_y), axis=1)

    # Distance transforms measure to obstacle-cell centres.  Subtract half a
    # pixel diagonal so the reported clearance is conservative at cell edges.
    return result - grid.resolution / np.sqrt(2.0)


def _report(
    kind: str,
    source: Path,
    x: np.ndarray,
    y: np.ndarray,
    grid: _MapGrid,
    config: dict,
) -> tuple[CandidateReport, np.ndarray, np.ndarray, np.ndarray]:
    spacing = min(float(config.get("validation_sample_spacing_m", 0.05)),
                  0.5 * grid.resolution)
    dense_x, dense_y = _resample(x, y, spacing)
    dense_yaw, dense_s, _ = _geometry(dense_x, dense_y)
    clearance = _footprint_clearance(
        grid, dense_x, dense_y, dense_yaw,
        float(config.get("vehicle_length", 0.58)),
        float(config.get("vehicle_width", 0.31)),
    )
    margin = float(config.get("path_safety_margin_m", 0.20))
    below = int(np.count_nonzero(clearance < margin))
    report = CandidateReport(
        kind=kind,
        source=str(source),
        points=int(dense_x.size),
        length_m=float(dense_s[-1]),
        minimum_footprint_clearance_m=float(np.min(clearance)),
        first_percentile_clearance_m=float(np.percentile(clearance, 1.0)),
        samples_below_margin=below,
        required_margin_m=margin,
        passed=below == 0,
    )
    return report, dense_x, dense_y, clearance


def _centerline_path(config: dict, config_path: Path, waypoint_path: Path) -> Path | None:
    map_yaml, _ = _map_paths(config, config_path)
    prefix = map_yaml.stem
    if prefix.endswith("_map"):
        prefix = prefix[:-4]
    exact_candidates = [
        map_yaml.parent / f"{prefix}_centerline.csv",
        waypoint_path.parent / f"{prefix}_centerline.csv",
    ]
    for candidate in exact_candidates:
        if candidate.is_file():
            return candidate
    matches = sorted(map_yaml.parent.glob("*centerline.csv"))
    return matches[0] if len(matches) == 1 else None


def _safe_start(x: np.ndarray, y: np.ndarray, clearance: np.ndarray, settle_distance: float) -> int:
    segment = _segments(x, y)
    curvature = np.abs(_curvature(x, y))
    score = np.empty(x.size, dtype=float)
    for start in range(x.size):
        distance, index, minimum = 0.0, start, float("inf")
        while distance < settle_distance:
            minimum = min(minimum, float(clearance[index]))
            distance += float(segment[index])
            index = (index + 1) % x.size
        score[start] = minimum - 0.25 * curvature[start]
    return int(np.argmax(score))


def _rotate(x: np.ndarray, y: np.ndarray, speed: np.ndarray, index: int) -> tuple[np.ndarray, ...]:
    return tuple(np.r_[value[index:], value[:index]] for value in (x, y, speed))


def select_reference(
    config: dict,
    config_path: Path,
    waypoint_path: Path,
    data: np.ndarray,
) -> ReferenceSelection:
    """Validate the requested path and select a safe immutable candidate."""
    grid = _load_map(config, config_path)
    x_index, y_index = int(config.get("wpt_xind", 1)), int(config.get("wpt_yind", 2))
    speed_index = int(config.get("wpt_vind", 5))
    requested_x, requested_y, requested_speed = _open_loop(
        data[:, x_index], data[:, y_index], data[:, speed_index]
    )
    requested, _, _, _ = _report(
        "raceline", waypoint_path, requested_x, requested_y, grid, config
    )
    selected, x, y, speed, source = requested, requested_x, requested_y, requested_speed, waypoint_path

    if not requested.passed:
        centerline = _centerline_path(config, config_path, waypoint_path)
        if centerline is None:
            raise ValueError(
                f"requested raceline failed validation: minimum footprint clearance "
                f"{requested.minimum_footprint_clearance_m:.3f} m < "
                f"{requested.required_margin_m:.3f} m, and no unique paired centerline was found"
            )
        center_data = np.loadtxt(centerline, delimiter=",", comments="#")
        if center_data.ndim != 2 or center_data.shape[1] < 2:
            raise ValueError(f"centerline file has an invalid shape: {center_data.shape}")
        spacing = float(config.get("reference_spacing_m", 0.20))
        x, y = _resample(center_data[:, 0], center_data[:, 1], spacing)
        maximum = float(config.get("reference_max_speed_mps", min(8.0, np.max(requested_speed))))
        speed = _speed_profile(x, y, config, maximum)
        selected, _, _, _ = _report("centerline", centerline, x, y, grid, config)
        source = centerline
        if not selected.passed:
            raise ValueError(
                f"both reference candidates failed validation; centerline minimum footprint "
                f"clearance is {selected.minimum_footprint_clearance_m:.3f} m"
            )

    # Re-evaluate the selected controller-resolution path to choose a long,
    # high-clearance settling region without changing its geometry.
    yaw_open, _, _ = _geometry(x, y)
    controller_clearance = _footprint_clearance(
        grid, x, y, yaw_open,
        float(config.get("vehicle_length", 0.58)),
        float(config.get("vehicle_width", 0.31)),
    )
    start = _safe_start(
        x, y, controller_clearance,
        float(config.get("start_settle_distance_m", 12.0)),
    )
    x, y, speed = _rotate(x, y, speed, start)
    yaw_open, s, yaw_per_lap = _geometry(x, y)
    yaw = np.r_[yaw_open, yaw_open[0] + yaw_per_lap]
    x, y, speed = np.r_[x, x[0]], np.r_[y, y[0]], np.r_[speed, speed[0]]
    start_pose = (float(x[0]), float(y[0]), float(yaw[0]))

    report_path = (
        Path(__file__).resolve().parents[1]
        / "results"
        / "track_validation"
        / f"{_map_paths(config, config_path)[0].stem}.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "map_raster_rule": "Gym-compatible grayscale > 128 is free",
                "vehicle_length_m": float(config.get("vehicle_length", 0.58)),
                "vehicle_width_m": float(config.get("vehicle_width", 0.31)),
                "requested": asdict(requested),
                "selected": asdict(selected),
                "selected_kind": selected.kind,
                "start_pose": {"x_m": start_pose[0], "y_m": start_pose[1], "yaw_rad": start_pose[2]},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"track validation: requested {requested.kind} "
        f"min={requested.minimum_footprint_clearance_m:.3f} m, "
        f"required={requested.required_margin_m:.3f} m -> "
        f"{'PASS' if requested.passed else 'REJECT'}"
    )
    if selected.kind != requested.kind:
        print(
            f"track validation: selected {selected.kind} from {source.name}; "
            f"min={selected.minimum_footprint_clearance_m:.3f} m -> PASS"
        )
    print(f"track validation report: {report_path}")
    return ReferenceSelection(
        x=x, y=y, yaw=yaw, speed=speed, s=s, yaw_per_lap=yaw_per_lap,
        requested=requested, selected=selected, selected_kind=selected.kind,
        selected_source=source, start_pose=start_pose,
    )
