"""C5: run the linear time-varying MPC controller in F1TENTH Gym.

Keep this file beside ``mpc_qp.py`` and ``c4_forward_sim.py`` and run it from
the repository root:

    python c5_f1tenth_gym.py

MPC state: z = [x, y, v, psi]
MPC input: u = [acceleration, steering_angle]
Gym action: [steering_angle, desired_speed]
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import numpy as np
import yaml

from c4_forward_sim import IterativeLinearMPC, wrap_angle
from mpc_qp import MPCConfig

SCRIPT_REVISION = "C5 geometric-heading fix v2"


@dataclass(frozen=True)
class ClosedTrack:
    x: np.ndarray
    y: np.ndarray
    yaw: np.ndarray
    speed: np.ndarray
    s: np.ndarray
    yaw_per_lap: float

    @property
    def length(self) -> float:
        return float(self.s[-1])

    def nearest_index(self, x: float, y: float) -> int:
        return int(np.argmin((self.x - x) ** 2 + (self.y - y) ** 2))

    def update_progress(self, state: np.ndarray, previous: float | None) -> float:
        """Return monotonically increasing progress, including lap wraparound."""
        nearest_s = float(self.s[self.nearest_index(state[0], state[1])])
        if previous is None:
            return nearest_s

        candidate = nearest_s + round((previous - nearest_s) / self.length) * self.length
        if candidate < previous - 0.5 * self.length:
            candidate += self.length
        return max(previous, candidate)

    def reference_horizon(
        self,
        state: np.ndarray,
        progress: float,
        horizon: int,
        dt: float,
    ) -> np.ndarray:
        """Build a horizon that continues smoothly through the start/finish line."""
        progress_mod = progress % self.length
        reference_speed = float(np.interp(progress_mod, self.s, self.speed))
        sample_progress = progress + np.arange(horizon + 1) * reference_speed * dt
        sample_s = np.mod(sample_progress, self.length)
        lap_number = np.floor(sample_progress / self.length)

        yaw = np.interp(sample_s, self.s, self.yaw) + lap_number * self.yaw_per_lap
        yaw += 2.0 * np.pi * np.round((state[3] - yaw[0]) / (2.0 * np.pi))

        return np.vstack(
            [
                np.interp(sample_s, self.s, self.x),
                np.interp(sample_s, self.s, self.y),
                np.interp(sample_s, self.s, self.speed),
                yaw,
            ]
        )

    def errors(self, state: np.ndarray) -> tuple[float, float]:
        index = self.nearest_index(state[0], state[1])
        cross_track = float(np.hypot(state[0] - self.x[index], state[1] - self.y[index]))
        heading = abs(float(wrap_angle(state[3] - self.yaw[index])))
        return cross_track, heading


def _resolve_from_config(config_path: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve()


def find_default_config() -> Path:
    candidates = (
        Path("config_example_map.yaml"),
        Path("f1tenth_gym/examples/config_example_map.yaml"),
        Path("../f1tenth_gym/examples/config_example_map.yaml"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "Could not find config_example_map.yaml. Pass its path with --config."
    )


def load_track(config: dict, config_path: Path, waypoint_override: str | None) -> ClosedTrack:
    if waypoint_override is None:
        if "wpt_path" not in config:
            raise KeyError("config has no wpt_path; pass the waypoint CSV with --waypoints")
        waypoint_path = _resolve_from_config(config_path, str(config["wpt_path"]))
    else:
        waypoint_path = Path(waypoint_override).resolve()

    delimiter = str(config.get("wpt_delim", ";"))
    skip_rows = int(config.get("wpt_rowskip", 0))
    data = np.loadtxt(waypoint_path, delimiter=delimiter, skiprows=skip_rows)
    if data.ndim != 2 or data.shape[0] < 3:
        raise ValueError(f"waypoint file has an invalid shape: {data.shape}")

    s_index = int(config.get("wpt_sind", 0))
    x_index = int(config.get("wpt_xind", 1))
    y_index = int(config.get("wpt_yind", 2))
    yaw_index = int(config.get("wpt_psiind", config.get("wpt_thind", 3)))
    speed_index = int(config.get("wpt_vind", 5))

    s = np.asarray(data[:, s_index], dtype=float)
    x = np.asarray(data[:, x_index], dtype=float)
    y = np.asarray(data[:, y_index], dtype=float)
    speed = np.asarray(data[:, speed_index], dtype=float)
    raw_yaw = np.unwrap(np.asarray(data[:, yaw_index], dtype=float))

    s = s - s[0]
    if np.any(np.diff(s) <= 0.0):
        s = np.r_[0.0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))]
    if s[-1] <= 0.0:
        raise ValueError("waypoint path length must be positive")

    # Raceline CSVs are not consistent about their heading convention.  In the
    # official example, psi is rotated by roughly 90 degrees from the Gym's
    # world-frame vehicle yaw.  The path tangent is convention-independent and
    # is the heading that a path-tracking controller should follow.
    tangent_x = np.gradient(x, s, edge_order=2)
    tangent_y = np.gradient(y, s, edge_order=2)
    yaw = np.unwrap(np.arctan2(tangent_y, tangent_x))
    yaw_offset = float(
        np.arctan2(
            np.mean(np.sin(yaw - raw_yaw)),
            np.mean(np.cos(yaw - raw_yaw)),
        )
    )

    # A closed planar lap normally changes unwrapped heading by +/- 2*pi.
    yaw_per_lap = 2.0 * np.pi * round((yaw[-1] - yaw[0]) / (2.0 * np.pi))
    if abs(yaw_per_lap) < np.pi:
        yaw_per_lap = float(yaw[-1] - yaw[0])

    print(f"loaded {len(x)} waypoints from {waypoint_path}")
    print(f"track length: {s[-1]:.2f} m; reference speed: {speed.min():.2f}-{speed.max():.2f} m/s")
    print(
        "CSV heading-to-path-tangent offset: "
        f"{np.rad2deg(yaw_offset):+.2f} deg; using geometric path tangent"
    )
    return ClosedTrack(x=x, y=y, yaw=yaw, speed=speed, s=s, yaw_per_lap=yaw_per_lap)


def make_environment(config: dict, config_path: Path, physics_dt: float):
    try:
        import gym
        from f110_gym.envs.base_classes import Integrator
    except ImportError as error:
        raise ImportError(
            "F1TENTH Gym is not importable in this Python environment."
        ) from error

    map_path = _resolve_from_config(config_path, str(config["map_path"]))
    return gym.make(
        "f110_gym:f110-v0",
        map=str(map_path),
        map_ext=str(config["map_ext"]),
        num_agents=1,
        timestep=physics_dt,
        integrator=Integrator.RK4,
    )


def reset_environment(env, initial_pose: np.ndarray):
    result = env.reset(initial_pose)
    if isinstance(result, tuple) and len(result) == 4:
        return result
    if isinstance(result, tuple) and len(result) == 2:
        observation, info = result
        return observation, 0.0, False, info
    raise RuntimeError("unexpected F1TENTH Gym reset return format")


def step_environment(env, action: np.ndarray):
    result = env.step(action)
    if isinstance(result, tuple) and len(result) == 4:
        return result
    if isinstance(result, tuple) and len(result) == 5:
        observation, reward, terminated, truncated, info = result
        return observation, reward, bool(terminated or truncated), info
    raise RuntimeError("unexpected F1TENTH Gym step return format")


class ContinuousYaw:
    def __init__(self) -> None:
        self.raw: float | None = None
        self.value: float | None = None

    def update(self, raw_yaw: float) -> float:
        if self.raw is None or self.value is None:
            self.raw = raw_yaw
            self.value = raw_yaw
        else:
            self.value += float(wrap_angle(raw_yaw - self.raw))
            self.raw = raw_yaw
        return self.value


def observation_state(observation: dict, yaw_tracker: ContinuousYaw) -> np.ndarray:
    return np.array(
        [
            float(observation["poses_x"][0]),
            float(observation["poses_y"][0]),
            max(0.0, float(observation["linear_vels_x"][0])),
            yaw_tracker.update(float(observation["poses_theta"][0])),
        ]
    )


def render_environment(env) -> None:
    try:
        env.render(mode="human_fast")
    except TypeError:
        env.render()


def lap_count(observation: dict) -> int:
    values = observation.get("lap_counts")
    return 0 if values is None else int(values[0])


def collision_detected(observation: dict) -> bool:
    values = observation.get("collisions")
    return False if values is None else bool(values[0])


def write_telemetry(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run C5 MPC in F1TENTH Gym")
    parser.add_argument("--config", help="path to config_example_map.yaml")
    parser.add_argument("--waypoints", help="optional waypoint CSV override")
    parser.add_argument("--no-render", action="store_true", help="disable the Gym window")
    parser.add_argument("--max-time", type=float, default=60.0, help="maximum simulated seconds")
    parser.add_argument("--wheelbase", type=float, default=0.3302)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    print(SCRIPT_REVISION)
    config_path = Path(arguments.config).resolve() if arguments.config else find_default_config()
    with config_path.open("r", encoding="utf-8") as handle:
        gym_config = yaml.safe_load(handle)

    track = load_track(gym_config, config_path, arguments.waypoints)
    physics_dt = 0.01
    control_dt = 0.1
    action_hold_steps = round(control_dt / physics_dt)
    if not np.isclose(action_hold_steps * physics_dt, control_dt):
        raise ValueError("control_dt must be an integer multiple of physics_dt")

    mpc_config = MPCConfig(
        horizon=8,
        dt=control_dt,
        max_speed=8.0,
        # Final C6 configuration: heading_2x.
        q=(4.0, 4.0, 1.0, 4.0),
        q_terminal=(8.0, 8.0, 2.0, 8.0),
        r=(0.1, 0.2),
        r_delta=(0.1, 1.0),
    )
    controller = IterativeLinearMPC(mpc_config, arguments.wheelbase)
    env = make_environment(gym_config, config_path, physics_dt)

    initial_pose = np.array(
        [[float(gym_config["sx"]), float(gym_config["sy"]), float(gym_config["stheta"])]]
    )
    observation, reward, done, _ = reset_environment(env, initial_pose)
    elapsed_time = float(reward)
    yaw_tracker = ContinuousYaw()
    progress: float | None = None
    action = np.zeros((1, 2), dtype=float)
    telemetry: list[dict[str, float | int | str]] = []
    update_times: list[float] = []
    steering_commands: list[float] = []
    update_number = 0

    try:
        if not arguments.no_render:
            render_environment(env)

        while not done and elapsed_time < arguments.max_time and lap_count(observation) < 1:
            state = observation_state(observation, yaw_tracker)
            progress = track.update_progress(state, progress)
            reference = track.reference_horizon(
                state, progress, mpc_config.horizon, mpc_config.dt
            )
            control, update_ms, status, iterations = controller.command(state, reference)

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
            telemetry.append(
                {
                    "time_s": elapsed_time,
                    "x_m": state[0],
                    "y_m": state[1],
                    "speed_mps": state[2],
                    "yaw_rad": state[3],
                    "accel_command_mps2": acceleration,
                    "steer_command_rad": steering,
                    "speed_command_mps": desired_speed,
                    "cross_track_error_m": cross_track,
                    "heading_error_rad": heading_error,
                    "mpc_update_ms": update_ms,
                    "mpc_iterations": iterations,
                    "solver_status": status,
                    "progress_m": progress,
                    "lap_count": lap_count(observation),
                }
            )
            update_times.append(update_ms)
            steering_commands.append(float(steering))
            update_number += 1

            if update_number % 20 == 0:
                print(
                    f"t={elapsed_time:6.2f} s  progress={progress:7.2f} m  "
                    f"CTE={cross_track:.3f} m  speed={state[2]:.2f} m/s  "
                    f"update={update_ms:.1f} ms"
                )

            for _ in range(action_hold_steps):
                observation, step_reward, done, _ = step_environment(env, action.copy())
                elapsed_time += float(step_reward)
                if not arguments.no_render:
                    render_environment(env)
                if done or lap_count(observation) >= 1:
                    break
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()

    telemetry_path = Path(__file__).resolve().parent / "c5_telemetry.csv"
    write_telemetry(telemetry_path, telemetry)

    if not telemetry:
        raise RuntimeError("C5 ended before the first MPC control update")

    cross_track_values = np.array([row["cross_track_error_m"] for row in telemetry], dtype=float)
    heading_values = np.array([row["heading_error_rad"] for row in telemetry], dtype=float)
    steering_change = np.diff(np.r_[0.0, np.asarray(steering_commands)])
    completed = lap_count(observation) >= 1
    collided = collision_detected(observation)

    print("\nC5 F1TENTH Gym result")
    print(f"lap completed: {completed}")
    print(f"collision: {collided}")
    print(f"simulated time: {elapsed_time:.3f} s")
    print(f"CTE RMSE: {np.sqrt(np.mean(cross_track_values**2)):.4f} m")
    print(f"heading RMSE: {np.rad2deg(np.sqrt(np.mean(heading_values**2))):.3f} deg")
    print(f"steering total variation: {np.sum(np.abs(steering_change)):.4f} rad")
    print(f"median MPC update: {median(update_times):.3f} ms")
    print(f"telemetry: {telemetry_path}")

    if collided:
        raise AssertionError("C5 failed: the vehicle collided")
    if not completed:
        raise AssertionError("C5 incomplete: no lap within the maximum simulated time")
    print("C5 validation passed")


if __name__ == "__main__":
    main()
