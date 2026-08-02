import csv
import time
from argparse import Namespace
from pathlib import Path

import gym
import matplotlib.pyplot as plt
import numpy as np
import yaml

from f110_gym.envs.base_classes import Integrator

from f1tenth_mpc.longitudinal_pid import LongitudinalPID
from f1tenth_mpc.path_loader import load_path
from f1tenth_mpc.pure_pursuit import PurePursuitController


PROJECT_ROOT = Path(__file__).resolve().parent
EXAMPLES_DIR = PROJECT_ROOT / "f1tenth_gym" / "examples"
RESULTS_DIR = PROJECT_ROOT / "results"


def main():
    config_file = EXAMPLES_DIR / "config_example_map.yaml"
    waypoint_file = EXAMPLES_DIR / "example_waypoints.csv"

    telemetry_file = RESULTS_DIR / "pure_pursuit_telemetry.csv"
    plot_file = RESULTS_DIR / "pure_pursuit_speed_tracking.png"

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with config_file.open("r") as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)

    conf = Namespace(**conf_dict)

    # Resolve the map path independently of the terminal directory.
    map_path = Path(conf.map_path)

    if not map_path.is_absolute():
        map_path = EXAMPLES_DIR / map_path

    path = load_path(waypoint_file)

    timestep = 0.01

    lateral_controller = PurePursuitController(
        wheelbase=0.17145 + 0.15875,
        lookahead_distance=1.5,
        max_steering_angle=np.deg2rad(24.0),
    )

    speed_controller = LongitudinalPID(
        kp=0.35,
        ki=0.08,
        kd=0.02,
        dt=timestep,
        output_limits=(-1.0, 1.0),
        integral_limits=(-2.0, 2.0),
        derivative_filter=0.2,
    )

    env = gym.make(
        "f110_gym:f110-v0",
        map=str(map_path),
        map_ext=conf.map_ext,
        num_agents=1,
        timestep=timestep,
        integrator=Integrator.RK4,
    )

    obs, step_reward, done, info = env.reset(
        np.array([[conf.sx, conf.sy, conf.stheta]])
    )

    env.render()

    reference_speed = 2.0
    minimum_speed_command = 0.0
    maximum_speed_command = 3.0

    elapsed_simulation_time = 0.0
    start_time = time.time()

    time_history = []
    reference_history = []
    measured_history = []
    command_history = []
    error_history = []

    telemetry_fields = [
        "time_s",
        "x_m",
        "y_m",
        "heading_rad",
        "measured_speed_mps",
        "reference_speed_mps",
        "commanded_speed_mps",
        "speed_correction_mps",
        "speed_error_mps",
        "steering_angle_rad",
        "cross_track_error_m",
        "nearest_waypoint_index",
        "target_waypoint_index",
        "collision",
        "lap_count",
    ]

    telemetry_rows = []

    try:
        while not done:
            current_x = float(obs["poses_x"][0])
            current_y = float(obs["poses_y"][0])
            current_heading = float(obs["poses_theta"][0])
            current_speed = float(obs["linear_vels_x"][0])

            state = np.array(
                [
                    current_x,
                    current_y,
                    current_heading,
                    current_speed,
                ]
            )

            steering_angle, target_index, nearest_index = (
                lateral_controller.compute_steering(state, path)
            )

            speed_correction, speed_error = speed_controller.update(
                target_speed=reference_speed,
                current_speed=current_speed,
            )

            commanded_speed = float(
                np.clip(
                    reference_speed + speed_correction,
                    minimum_speed_command,
                    maximum_speed_command,
                )
            )

            nearest_x = float(path["x"][nearest_index])
            nearest_y = float(path["y"][nearest_index])

            cross_track_error = float(
                np.hypot(
                    current_x - nearest_x,
                    current_y - nearest_y,
                )
            )

            action = np.array(
                [[steering_angle, commanded_speed]],
                dtype=float,
            )

            obs, step_reward, done, info = env.step(action)
            elapsed_simulation_time += step_reward

            collision = bool(obs["collisions"][0])

            lap_count = (
                int(obs["lap_counts"][0])
                if "lap_counts" in obs
                else 0
            )

            time_history.append(elapsed_simulation_time)
            reference_history.append(reference_speed)
            measured_history.append(current_speed)
            command_history.append(commanded_speed)
            error_history.append(speed_error)

            telemetry_rows.append(
                {
                    "time_s": elapsed_simulation_time,
                    "x_m": current_x,
                    "y_m": current_y,
                    "heading_rad": current_heading,
                    "measured_speed_mps": current_speed,
                    "reference_speed_mps": reference_speed,
                    "commanded_speed_mps": commanded_speed,
                    "speed_correction_mps": speed_correction,
                    "speed_error_mps": speed_error,
                    "steering_angle_rad": steering_angle,
                    "cross_track_error_m": cross_track_error,
                    "nearest_waypoint_index": nearest_index,
                    "target_waypoint_index": target_index,
                    "collision": collision,
                    "lap_count": lap_count,
                }
            )

            env.render(mode="human")

    except Exception as error:
        # Closing the rendering window raises an exception in this Gym version.
        if "Rendering window was closed" not in str(error):
            raise

        print("Rendering window was closed.")

    finally:
        env.close()

    if not telemetry_rows:
        raise RuntimeError(
            "The simulation ended before any telemetry was recorded."
        )

    with telemetry_file.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=telemetry_fields,
        )
        writer.writeheader()
        writer.writerows(telemetry_rows)

    speed_errors = np.asarray(error_history, dtype=float)

    speed_rmse = float(
        np.sqrt(np.mean(speed_errors**2))
    )
    speed_mae = float(
        np.mean(np.abs(speed_errors))
    )
    maximum_speed_error = float(
        np.max(np.abs(speed_errors))
    )

    final_collision = bool(obs["collisions"][0])

    final_lap_count = (
        int(obs["lap_counts"][0])
        if "lap_counts" in obs
        else 0
    )

    print(f"Simulation time: {elapsed_simulation_time:.2f} s")
    print(f"Real time: {time.time() - start_time:.2f} s")
    print(f"Collision: {final_collision}")
    print(f"Lap count: {final_lap_count}")
    print(f"Speed RMSE: {speed_rmse:.3f} m/s")
    print(f"Speed MAE: {speed_mae:.3f} m/s")
    print(
        f"Maximum speed error: "
        f"{maximum_speed_error:.3f} m/s"
    )
    print(f"Telemetry samples: {len(telemetry_rows)}")
    print(f"Telemetry saved to: {telemetry_file}")

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        time_history,
        reference_history,
        linestyle="--",
        color="black",
        label="Reference speed",
    )

    ax.plot(
        time_history,
        measured_history,
        color="navy",
        label="Measured speed",
    )

    ax.plot(
        time_history,
        command_history,
        color="orange",
        alpha=0.8,
        label="PID speed command",
    )

    ax.set_title("Longitudinal PID Speed Tracking")
    ax.set_xlabel("Simulation time [s]")
    ax.set_ylabel("Speed [m/s]")
    ax.grid(True)
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        plot_file,
        dpi=200,
        bbox_inches="tight",
    )

    print(f"Speed plot saved to: {plot_file}")

    plt.show()


if __name__ == "__main__":
    main()