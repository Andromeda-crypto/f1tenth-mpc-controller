"""D5: record presentation-ready controller videos from F1TENTH Gym."""

from __future__ import annotations

import argparse
import copy
import csv
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pyglet
import yaml

from experiments import d1_gym_comparison as d1
from experiments.c5_f1tenth_gym import find_default_config, load_track
from experiments.d3_multitrack import TRACKS
from f1tenth_mpc.mpc_qp import MPCConfig


ROOT = Path(__file__).resolve().parents[1]
D3_DIR = ROOT / "results" / "d3"
OUTPUT_DIR = ROOT / "results" / "d5"

PHYSICS_DT = 0.01
CONTROL_DT = 0.1
VIDEO_FPS = 30

CONTROLLER_OPTIONS = {
    "mpc": ("MPC",),
    "pure-pursuit": ("Pure Pursuit",),
    "both": ("MPC", "Pure Pursuit"),
}

CONTROLLER_SLUGS = {
    "MPC": "mpc",
    "Pure Pursuit": "pure_pursuit",
}

COLORS = {
    "MPC": (30, 100, 210),
    "Pure Pursuit": (230, 95, 20),
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record portfolio-ready F1TENTH controller videos."
    )
    parser.add_argument(
        "--track",
        choices=TRACKS,
        default="zandvoort",
    )
    parser.add_argument(
        "--controller",
        choices=CONTROLLER_OPTIONS,
        default="both",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=VIDEO_FPS,
    )
    parser.add_argument(
        "--max-time",
        type=float,
        default=120.0,
    )
    return parser.parse_args()


def load_telemetry(track: str, controller: str) -> list[dict[str, str]]:
    slug = CONTROLLER_SLUGS[controller]
    path = D3_DIR / track / f"{slug}_telemetry.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing frozen D3 telemetry: {path}"
        )

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise RuntimeError(f"Empty telemetry file: {path}")

    return rows


def telemetry_at_time(
    telemetry: list[dict[str, str]],
    simulated_time: float,
) -> dict[str, str]:
    index = int(simulated_time / CONTROL_DT)
    index = min(max(index, 0), len(telemetry) - 1)
    return telemetry[index]


def capture_framebuffer() -> np.ndarray:
    buffer = pyglet.image.get_buffer_manager().get_color_buffer()
    image_data = buffer.get_image_data()

    width = image_data.width
    height = image_data.height

    raw = image_data.get_data(
        "RGB",
        pitch=-width * 3,
    )

    return np.frombuffer(raw, dtype=np.uint8).reshape(
        height,
        width,
        3,
    )


def add_overlay(
    frame: np.ndarray,
    *,
    controller: str,
    track: str,
    simulated_time: float,
    telemetry_row: dict[str, str],
) -> np.ndarray:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.load_default()

    panel_x = 18
    panel_y = 18
    panel_width = 290
    panel_height = 132

    draw.rounded_rectangle(
        (
            panel_x,
            panel_y,
            panel_x + panel_width,
            panel_y + panel_height,
        ),
        radius=12,
        fill=(8, 12, 18, 210),
        outline=(*COLORS[controller], 255),
        width=3,
    )

    speed = float(telemetry_row["speed_mps"])
    cte = abs(float(telemetry_row["cross_track_error_m"]))
    steering = np.rad2deg(
        float(telemetry_row["steer_command_rad"])
    )

    title = f"{controller} | {track.title()}"
    lines = (
        title,
        f"Simulated time   {simulated_time:6.2f} s",
        f"Speed            {speed:6.2f} m/s",
        f"Cross-track err.  {cte:6.3f} m",
        f"Steering         {steering:6.2f} deg",
    )

    y = panel_y + 13

    for index, line in enumerate(lines):
        color = (
            (*COLORS[controller], 255)
            if index == 0
            else (245, 247, 250, 255)
        )
        draw.text(
            (panel_x + 15, y),
            line,
            font=font,
            fill=color,
        )
        y += 22

    return np.asarray(image)


class FrameRecorder:
    def __init__(
        self,
        *,
        output_path: Path,
        controller: str,
        track: str,
        telemetry: list[dict[str, str]],
        fps: int,
    ) -> None:
        self.output_path = output_path
        self.controller = controller
        self.track = track
        self.telemetry = telemetry
        self.fps = fps

        self.render_calls = 0
        self.frames_written = 0
        self.next_frame_time = 0.0
        self.writer = None

    def __enter__(self) -> "FrameRecorder":
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        self.writer = imageio.get_writer(
            self.output_path,
            fps=self.fps,
            codec="libx264",
            quality=8,
            macro_block_size=2,
            ffmpeg_params=[
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
            ],
        )
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self.writer is not None:
            self.writer.close()

    def render(self, env) -> None:
    # This function is called once before stepping, then once per physics step.
        simulated_time = max(
            0.0,
            (self.render_calls - 1) * PHYSICS_DT,
        )
        self.render_calls += 1

        # Do not render frames that the 30 FPS video will discard.
        if simulated_time + 1.0e-9 < self.next_frame_time:
            return

        env.render(mode="human_fast")

        frame = capture_framebuffer()
        row = telemetry_at_time(
            self.telemetry,
            simulated_time,
        )
        frame = add_overlay(
            frame,
            controller=self.controller,
            track=self.track,
            simulated_time=simulated_time,
            telemetry_row=row,
        )

        self.writer.append_data(frame)
        self.frames_written += 1
        self.next_frame_time += 1.0 / self.fps

        if self.frames_written % (self.fps * 5) == 0:
            print(
                f"recorded {self.frames_written} frames "
                f"({self.frames_written / self.fps:.1f}s video, "
                f"{simulated_time:.1f}s simulated)"
            )


def make_track_configuration(
    track_name: str,
) -> tuple[Path, dict, object]:
    config_path = find_default_config()

    with config_path.open(encoding="utf-8") as handle:
        base_config = yaml.safe_load(handle)

    map_path, waypoint_file = TRACKS[track_name]
    config = copy.deepcopy(base_config)

    config.update(
        map_path=f"../data/maps/{map_path}",
        map_ext=".png",
        wpt_path=f"../data/waypoints/{waypoint_file}",
        wpt_delim=";",
        wpt_rowskip=3,
    )

    track = load_track(config, config_path, None)

    config.update(
        sx=float(track.x[0]),
        sy=float(track.y[0]),
        stheta=float(track.yaw[0]),
    )

    return config_path, config, track


def record_controller(
    *,
    track_name: str,
    controller: str,
    fps: int,
    max_time: float,
) -> None:
    config_path, gym_config, track = make_track_configuration(
        track_name
    )
    telemetry = load_telemetry(track_name, controller)

    mpc_config = MPCConfig(
        horizon=8,
        dt=CONTROL_DT,
        max_speed=8.0,
        q=(4.0, 4.0, 1.0, 4.0),
        q_terminal=(8.0, 8.0, 2.0, 8.0),
        r=(0.1, 0.2),
        r_delta=(0.1, 1.0),
    )

    slug = CONTROLLER_SLUGS[controller]
    output_path = (
        OUTPUT_DIR
        / track_name
        / f"{track_name}_{slug}.mp4"
    )

    original_render = d1.render_environment

    try:
        with FrameRecorder(
            output_path=output_path,
            controller=controller,
            track=track_name,
            telemetry=telemetry,
            fps=fps,
        ) as recorder:
            d1.render_environment = recorder.render

            print(
                f"\nRecording {controller} on "
                f"{track_name.title()}..."
            )

            result = d1.run_controller(
                controller_name=controller,
                track=track,
                gym_config=gym_config,
                config_path=config_path,
                mpc_config=mpc_config,
                wheelbase=0.3302,
                lookahead=1.2,
                solver="native",
                physics_dt=PHYSICS_DT,
                control_dt=CONTROL_DT,
                max_time=max_time,
                render=True,
            )

            if not result.lap_completed or result.collision:
                raise RuntimeError(
                    f"Recording run failed: "
                    f"lap_completed={result.lap_completed}, "
                    f"collision={result.collision}"
                )

            print(
                f"wrote {output_path} "
                f"({recorder.frames_written} frames)"
            )
    finally:
        d1.render_environment = original_render


def main() -> None:
    args = parse_arguments()

    if args.fps <= 0:
        raise ValueError("--fps must be positive")

    controllers = CONTROLLER_OPTIONS[args.controller]

    for controller in controllers:
        record_controller(
            track_name=args.track,
            controller=controller,
            fps=args.fps,
            max_time=args.max_time,
        )

    print(
        f"\nD5 complete: recordings written under "
        f"{OUTPUT_DIR / args.track}"
    )


if __name__ == "__main__":
    main()