"""D5: render a portfolio-ready comparison video from frozen D3 telemetry.

This script does NOT run F1TENTH Gym, solve MPC, or execute either controller.
It replays the authoritative D3 telemetry and interpolates it to video frame rate.
"""

from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import yaml

from experiments.c5_f1tenth_gym import find_default_config, load_track
from experiments.d3_multitrack import TRACKS


ROOT = Path(__file__).resolve().parents[1]
D3_DIR = ROOT / "results" / "d3"
OUTPUT_DIR = ROOT / "results" / "d5"

DEFAULT_FPS = 30
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720

MPC_COLOR = (50, 130, 255)
PP_COLOR = (255, 125, 45)
REFERENCE_COLOR = (95, 105, 120)
BACKGROUND = (13, 17, 23)
PANEL = (20, 25, 33, 225)
TEXT = (238, 242, 247)
MUTED = (155, 166, 180)


@dataclass
class ReplayData:
    time: np.ndarray
    x: np.ndarray
    y: np.ndarray
    speed: np.ndarray
    cte: np.ndarray
    steering: np.ndarray

    @property
    def duration(self) -> float:
        return float(self.time[-1])


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--track",
        choices=TRACKS,
        default="zandvoort",
    )
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    return parser.parse_args()


def read_replay(path: Path) -> ReplayData:
    if not path.exists():
        raise FileNotFoundError(f"Missing D3 telemetry: {path}")

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise RuntimeError(f"Empty telemetry: {path}")

    def values(name: str) -> np.ndarray:
        return np.asarray([float(row[name]) for row in rows], dtype=float)

    time = values("time_s")

    # D3 telemetry begins after the first control interval. Add t=0 using
    # the first recorded state so the video timeline matches lap time.
    if time[0] > 0.0:
        time = np.r_[0.0, time]

        def prepend(array: np.ndarray) -> np.ndarray:
            return np.r_[array[0], array]
    else:
        def prepend(array: np.ndarray) -> np.ndarray:
            return array

    return ReplayData(
        time=time,
        x=prepend(values("x_m")),
        y=prepend(values("y_m")),
        speed=prepend(values("speed_mps")),
        cte=prepend(values("cross_track_error_m")),
        steering=prepend(values("steer_command_rad")),
    )


def load_reference(track_name: str):
    config_path = find_default_config()

    with config_path.open(encoding="utf-8") as handle:
        config = copy.deepcopy(yaml.safe_load(handle))

    map_path, waypoint_file = TRACKS[track_name]

    config.update(
        map_path=f"../data/maps/{map_path}",
        map_ext=".png",
        wpt_path=f"../data/waypoints/{waypoint_file}",
        wpt_delim=";",
        wpt_rowskip=3,
    )

    return load_track(config, config_path, None)


def interpolate(data: ReplayData, t: float) -> dict[str, float]:
    t = float(np.clip(t, data.time[0], data.time[-1]))

    x = float(np.interp(t, data.time, data.x))
    y = float(np.interp(t, data.time, data.y))

    # Estimate visual heading from nearby trajectory points. This avoids
    # angle-wrap problems and requires no special yaw telemetry field.
    delta = 0.03
    t0 = max(data.time[0], t - delta)
    t1 = min(data.time[-1], t + delta)

    x0 = np.interp(t0, data.time, data.x)
    y0 = np.interp(t0, data.time, data.y)
    x1 = np.interp(t1, data.time, data.x)
    y1 = np.interp(t1, data.time, data.y)

    heading = float(np.arctan2(y1 - y0, x1 - x0))

    return {
        "x": x,
        "y": y,
        "heading": heading,
        "speed": float(np.interp(t, data.time, data.speed)),
        "cte": float(np.interp(t, data.time, data.cte)),
        "steering": float(np.interp(t, data.time, data.steering)),
    }


def get_font(size: int, bold: bool = False):
    candidates = (
        ["arialbd.ttf", "DejaVuSans-Bold.ttf"]
        if bold
        else ["arial.ttf", "DejaVuSans.ttf"]
    )

    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass

    return ImageFont.load_default()


class Scene:
    def __init__(
        self,
        reference_x: np.ndarray,
        reference_y: np.ndarray,
        width: int,
        height: int,
    ):
        self.width = width
        self.height = height

        # Reserve space at the bottom for telemetry panels.
        self.left = 55
        self.right = width - 55
        self.top = 70
        self.bottom = height - 190

        xmin = float(np.min(reference_x))
        xmax = float(np.max(reference_x))
        ymin = float(np.min(reference_y))
        ymax = float(np.max(reference_y))

        x_range = max(xmax - xmin, 1e-6)
        y_range = max(ymax - ymin, 1e-6)

        available_w = self.right - self.left
        available_h = self.bottom - self.top

        self.scale = min(
            available_w / x_range,
            available_h / y_range,
        ) * 0.94

        self.cx = 0.5 * (xmin + xmax)
        self.cy = 0.5 * (ymin + ymax)

        self.screen_cx = 0.5 * (self.left + self.right)
        self.screen_cy = 0.5 * (self.top + self.bottom)

        self.background = Image.new(
            "RGB",
            (width, height),
            BACKGROUND,
        )

        self._draw_static(reference_x, reference_y)

    def point(self, x: float, y: float) -> tuple[int, int]:
        px = self.screen_cx + (x - self.cx) * self.scale
        py = self.screen_cy - (y - self.cy) * self.scale
        return int(px), int(py)

    def _draw_static(
        self,
        reference_x: np.ndarray,
        reference_y: np.ndarray,
    ) -> None:
        draw = ImageDraw.Draw(self.background)

        title_font = get_font(28, bold=True)
        subtitle_font = get_font(15)

        draw.text(
            (55, 25),
            "MPC vs Pure Pursuit",
            font=title_font,
            fill=TEXT,
        )

        draw.text(
            (340, 34),
            "F1TENTH closed-loop comparison",
            font=subtitle_font,
            fill=MUTED,
        )

        points = [
            self.point(float(x), float(y))
            for x, y in zip(reference_x, reference_y)
        ]

        if len(points) >= 2:
            # Dark outer stroke makes the raceline visually distinct.
            draw.line(
                points + [points[0]],
                fill=(35, 41, 50),
                width=8,
                joint="curve",
            )
            draw.line(
                points + [points[0]],
                fill=REFERENCE_COLOR,
                width=2,
                joint="curve",
            )


def draw_vehicle(
    draw: ImageDraw.ImageDraw,
    scene: Scene,
    state: dict[str, float],
    color: tuple[int, int, int],
    controller: str,
) -> None:
    cx, cy = scene.point(state["x"], state["y"])
    angle = -state["heading"]

    forward = np.array([np.cos(angle), np.sin(angle)])
    side = np.array([-np.sin(angle), np.cos(angle)])
    center = np.array([cx, cy], dtype=float)

    if controller == "MPC":
        # Distinct arrow/diamond silhouette.
        nose = center + forward * 20
        left = center + side * 9
        tail = center - forward * 14
        right = center - side * 9

        polygon = [
            tuple(nose.astype(int)),
            tuple(left.astype(int)),
            tuple(tail.astype(int)),
            tuple(right.astype(int)),
        ]

    else:
        # Pure Pursuit retains the triangular vehicle marker.
        nose = center + forward * 17
        rear_left = center - forward * 10 + side * 10
        rear_right = center - forward * 10 - side * 10

        polygon = [
            tuple(nose.astype(int)),
            tuple(rear_left.astype(int)),
            tuple(rear_right.astype(int)),
        ]

    draw.polygon(
        polygon,
        fill=color,
        outline=(245, 245, 245),
    )

def draw_panel(
    draw: ImageDraw.ImageDraw,
    *,
    x: int,
    y: int,
    width: int,
    title: str,
    color: tuple[int, int, int],
    state: dict[str, float],
    finished: bool,
) -> None:
    draw.rounded_rectangle(
        (x, y, x + width, y + 122),
        radius=12,
        fill=PANEL,
        outline=color,
        width=2,
    )

    title_font = get_font(18, bold=True)
    value_font = get_font(15)

    draw.text(
        (x + 16, y + 12),
        title,
        font=title_font,
        fill=color,
    )

    status = "FINISHED" if finished else "RUNNING"

    draw.text(
        (x + width - 92, y + 15),
        status,
        font=get_font(12, bold=True),
        fill=MUTED if not finished else (120, 220, 150),
    )

    steering_deg = np.rad2deg(state["steering"])

    lines = [
        f"Speed       {state['speed']:5.2f} m/s",
        f"CTE         {abs(state['cte']):5.3f} m",
        f"Steering    {steering_deg:+6.2f} deg",
    ]

    for index, line in enumerate(lines):
        draw.text(
            (x + 16, y + 45 + index * 23),
            line,
            font=value_font,
            fill=TEXT,
        )


def render_video(
    *,
    track_name: str,
    mpc: ReplayData,
    pp: ReplayData,
    reference,
    fps: int,
    width: int,
    height: int,
) -> Path:
    output_dir = OUTPUT_DIR / track_name
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{track_name}_comparison.mp4"

    scene = Scene(
        np.asarray(reference.x),
        np.asarray(reference.y),
        width,
        height,
    )

    duration = max(mpc.duration, pp.duration)
    frame_count = int(np.ceil(duration * fps)) + 1

    mpc_trail: list[tuple[int, int]] = []
    pp_trail: list[tuple[int, int]] = []

    max_trail_frames = fps * 4

    print(
        f"Rendering {duration:.2f}s at {fps} FPS "
        f"({frame_count} frames)..."
    )

    writer = imageio.get_writer(
        output_path,
        fps=fps,
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

    try:
        for frame_index in range(frame_count):
            t = min(frame_index / fps, duration)

            mpc_state = interpolate(mpc, t)
            pp_state = interpolate(pp, t)

            mpc_trail.append(
                scene.point(mpc_state["x"], mpc_state["y"])
            )
            pp_trail.append(
                scene.point(pp_state["x"], pp_state["y"])
            )

            mpc_trail = mpc_trail[-max_trail_frames:]
            pp_trail = pp_trail[-max_trail_frames:]

            image = scene.background.copy()
            draw = ImageDraw.Draw(image, "RGBA")

            if len(mpc_trail) > 1:
                draw.line(
                    mpc_trail,
                    fill=(*MPC_COLOR, 175),
                    width=4,
                )

            if len(pp_trail) > 1:
                draw.line(
                    pp_trail,
                    fill=(*PP_COLOR, 175),
                    width=4,
                )

            draw_vehicle(draw,scene,mpc_state,MPC_COLOR,"MPC")
            draw_vehicle(draw,scene,pp_state,PP_COLOR,"Pure Pursuit")
            

            panel_y = height - 155
            panel_width = 355

            draw_panel(
                draw,
                x=55,
                y=panel_y,
                width=panel_width,
                title="MPC",
                color=MPC_COLOR,
                state=mpc_state,
                finished=t >= mpc.duration,
            )

            draw_panel(
                draw,
                x=width - 55 - panel_width,
                y=panel_y,
                width=panel_width,
                title="Pure Pursuit",
                color=PP_COLOR,
                state=pp_state,
                finished=t >= pp.duration,
            )

            clock = f"{t:05.2f} s"
            clock_font = get_font(24, bold=True)

            bbox = draw.textbbox((0, 0), clock, font=clock_font)
            clock_width = bbox[2] - bbox[0]

            draw.text(
                ((width - clock_width) // 2, height - 105),
                clock,
                font=clock_font,
                fill=TEXT,
            )

            writer.append_data(np.asarray(image))

            if (
                frame_index % (fps * 5) == 0
                or frame_index == frame_count - 1
            ):
                print(
                    f"  frame {frame_index + 1:4d}/{frame_count} "
                    f"({100 * (frame_index + 1) / frame_count:5.1f}%)"
                )

    finally:
        writer.close()

    return output_path


def main() -> None:
    args = arguments()

    if args.fps <= 0:
        raise ValueError("--fps must be positive")

    telemetry_dir = D3_DIR / args.track

    mpc = read_replay(
        telemetry_dir / "mpc_telemetry.csv"
    )
    pp = read_replay(
        telemetry_dir / "pure_pursuit_telemetry.csv"
    )

    reference = load_reference(args.track)

    print(f"Track:        {args.track}")
    print(f"MPC duration: {mpc.duration:.3f}s")
    print(f"PP duration:  {pp.duration:.3f}s")

    output = render_video(
        track_name=args.track,
        mpc=mpc,
        pp=pp,
        reference=reference,
        fps=args.fps,
        width=args.width,
        height=args.height,
    )

    print(f"\nD5 complete.")
    print(f"Video: {output}")


if __name__ == "__main__":
    main()