from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


WAYPOINT_COLUMNS = ("s","x","y","psi","kappa","vx","ax")


def load_path(csv_path):
    """
    Load an F1TENTH waypoint CSV.

    Expected columns:
        s_m; x_m; y_m; psi_rad; kappa_radpm; vx_mps; ax_mps2

    Returns:
        Dictionary mapping simplified column names to NumPy arrays.
    """
    csv_path = Path(csv_path)

    if not csv_path.is_file():
        raise FileNotFoundError(f"Waypoint file not found: {csv_path}")

    data = np.loadtxt(
        csv_path,
        delimiter=";",
        comments="#",
    )

    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.shape[1] != len(WAYPOINT_COLUMNS):
        raise ValueError(
            f"Expected {len(WAYPOINT_COLUMNS)} columns, "
            f"but found {data.shape[1]} in {csv_path}."
        )

    return {
        name: data[:, index]
        for index, name in enumerate(WAYPOINT_COLUMNS)
    }


def plot_path(path):
    """Plot the loaded reference trajectory."""
    fig, ax = plt.subplots(figsize=(10, 8))

    ax.plot(
        path["x"],
        path["y"],
        color="navy",
        linewidth=2,
        label="Reference path",
    )

    # Show occasional heading vectors.
    spacing = max(1, len(path["x"]) // 25)

    ax.quiver(
        path["x"][::spacing],
        path["y"][::spacing],
        np.cos(path["psi"][::spacing]),
        np.sin(path["psi"][::spacing]),
        color="red",
        scale=25,
        width=0.003,
        label="Reference heading",
    )

    ax.scatter(
        path["x"][0],
        path["y"][0],
        color="green",
        s=70,
        zorder=3,
        label="Start",
    )

    ax.set_title("F1TENTH Reference Path")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)
    ax.legend()

    plt.show()


def main():
    waypoint_file = (
        Path(__file__).resolve().parent
        / "f1tenth_gym"
        / "examples"
        / "example_waypoints.csv"
    )

    path = load_path(waypoint_file)

    print(f"Loaded {len(path['x'])} waypoints")
    print(f"Path length: {path['s'][-1]:.2f} m")
    print(
        f"Speed range: {path['vx'].min():.2f}–"
        f"{path['vx'].max():.2f} m/s"
    )

    closure_error = np.hypot(
        path["x"][-1] - path["x"][0],
        path["y"][-1] - path["y"][0],
    )
    print(f"Start-to-end distance: {closure_error:.4f} m")

    plot_path(path)


if __name__ == "__main__":
    main()