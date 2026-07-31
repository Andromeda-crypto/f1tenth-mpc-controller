import numpy as np


def wrap_angle(angle):
    """Wrap an angle to [-pi, pi]."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


class PurePursuitController:
    def __init__(
        self,
        wheelbase,
        lookahead_distance,
        max_steering_angle,
    ):
        if wheelbase <= 0.0:
            raise ValueError("wheelbase must be positive")

        if lookahead_distance <= 0.0:
            raise ValueError("lookahead_distance must be positive")

        self.wheelbase = wheelbase
        self.lookahead_distance = lookahead_distance
        self.max_steering_angle = max_steering_angle

    def find_nearest_index(self, position, path):
        """Find the path point closest to the vehicle."""
        dx = path["x"] - position[0]
        dy = path["y"] - position[1]

        squared_distances = dx**2 + dy**2
        return int(np.argmin(squared_distances)) 

    def find_target_index(self, position, path, nearest_index):
        """
        Walk forward along the closed path until the target is at least
        one lookahead distance from the vehicle.
        """
        waypoint_count = len(path["x"])
        target_index = nearest_index

        for _ in range(waypoint_count):
            target_index = (target_index + 1) % waypoint_count

            dx = path["x"][target_index] - position[0]
            dy = path["y"][target_index] - position[1]
            distance = np.hypot(dx, dy)

            if distance >= self.lookahead_distance:
                return target_index

        raise RuntimeError("No valid lookahead target found")

    def compute_steering(self, state, path):
        """
        Compute a Pure Pursuit steering command.

        State:
            [x, y, heading, velocity]

        Returns:
            steering_angle, target_index, nearest_index
        """
        x, y, heading, _ = state
        position = np.array([x, y])

        nearest_index = self.find_nearest_index(position, path)

        target_index = self.find_target_index(
            position,
            path,
            nearest_index,
        )

        target_x = path["x"][target_index]
        target_y = path["y"][target_index]

        dx = target_x - x
        dy = target_y - y

        target_heading = np.arctan2(dy, dx)
        alpha = wrap_angle(target_heading - heading)
        actual_lookahead = np.hypot(dx, dy)

        steering_angle = np.arctan2(
            2.0 * self.wheelbase * np.sin(alpha),
            actual_lookahead,
        )

        steering_angle = np.clip(
            steering_angle,
            -self.max_steering_angle,
            self.max_steering_angle,
        )

        return steering_angle, target_index, nearest_index