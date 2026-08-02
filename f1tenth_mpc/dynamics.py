import numpy as np


WHEELBASE = 0.33  # metres; 


def step(state, control, dt, wheelbase=WHEELBASE):
    """
    Advance the kinematic bicycle model by one Euler-integration step.

    State:
        [x, y, theta, velocity]

    Control:
        [acceleration, steering_angle]
    """
    x, y, theta, velocity = state
    acceleration, steering_angle = control

    x_dot = velocity * np.cos(theta)
    y_dot = velocity * np.sin(theta)
    theta_dot = (velocity / wheelbase) * np.tan(steering_angle)
    velocity_dot = acceleration

    next_state = np.array([
        x + x_dot * dt,
        y + y_dot * dt,
        theta + theta_dot * dt,
        velocity + velocity_dot * dt,
    ])

    return next_state