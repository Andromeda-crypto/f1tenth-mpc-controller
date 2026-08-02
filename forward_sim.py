import numpy as np
import matplotlib.pyplot as plt
from dynamics import step, WHEELBASE

def run_simulation(initial_state, controls, dt, steps):
    """
    Runs the simulation for a given number of steps.
    
    controls can be a single control [accel, steer] or a list of controls.
    """
    state = np.array(initial_state)
    history = [state]
    
    for i in range(steps):
        if isinstance(controls[0], (list, np.ndarray)):
            control = controls[i]
        else:
            control = controls
            
        state = step(state, control, dt)
        history.append(state)
        
    return np.array(history)

def main():
    # Simulation parameters
    dt = 0.1
    total_time = 10.0
    steps = int(total_time / dt)
    
        # Initial state: [x, y, theta, v]
        # Facing forward, starting at 1.0 m/s
    initial_state = [0.0, 0.0, 0.0, 1.0]
    
        # Constant control: [acceleration, steering_angle]
        # Constant acceleration test
    control = [0.7, 0.0] 
    
    print(f"Simulating constant acceleration {control[0]} m/s^2 from {initial_state[3]} m/s")
    history = run_simulation(initial_state, control, dt, steps) 
        # Extract path
    x = history[:, 0]
    y = history[:, 1]
    theta = history[:, 2]
    v = history[:, 3]
    
        # Plotting
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(x, y, label='Trajectory', marker='.')
    plt.axis('equal')
    plt.title("Forward Simulation: Constant Acceleration")
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    time_axis = np.linspace(0, total_time, steps + 1)
    plt.plot(time_axis, v, label='Velocity [m/s]')
    plt.xlabel("Time [s]")
    plt.ylabel("Velocity [m/s]")
    plt.legend()
    plt.grid(True)
    
        # Sanity check
    total_dist = np.sqrt(x[-1]**2 + y[-1]**2)
    expected_dist_cont = initial_state[3] * total_time + 0.5 * control[0] * total_time**2
    expected_dist_euler = initial_state[3] * total_time + 0.5 * control[0] * (total_time - dt) * total_time
    
    print(f"Final distance: {total_dist:.4f}m")
    print(f"Continuous-time expected distance: {expected_dist_cont:.4f}m")
    print(f"Euler-integration expected distance (dt={dt}): {expected_dist_euler:.4f}m")
    print(f"Final velocity: {v[-1]:.2f}m/s (Expected: {initial_state[3] + control[0]*total_time:.2f}m/s)")


    
    plt.show()

if __name__ == "__main__":
    main()
