# F1TENTH MPC Controller

A linear time-varying model predictive controller for autonomous path tracking in F1TENTH Gym, evaluated against a tuned Pure Pursuit baseline.

The project develops the controller in stages: kinematic bicycle model validation, affine linearization, constrained MPC formulation, nonlinear forward simulation, F1TENTH Gym integration, cost-weight tuning, and a controlled MPC-versus-Pure-Pursuit comparison.

## Controller

The MPC state and input are

```text
z = [x, y, v, psi]
u = [acceleration, steering angle]
```

At each 100 ms controller update, the nonlinear kinematic bicycle model is linearized along a nominal trajectory. A finite-horizon quadratic program then minimizes state-tracking error, terminal error, control effort, and input variation while enforcing speed, acceleration, steering-angle, and steering-rate limits. CVXPY formulates the problem and OSQP solves it. Only the first optimized input is applied before the horizon is rebuilt.

The final F1TENTH configuration uses an eight-step horizon and the following diagonal weights:

| Cost | Diagonal weights |
|---|---|
| `Q` | `(4.0, 4.0, 1.0, 4.0)` |
| `Q_terminal` | `(8.0, 8.0, 2.0, 8.0)` |
| `R` | `(0.1, 0.2)` |
| `R_delta` | `(0.1, 1.0)` |

These weights were selected from an eight-case sweep because they improved cross-track error, heading error, and steering variation over the initial configuration without changing lap time.

## Evaluation

Both controllers were evaluated with the same:

- F1TENTH Gym map and 156.36 m raceline
- 5.37-8.00 m/s reference-speed profile
- initial pose and vehicle geometry
- RK4 simulator with a 10 ms physics step
- 100 ms controller update period
- steering, steering-rate, acceleration, and speed limits
- lap-completion, collision, tracking, steering, and timing metrics

Pure Pursuit was not represented by an arbitrary lookahead. Six lookaheads from 0.8 m to 2.2 m were tested; four completed a collision-free lap. The 1.0 m case is used below because it was the fastest and had the lowest cross-track error among the valid Pure Pursuit runs.

| Metric | MPC: selected weights | Pure Pursuit: 1.0 m |
|---|---:|---:|
| Collision-free lap | Yes | Yes |
| Lap time | **23.610 s** | 24.070 s |
| Cross-track RMSE | 0.1439 m | **0.1241 m** |
| Heading RMSE | **4.732 deg** | 5.716 deg |
| Steering total variation | 6.0560 rad | **3.9778 rad** |
| Median controller update | 96.31 ms | **0.17 ms** |
| 95th-percentile update | 195.79 ms | **0.28 ms** |
| 100 ms deadline misses | 44.49% | **0.00%** |

The result is a trade-off, not a universal MPC victory. MPC completed the lap 0.46 s faster and tracked heading more accurately. Pure Pursuit achieved lower cross-track error, smoother steering, and dramatically lower computation time.

The MPC completed the lap without solver failures, but this CVXPY implementation is not consistently real-time at 10 Hz. Timing also varied between the tuning and comparison runs, so the measurements should be treated as machine- and workload-dependent. Direct OSQP matrix updates, solver-cache reuse for time-varying dynamics, and repeated-trial timing analysis are future optimization work.

## Repository layout

| Path | Purpose |
|---|---|
| `dynamics.py`, `forward_sim.py` | Initial kinematic-model and forward-simulation validation |
| `C2_KBM_Linearization.pdf` | Derivation of the affine discrete-time model |
| `mpc_qp.py` | Parameterized constrained MPC quadratic program |
| `test_mpc_qp.py` | Linearization, constraint, residual, and solver checks |
| `c4_forward_sim.py` | Closed-loop nonlinear simulation with an offline baseline |
| `c4_results/` | C4 telemetry, summaries, and plots |
| `path_loader.py` | Raceline loading and inspection |
| `pure_pursuit.py` | Pure Pursuit steering controller |
| `longitudinal_pid.py` | Longitudinal speed controller used in the earlier baseline bring-up |
| `c5_f1tenth_gym.py` | Final MPC lap in F1TENTH Gym with telemetry |
| `c6_tuning_sweep.py` | Eight-case MPC cost-weight sweep |
| `c6_controller_comparison.py` | Selected MPC versus six Pure Pursuit lookaheads |
| `c6_results/` | C6 comparison CSV files and plots |

## Setup

The final evaluation environment used Python 3.8.10. Create a local environment from the repository root and install the tested direct dependencies.

### Windows PowerShell

```powershell
py -3.8 -m venv .gym_env
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
& .\.gym_env\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### macOS or Linux

```bash
python3.8 -m venv .gym_env
source .gym_env/bin/activate
python -m pip install -r requirements.txt
```

The local `.gym_env/` directory is ignored by Git. The included map configuration, map image, and waypoint CSV allow the example-track evaluations to be run from the repository root.

## Reproduce the stages

Run the QP checks:

```bash
python test_mpc_qp.py
```

Run the standalone nonlinear forward simulation:

```bash
python c4_forward_sim.py
```

Run the final MPC lap in F1TENTH Gym without opening the renderer:

```bash
python c5_f1tenth_gym.py --no-render
```

Reproduce the MPC weight sweep:

```bash
python c6_tuning_sweep.py
```

Reproduce the fair controller comparison:

```bash
python c6_controller_comparison.py
```

The C6 scripts overwrite their summary CSV and plot outputs when rerun. The published table reports one deterministic lap per configuration; repeated trials under controlled CPU load are still needed for statistical timing claims.

## Scope and next work

The current controller uses a kinematic bicycle prediction model and successive linearization. The next technical step is not additional cost tuning: it is reducing and stabilizing optimization latency, followed by repeated trials and higher-speed or perturbed-condition testing. A later extension can compare the kinematic formulation with a dynamic bicycle model near the handling limits.
