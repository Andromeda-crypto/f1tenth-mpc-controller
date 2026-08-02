"""Parameterized linear time-varying MPC quadratic program.

State order: z = [x, y, v, psi]
Input order: u = [acceleration, steering_angle]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Sequence

import cvxpy as cp
import numpy as np


NX = 4
NU = 2


def _diag(values: Sequence[float], size: int, name: str) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.shape != (size,) or np.any(values < 0.0):
        raise ValueError(f"{name} must contain {size} nonnegative diagonal weights")
    return np.diag(values)


@dataclass(frozen=True)
class MPCConfig:
    horizon: int = 8
    q: tuple[float, ...] = (4.0, 4.0, 1.0, 2.0)
    q_terminal: tuple[float, ...] = (8.0, 8.0, 2.0, 4.0)
    r: tuple[float, ...] = (0.1, 0.2)
    r_delta: tuple[float, ...] = (0.1, 1.0)
    min_speed: float = 0.0
    max_speed: float = 8.0
    min_accel: float = -8.0
    max_accel: float = 8.0
    max_steer: float = np.deg2rad(24.0)
    max_steer_rate: float = np.deg2rad(180.0)
    dt: float = 0.1
    solver_options: dict = field(
        default_factory=lambda: {
            "eps_abs": 1e-5,
            "eps_rel": 1e-5,
            "max_iter": 10_000,
            "polishing": True,
            "verbose": False,
        }
    )

    def __post_init__(self) -> None:
        if self.horizon < 1 or self.dt <= 0.0:
            raise ValueError("horizon and dt must be positive")
        if self.min_speed > self.max_speed or self.min_accel > self.max_accel:
            raise ValueError("lower bounds must not exceed upper bounds")
        if self.max_steer <= 0.0 or self.max_steer_rate <= 0.0:
            raise ValueError("steering limits must be positive")


@dataclass(frozen=True)
class MPCResult:
    states: np.ndarray
    controls: np.ndarray
    status: str
    objective: float
    wall_time_ms: float
    solver_time_ms: float | None

    @property
    def first_control(self) -> np.ndarray:
        return self.controls[:, 0].copy()


def linearize_discrete_kbm(
    z_bar: np.ndarray,
    u_bar: np.ndarray,
    *,
    dt: float,
    wheelbase: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Euler-discretize the affine KBM linearization at (z_bar, u_bar)."""
    z_bar = np.asarray(z_bar, dtype=float)
    u_bar = np.asarray(u_bar, dtype=float)
    if z_bar.shape != (NX,) or u_bar.shape != (NU,):
        raise ValueError("z_bar must have shape (4,) and u_bar shape (2,)")
    if dt <= 0.0 or wheelbase <= 0.0:
        raise ValueError("dt and wheelbase must be positive")

    _, _, speed, yaw = z_bar
    accel, steer = u_bar
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    cos_steer = np.cos(steer)
    if abs(cos_steer) < 1e-6:
        raise ValueError("steering operating point is too close to +/- pi/2")

    f_bar = np.array(
        [speed * cos_yaw, speed * sin_yaw, accel, speed * np.tan(steer) / wheelbase]
    )
    a_c = np.zeros((NX, NX))
    a_c[0, 2] = cos_yaw
    a_c[0, 3] = -speed * sin_yaw
    a_c[1, 2] = sin_yaw
    a_c[1, 3] = speed * cos_yaw
    a_c[3, 2] = np.tan(steer) / wheelbase

    b_c = np.zeros((NX, NU))
    b_c[2, 0] = 1.0
    b_c[3, 1] = speed / (wheelbase * cos_steer**2)

    a_d = np.eye(NX) + dt * a_c
    b_d = dt * b_c
    c_d = dt * (f_bar - a_c @ z_bar - b_c @ u_bar)
    return a_d, b_d, c_d


class LinearMPCQP:
    """Compile once, update Parameters, and repeatedly solve with OSQP."""

    def __init__(self, config: MPCConfig = MPCConfig()) -> None:
        self.config = config
        n = config.horizon
        self.z = cp.Variable((NX, n + 1), name="z")
        self.u = cp.Variable((NU, n), name="u")
        self.z0 = cp.Parameter(NX, name="z0")
        self.z_ref = cp.Parameter((NX, n + 1), name="z_ref")
        self.u_previous = cp.Parameter(NU, name="u_previous")
        self.a = [cp.Parameter((NX, NX), name=f"A_{k}") for k in range(n)]
        self.b = [cp.Parameter((NX, NU), name=f"B_{k}") for k in range(n)]
        self.c = [cp.Parameter(NX, name=f"C_{k}") for k in range(n)]

        q = _diag(config.q, NX, "q")
        qf = _diag(config.q_terminal, NX, "q_terminal")
        r = _diag(config.r, NU, "r")
        rd = _diag(config.r_delta, NU, "r_delta")

        objective = 0.0
        constraints: list[cp.Constraint] = [self.z[:, 0] == self.z0]
        for k in range(n):
            objective += cp.quad_form(self.z[:, k] - self.z_ref[:, k], q)
            objective += cp.quad_form(self.u[:, k], r)
            if k == 0:
                # Equivalent to ||u_0-u_previous||_Rd^2 after dropping the
                # parameter-only constant. This form keeps the problem DPP.
                delta_u = self.u[:, k] - self.u_previous
                objective += cp.quad_form(self.u[:, k], rd)
                objective += -2.0 * (rd @ self.u_previous) @ self.u[:, k]
            else:
                delta_u = self.u[:, k] - self.u[:, k - 1]
                objective += cp.quad_form(delta_u, rd)
            constraints += [
                self.z[:, k + 1] == self.a[k] @ self.z[:, k] + self.b[k] @ self.u[:, k] + self.c[k],
                self.u[0, k] >= config.min_accel,
                self.u[0, k] <= config.max_accel,
                cp.abs(self.u[1, k]) <= config.max_steer,
                cp.abs(delta_u[1]) <= config.max_steer_rate * config.dt,
            ]
        objective += cp.quad_form(self.z[:, n] - self.z_ref[:, n], qf)
        constraints += [
            self.z[2, :] >= config.min_speed,
            self.z[2, :] <= config.max_speed,
        ]
        self.problem = cp.Problem(cp.Minimize(objective), constraints)

    def solve(
        self,
        z0: np.ndarray,
        z_ref: np.ndarray,
        a_seq: np.ndarray,
        b_seq: np.ndarray,
        c_seq: np.ndarray,
        *,
        u_previous: np.ndarray | None = None,
        reuse_solver_cache: bool = True,
    ) -> MPCResult:
        n = self.config.horizon
        expected = {
            "z0": ((NX,), z0),
            "z_ref": ((NX, n + 1), z_ref),
            "a_seq": ((n, NX, NX), a_seq),
            "b_seq": ((n, NX, NU), b_seq),
            "c_seq": ((n, NX), c_seq),
        }
        arrays: dict[str, np.ndarray] = {}
        for name, (shape, value) in expected.items():
            array = np.asarray(value, dtype=float)
            if array.shape != shape or not np.all(np.isfinite(array)):
                raise ValueError(f"{name} must be finite with shape {shape}")
            arrays[name] = array
        previous = np.zeros(NU) if u_previous is None else np.asarray(u_previous, dtype=float)
        if previous.shape != (NU,) or not np.all(np.isfinite(previous)):
            raise ValueError("u_previous must be finite with shape (2,)")

        self.z0.value = arrays["z0"]
        self.z_ref.value = arrays["z_ref"]
        self.u_previous.value = previous
        for k in range(n):
            self.a[k].value = arrays["a_seq"][k]
            self.b[k].value = arrays["b_seq"][k]
            self.c[k].value = arrays["c_seq"][k]

        started = perf_counter()
        value = self.problem.solve(
            solver=cp.OSQP,
            # With time-varying A/B matrices, entries can move between zero and
            # nonzero. OSQP cannot warm-update a cached sparse matrix when that
            # pattern changes, so C4 sets this false and rebuilds the workspace.
            warm_start=reuse_solver_cache,
            # CVXPY 1.9 currently misclassifies this matrix-parameterized
            # dynamics problem during solve even though problem.is_dpp() is
            # true. C3's measured target includes canonicalization overhead.
            ignore_dpp=True,
            **self.config.solver_options,
        )
        wall_time_ms = (perf_counter() - started) * 1000.0
        if self.problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            raise RuntimeError(f"MPC solve failed with status {self.problem.status}")
        solver_time = getattr(self.problem.solver_stats, "solve_time", None)
        return MPCResult(
            states=np.asarray(self.z.value),
            controls=np.asarray(self.u.value),
            status=self.problem.status,
            objective=float(value),
            wall_time_ms=wall_time_ms,
            solver_time_ms=None if solver_time is None else solver_time * 1000.0,
        )
