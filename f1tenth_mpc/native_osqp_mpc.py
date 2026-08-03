"""Persistent native-OSQP backend for the C7 LTV MPC benchmark."""

from __future__ import annotations

from time import perf_counter

import numpy as np
import osqp
from scipy import sparse

from f1tenth_mpc.mpc_qp import MPCConfig, MPCResult, NU, NX


class NativeOSQPLinearMPCQP:
    """Fixed-sparsity QP whose numeric data are updated in place."""

    def __init__(self, config: MPCConfig = MPCConfig()) -> None:
        self.config = config
        self.n = config.horizon
        self.nz = NX * (self.n + 1)
        self.nv = self.nz + NU * self.n
        self._make_objective()
        self._make_constraints()

        options = dict(config.solver_options)
        self.solver = osqp.OSQP()
        self.solver.setup(P=self.P, q=np.zeros(self.nv), A=self.A, l=self.l, u=self.u, **options)

    def _zi(self, k: int, i: int) -> int:
        return NX * k + i

    def _ui(self, k: int, i: int) -> int:
        return self.nz + NU * k + i

    def _make_objective(self) -> None:
        q = np.asarray(self.config.q)
        qf = np.asarray(self.config.q_terminal)
        r = np.asarray(self.config.r)
        rd = np.asarray(self.config.r_delta)
        P = sparse.lil_matrix((self.nv, self.nv))
        for k in range(self.n + 1):
            weights = qf if k == self.n else q
            for i in range(NX):
                P[self._zi(k, i), self._zi(k, i)] = 2.0 * weights[i]
        for k in range(self.n):
            for i in range(NU):
                P[self._ui(k, i), self._ui(k, i)] += 2.0 * r[i]
                P[self._ui(k, i), self._ui(k, i)] += 2.0 * rd[i]
                if k > 0:
                    P[self._ui(k - 1, i), self._ui(k - 1, i)] += 2.0 * rd[i]
                    # OSQP accepts the upper triangular part of P.
                    P[self._ui(k - 1, i), self._ui(k, i)] += -2.0 * rd[i]
        self.P = sparse.triu(P.tocsc(), format="csc")

    def _make_constraints(self) -> None:
        # Rows: initial state, dynamics, speed, acceleration, steering, steer rate.
        r0 = 0
        self.initial_rows = np.arange(r0, r0 + NX); r0 += NX
        self.dynamics_rows = np.arange(r0, r0 + self.n * NX).reshape(self.n, NX); r0 += self.n * NX
        self.speed_rows = np.arange(r0, r0 + self.n + 1); r0 += self.n + 1
        self.accel_rows = np.arange(r0, r0 + self.n); r0 += self.n
        self.steer_rows = np.arange(r0, r0 + self.n); r0 += self.n
        self.rate_rows = np.arange(r0, r0 + self.n); r0 += self.n

        A = sparse.lil_matrix((r0, self.nv))
        for i, row in enumerate(self.initial_rows):
            A[row, self._zi(0, i)] = 1.0
        for k in range(self.n):
            for i, row in enumerate(self.dynamics_rows[k]):
                A[row, self._zi(k + 1, i)] = -1.0
                # Explicit placeholders guarantee a constant update pattern.
                for j in range(NX): A[row, self._zi(k, j)] = 1.0
                for j in range(NU): A[row, self._ui(k, j)] = 1.0
        for k, row in enumerate(self.speed_rows): A[row, self._zi(k, 2)] = 1.0
        for k, row in enumerate(self.accel_rows): A[row, self._ui(k, 0)] = 1.0
        for k, row in enumerate(self.steer_rows): A[row, self._ui(k, 1)] = 1.0
        for k, row in enumerate(self.rate_rows):
            A[row, self._ui(k, 1)] = 1.0
            if k > 0: A[row, self._ui(k - 1, 1)] = -1.0

        self.A = A.tocsc()
        self._position = {}
        for col in range(self.A.shape[1]):
            for p in range(self.A.indptr[col], self.A.indptr[col + 1]):
                self._position[(int(self.A.indices[p]), col)] = p

        self.l = np.empty(r0)
        self.u = np.empty(r0)
        self.l[self.speed_rows] = self.config.min_speed
        self.u[self.speed_rows] = self.config.max_speed
        self.l[self.accel_rows] = self.config.min_accel
        self.u[self.accel_rows] = self.config.max_accel
        self.l[self.steer_rows] = -self.config.max_steer
        self.u[self.steer_rows] = self.config.max_steer
        move = self.config.max_steer_rate * self.config.dt
        self.l[self.rate_rows] = -move
        self.u[self.rate_rows] = move
        self.l[self.initial_rows] = self.u[self.initial_rows] = 0.0
        self.l[self.dynamics_rows.ravel()] = self.u[self.dynamics_rows.ravel()] = 0.0

    def _validate(self, z0, z_ref, a_seq, b_seq, c_seq, u_previous):
        expected = ((NX,), (NX, self.n + 1), (self.n, NX, NX), (self.n, NX, NU), (self.n, NX))
        values = [np.asarray(v, dtype=float) for v in (z0, z_ref, a_seq, b_seq, c_seq)]
        if any(v.shape != shape or not np.all(np.isfinite(v)) for v, shape in zip(values, expected)):
            raise ValueError("native OSQP inputs have invalid shape or non-finite values")
        previous = np.zeros(NU) if u_previous is None else np.asarray(u_previous, dtype=float)
        if previous.shape != (NU,) or not np.all(np.isfinite(previous)):
            raise ValueError("u_previous must be finite with shape (2,)")
        return (*values, previous)

    def solve(self, z0, z_ref, a_seq, b_seq, c_seq, *, u_previous=None, reuse_solver_cache=True) -> MPCResult:
        z0, z_ref, a_seq, b_seq, c_seq, previous = self._validate(
            z0, z_ref, a_seq, b_seq, c_seq, u_previous
        )
        linear = np.zeros(self.nv)
        for k in range(self.n + 1):
            weights = np.asarray(self.config.q_terminal if k == self.n else self.config.q)
            linear[NX * k:NX * (k + 1)] = -2.0 * weights * z_ref[:, k]
        linear[self.nz:self.nz + NU] += -2.0 * np.asarray(self.config.r_delta) * previous

        Ax = self.A.data.copy()
        for k in range(self.n):
            for i, row in enumerate(self.dynamics_rows[k]):
                for j in range(NX): Ax[self._position[(int(row), self._zi(k, j))]] = a_seq[k, i, j]
                for j in range(NU): Ax[self._position[(int(row), self._ui(k, j))]] = b_seq[k, i, j]
        self.l[self.initial_rows] = self.u[self.initial_rows] = z0
        rhs = -c_seq.ravel()
        self.l[self.dynamics_rows.ravel()] = self.u[self.dynamics_rows.ravel()] = rhs
        move = self.config.max_steer_rate * self.config.dt
        self.l[self.rate_rows[0]] = previous[1] - move
        self.u[self.rate_rows[0]] = previous[1] + move

        started = perf_counter()
        self.solver.update(q=linear, Ax=Ax, l=self.l, u=self.u)
        result = self.solver.solve()
        wall_ms = (perf_counter() - started) * 1000.0
        status = result.info.status.lower().replace(" ", "_")
        if status not in ("solved", "solved_inaccurate"):
            raise RuntimeError(f"native OSQP solve failed with status {result.info.status}")
        vector = np.asarray(result.x)
        states = vector[:self.nz].reshape(self.n + 1, NX).T
        controls = vector[self.nz:].reshape(self.n, NU).T
        # Match LinearMPCQP's reported objective. Its k=0 rate term intentionally
        # drops the parameter-only previous-control constant.
        objective = 0.0
        q = np.asarray(self.config.q); qf = np.asarray(self.config.q_terminal)
        r = np.asarray(self.config.r); rd = np.asarray(self.config.r_delta)
        for k in range(self.n):
            objective += np.sum(q * (states[:, k] - z_ref[:, k]) ** 2)
            objective += np.sum(r * controls[:, k] ** 2)
            if k == 0:
                objective += np.sum(rd * controls[:, k] ** 2)
                objective += np.sum(-2.0 * rd * previous * controls[:, k])
            else:
                objective += np.sum(rd * (controls[:, k] - controls[:, k - 1]) ** 2)
        objective += np.sum(qf * (states[:, -1] - z_ref[:, -1]) ** 2)
        return MPCResult(states, controls, "optimal" if status == "solved" else "optimal_inaccurate",
                         float(objective), wall_ms, float(result.info.run_time) * 1000.0)
