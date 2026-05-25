"""Sequential linearisation of the kinematic bicycle around a warm-start trajectory.

The reference framework — Lexicographic Constraint Programming (LCP) — requires
the per-tick OCP to be **convex** so that Theorem 4.1 (support-normal equivalence)
and Algorithms 1A/1B (the offline weight calibration) apply with their full
formal guarantees. Our dynamics are the nonlinear kinematic bicycle:

.. math::

    \\dot p_x = v \\cos\\psi, \\quad
    \\dot p_y = v \\sin\\psi, \\quad
    \\dot\\psi = (v / L) \\tan\\delta, \\quad
    \\dot v = a.

Because :math:`v \\cos\\psi`, :math:`v \\sin\\psi`, and :math:`(v/L)\\tan\\delta`
are nonlinear in the decision variables, a direct CasADi/IPOPT formulation
produces a *nonconvex* NLP. To recover convexity we use **Sequential Linear
Programming / Sequential Quadratic Programming (SLP/SQP)**: at every MPC tick
we linearise the discrete-time dynamics around a *warm-start trajectory*
:math:`(\\bar x_k, \\bar u_k)` and solve the resulting linear-dynamics QP. The
linearisation is iterated 1–3 times per tick until the trust region is
satisfied; the iterates are guaranteed to converge to a stationary point of the
original nonlinear problem under standard SQP regularity (Nocedal & Wright,
*Numerical Optimization*, §18).

Discrete RK4 dynamics
---------------------

The continuous bicycle ODE :math:`\\dot x = f_{\\text{ct}}(x, u)` is integrated
to discrete-time via classical 4th-order Runge–Kutta over step :math:`\\Delta t`,
producing :math:`x_{k+1} = f_{\\text{RK4}}(x_k, u_k, \\Delta t)`.

Linearisation
-------------

Around the warm-start trajectory :math:`(\\bar x_k, \\bar u_k)`, the discrete
dynamics admit the first-order Taylor expansion

.. math::

    x_{k+1} \\approx f_{\\text{RK4}}(\\bar x_k, \\bar u_k, \\Delta t)
                  + A_k \\, (x_k - \\bar x_k)
                  + B_k \\, (u_k - \\bar u_k),

where :math:`A_k := \\partial f_{\\text{RK4}}/\\partial x` and
:math:`B_k := \\partial f_{\\text{RK4}}/\\partial u`, both evaluated at
:math:`(\\bar x_k, \\bar u_k)`. Rearranging,

.. math::

    x_{k+1} = A_k x_k + B_k u_k + c_k,
    \\qquad c_k := f_{\\text{RK4}}(\\bar x_k, \\bar u_k, \\Delta t)
                  - A_k \\bar x_k - B_k \\bar u_k.

The triple :math:`(A_k, B_k, c_k)` is the standard *affine time-varying
linearisation* the convex MPC consumes. Because :math:`A_k`, :math:`B_k`, and
:math:`c_k` are *parameters* (not decision variables) in the OCP, the dynamics
constraint :math:`x_{k+1} = A_k x_k + B_k u_k + c_k` is linear in the
decision variables, preserving convexity.

This module provides :func:`linearise_around` which evaluates the analytic
Jacobians using CasADi's reverse-mode autodiff over a single RK4 step. The
expensive object is built once at construction and reused across ticks; per-tick
the caller supplies a fresh warm-start trajectory and receives a per-step list
of :class:`AffineDynamics` records.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import casadi as ca
import numpy as np

from .bicycle_model import NU, NX, continuous_dynamics, discrete_dynamics


@dataclass
class AffineDynamics:
    """Affine time-varying linearisation of one RK4 step.

    Represents :math:`x_{k+1} = A x_k + B u_k + c` where the dynamics constraint
    is *linear in the decision variables* (``x_k``, ``u_k``, ``x_{k+1}``).
    """

    A: np.ndarray  # shape (NX, NX)
    B: np.ndarray  # shape (NX, NU)
    c: np.ndarray  # shape (NX,)


class BicycleLinearisation:
    """Analytic linearisation of the kinematic bicycle's RK4 step.

    The CasADi expression for one RK4 step is built once at construction and
    differentiated symbolically to produce two reusable functions: ``A_fn`` and
    ``B_fn`` returning :math:`\\partial f/\\partial x` and :math:`\\partial f/\\partial u`
    at any operating point. The per-tick linearisation then just calls these.

    Parameters
    ----------
    wheel_base:
        Vehicle wheel base ``L`` (m).
    dt:
        Discretisation step :math:`\\Delta t` (s).
    """

    def __init__(self, wheel_base: float, dt: float) -> None:
        self._wheel_base = float(wheel_base)
        self._dt = float(dt)
        self._step_fn = discrete_dynamics(wheel_base, dt)

        # Build symbolic Jacobian functions over the RK4 step.
        x = ca.SX.sym("x", NX)
        u = ca.SX.sym("u", NU)
        x_next = self._step_fn(x, u)
        self._A_fn = ca.Function("A_fn", [x, u], [ca.jacobian(x_next, x)])
        self._B_fn = ca.Function("B_fn", [x, u], [ca.jacobian(x_next, u)])

    @property
    def dt(self) -> float:
        return self._dt

    @property
    def wheel_base(self) -> float:
        return self._wheel_base

    def linearise_around(
        self, x_bar: np.ndarray, u_bar: np.ndarray
    ) -> AffineDynamics:
        """Return :math:`(A, B, c)` such that
        :math:`x_{k+1} \\approx A x_k + B u_k + c`
        is a first-order accurate approximation of ``RK4(x_k, u_k)`` near
        :math:`(\\bar x, \\bar u)`."""
        x_arr = np.asarray(x_bar, dtype=np.float64).flatten()
        u_arr = np.asarray(u_bar, dtype=np.float64).flatten()
        if x_arr.shape != (NX,):
            raise ValueError(f"x_bar must have shape ({NX},), got {x_arr.shape}")
        if u_arr.shape != (NU,):
            raise ValueError(f"u_bar must have shape ({NU},), got {u_arr.shape}")
        A = np.asarray(self._A_fn(x_arr, u_arr)).reshape(NX, NX).astype(np.float64)
        B = np.asarray(self._B_fn(x_arr, u_arr)).reshape(NX, NU).astype(np.float64)
        x_next_bar = np.asarray(self._step_fn(x_arr, u_arr)).flatten().astype(np.float64)
        c = x_next_bar - A @ x_arr - B @ u_arr
        return AffineDynamics(A=A, B=B, c=c)

    def linearise_trajectory(
        self, X_bar: np.ndarray, U_bar: np.ndarray
    ) -> List[AffineDynamics]:
        """Linearise every step of a warm-start trajectory.

        Parameters
        ----------
        X_bar:
            Warm-start state trajectory, shape ``(NX, N+1)``.
        U_bar:
            Warm-start control trajectory, shape ``(NU, N)``.

        Returns
        -------
        A list of length ``N`` containing one :class:`AffineDynamics` per step.
        """
        if X_bar.ndim != 2 or X_bar.shape[0] != NX:
            raise ValueError(f"X_bar must have shape (NX={NX}, N+1), got {X_bar.shape}")
        if U_bar.ndim != 2 or U_bar.shape[0] != NU:
            raise ValueError(f"U_bar must have shape (NU={NU}, N), got {U_bar.shape}")
        if X_bar.shape[1] != U_bar.shape[1] + 1:
            raise ValueError(
                f"X_bar has {X_bar.shape[1]} columns but U_bar has {U_bar.shape[1]} — "
                "expected X_bar.shape[1] == U_bar.shape[1] + 1"
            )
        N = U_bar.shape[1]
        return [self.linearise_around(X_bar[:, k], U_bar[:, k]) for k in range(N)]


# ----------------------------------------------------------------------
# Trust-region / step-size machinery
# ----------------------------------------------------------------------


@dataclass
class TrustRegion:
    """Box trust-region around the warm-start trajectory.

    The convex MPC's decision variables are constrained to lie within
    ``Δx_max``, ``Δu_max`` of the linearisation point. This prevents the
    SLP/SQP iterates from straying outside the region where the linearisation
    is accurate. The standard SQP convergence proofs (Nocedal & Wright §18.7)
    require the trust region to be tightened on rejected steps and loosened on
    accepted ones; we provide the data structure here, the policy lives in the
    iterating routine (``lcp_mpc.solve_sqp_iterations``).
    """

    dx_max: np.ndarray   # (NX,) per-state trust-region half-width
    du_max: np.ndarray   # (NU,) per-control trust-region half-width

    @classmethod
    def default(cls) -> "TrustRegion":
        # Generous defaults: 5 m position, 0.3 rad heading, 5 m/s velocity,
        # 2 m/s^2 accel, 0.2 rad steering.
        return cls(
            dx_max=np.array([5.0, 5.0, 0.3, 5.0], dtype=np.float64),
            du_max=np.array([2.0, 0.2], dtype=np.float64),
        )


def sqp_convergence_metric(
    affine_steps: List[AffineDynamics],
    X_sol: np.ndarray,
    U_sol: np.ndarray,
    X_bar: np.ndarray,
    U_bar: np.ndarray,
    nonlinear_step_fn,
) -> Tuple[float, float]:
    """Diagnostic metric for SQP convergence.

    For each step k, computes the residual between the linearised dynamics
    ``A_k x_k + B_k u_k + c_k`` (which the convex MPC enforced as an equality)
    and the true nonlinear RK4 ``f_NL(x_k, u_k)``. If the SQP iterates have
    converged, this residual is below the trust-region tolerance.

    Returns
    -------
    (max_dynamics_residual, max_step_size) where:
      - ``max_dynamics_residual`` is the largest ‖A_k x_k + B_k u_k + c_k − f_NL(x_k, u_k)‖₂
        across all steps;
      - ``max_step_size`` is the largest ‖(x_k − bar x_k, u_k − bar u_k)‖_∞
        across all steps — i.e., how far the new solution moved from the
        linearisation point.

    Both quantities decrease toward zero as the SQP iteration converges. The
    caller can use either to decide whether to do another linearise-then-solve
    pass.
    """
    N = U_sol.shape[1]
    max_res = 0.0
    max_step = 0.0
    for k in range(N):
        # Linearised prediction of x_{k+1}.
        affine_pred = affine_steps[k].A @ X_sol[:, k] + affine_steps[k].B @ U_sol[:, k] + affine_steps[k].c
        # Nonlinear truth.
        nonlinear_pred = np.asarray(nonlinear_step_fn(X_sol[:, k], U_sol[:, k])).flatten()
        res = float(np.linalg.norm(affine_pred - nonlinear_pred))
        max_res = max(max_res, res)
        step = float(
            max(
                np.max(np.abs(X_sol[:, k] - X_bar[:, k])),
                np.max(np.abs(U_sol[:, k] - U_bar[:, k])),
            )
        )
        max_step = max(max_step, step)
    # Last state residual (no control after step N-1).
    return max_res, max_step
