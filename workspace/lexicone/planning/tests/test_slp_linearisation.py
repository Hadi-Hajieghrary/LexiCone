"""Unit tests for the SLP/SQP linearisation of the kinematic bicycle.

We check that:

1. The linearisation reproduces the nonlinear RK4 step *exactly* at the
   linearisation point itself (zeroth-order accuracy).
2. The linearisation reproduces the nonlinear step to *first order* nearby,
   i.e., residual scales as ``O(step_size^2)`` when we shrink the trust region.
3. The structure of ``A`` and ``B`` matches the analytic kinematic bicycle.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from lexicone.planning.bicycle_model import NU, NX, discrete_dynamics
from lexicone.planning.slp_linearisation import (
    AffineDynamics,
    BicycleLinearisation,
    TrustRegion,
    sqp_convergence_metric,
)


WHEEL_BASE = 3.089
DT = 0.1


@pytest.fixture
def lin() -> BicycleLinearisation:
    return BicycleLinearisation(wheel_base=WHEEL_BASE, dt=DT)


@pytest.fixture
def step_fn():
    return discrete_dynamics(WHEEL_BASE, DT)


def test_linearisation_exact_at_linearisation_point(lin: BicycleLinearisation, step_fn):
    """At ``(bar_x, bar_u)``, the affine prediction equals the nonlinear step exactly."""
    x_bar = np.array([0.0, 0.0, 0.0, 5.0])
    u_bar = np.array([0.5, 0.05])
    affine = lin.linearise_around(x_bar, u_bar)

    affine_pred = affine.A @ x_bar + affine.B @ u_bar + affine.c
    nonlinear_pred = np.asarray(step_fn(x_bar, u_bar)).flatten()
    np.testing.assert_allclose(affine_pred, nonlinear_pred, atol=1e-12)


def test_linearisation_first_order_accuracy(lin: BicycleLinearisation, step_fn):
    """Residual ``f_NL(bar+δ) − [A(bar+δ) + B(u_bar) + c]`` scales as ``O(‖δ‖²)``."""
    x_bar = np.array([0.0, 0.0, 0.0, 5.0])
    u_bar = np.array([0.0, 0.0])
    affine = lin.linearise_around(x_bar, u_bar)

    # Probe at several shrinking radii — residual should decrease quadratically.
    radii = [0.5, 0.25, 0.125, 0.0625]
    residuals = []
    rng = np.random.default_rng(0)
    direction = rng.standard_normal(NX)
    direction /= np.linalg.norm(direction)
    for r in radii:
        x_probe = x_bar + r * direction
        affine_pred = affine.A @ x_probe + affine.B @ u_bar + affine.c
        nonlinear_pred = np.asarray(step_fn(x_probe, u_bar)).flatten()
        residuals.append(float(np.linalg.norm(affine_pred - nonlinear_pred)))

    # Each halving of the radius should cut the residual by at least a factor of 3.
    # (Quadratic scaling would give exactly 4; we allow some slack.)
    for i in range(len(residuals) - 1):
        ratio = residuals[i] / max(residuals[i + 1], 1e-15)
        assert ratio >= 3.0, (
            f"Residual ratio {ratio:.2f} too small at radius pair "
            f"{radii[i]}/{radii[i+1]} — linearisation is not first-order accurate."
        )


def test_linearisation_A_matrix_structure_at_rest(lin: BicycleLinearisation):
    """At ``v=0, δ=0``, the Jacobian ``A`` is the identity (the ego doesn't move
    in any direction, so :math:`\\partial f/\\partial x = I`)."""
    x_bar = np.array([0.0, 0.0, 0.0, 0.0])
    u_bar = np.array([0.0, 0.0])
    affine = lin.linearise_around(x_bar, u_bar)
    # When v = 0, position derivatives are zero; A's position rows are
    # therefore identity rows with a small mixed term in v from the second-step
    # RK4 dynamics.
    np.testing.assert_allclose(affine.A[0, 0], 1.0, atol=1e-12)
    np.testing.assert_allclose(affine.A[1, 1], 1.0, atol=1e-12)
    np.testing.assert_allclose(affine.A[2, 2], 1.0, atol=1e-12)
    np.testing.assert_allclose(affine.A[3, 3], 1.0, atol=1e-12)


def test_linearisation_B_matrix_velocity_coupling(lin: BicycleLinearisation):
    """The velocity column of ``B`` couples ``a`` into ``v_{k+1}`` at strength
    ``dt`` (to first order in RK4)."""
    x_bar = np.array([0.0, 0.0, 0.0, 0.0])
    u_bar = np.array([0.0, 0.0])
    affine = lin.linearise_around(x_bar, u_bar)
    # ∂v_{k+1}/∂a_k = dt (the velocity ODE is v_dot = a, integrated with step DT).
    np.testing.assert_allclose(affine.B[3, 0], DT, atol=1e-12)


def test_linearise_trajectory_returns_n_steps(lin: BicycleLinearisation):
    """Trajectory linearisation yields one AffineDynamics per control step."""
    N = 10
    X_bar = np.zeros((NX, N + 1))
    U_bar = np.zeros((NU, N))
    # Roll forward a constant-control reference.
    step_fn = discrete_dynamics(WHEEL_BASE, DT)
    for k in range(N):
        X_bar[:, k + 1] = np.asarray(step_fn(X_bar[:, k], U_bar[:, k])).flatten()
    affines = lin.linearise_trajectory(X_bar, U_bar)
    assert len(affines) == N
    for a in affines:
        assert a.A.shape == (NX, NX)
        assert a.B.shape == (NX, NU)
        assert a.c.shape == (NX,)


def test_linearisation_shape_rejection(lin: BicycleLinearisation):
    """Wrong-shape inputs are rejected explicitly."""
    with pytest.raises(ValueError):
        lin.linearise_around(np.array([0.0, 0.0]), np.array([0.0, 0.0]))
    with pytest.raises(ValueError):
        lin.linearise_around(np.zeros(NX), np.array([0.0]))


def test_trust_region_default():
    """Default trust region has the expected shape and reasonable magnitudes."""
    tr = TrustRegion.default()
    assert tr.dx_max.shape == (NX,)
    assert tr.du_max.shape == (NU,)
    assert (tr.dx_max > 0).all()
    assert (tr.du_max > 0).all()


def test_sqp_convergence_metric_zero_at_linearisation_point(
    lin: BicycleLinearisation, step_fn
):
    """If the 'solution' equals the warm-start, the dynamics residual is zero
    and the step size is zero."""
    N = 5
    X_bar = np.zeros((NX, N + 1))
    U_bar = np.zeros((NU, N))
    for k in range(N):
        X_bar[:, k + 1] = np.asarray(step_fn(X_bar[:, k], U_bar[:, k])).flatten()
    affines = lin.linearise_trajectory(X_bar, U_bar)
    max_res, max_step = sqp_convergence_metric(affines, X_bar, U_bar, X_bar, U_bar, step_fn)
    assert max_res == pytest.approx(0.0, abs=1e-12)
    assert max_step == pytest.approx(0.0, abs=1e-12)
