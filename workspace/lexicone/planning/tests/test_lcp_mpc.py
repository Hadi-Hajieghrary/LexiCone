"""Unit tests for the convex LCP MPC.

We do not exercise the lex cascade or weight calibration here — those are
tested separately. The goal is to confirm:

1. The MPC builds with various level configurations.
2. With all rule slots inactive (mask=0), the MPC behaves like the legacy
   reference-tracking MPC.
3. Activating one rule slot with a soft speed-limit constraint correctly
   biases the trajectory.
4. Both L₁ and L₂ penalty forms produce trajectories.
5. The Tikhonov stabiliser keeps the L₁ problem well-conditioned.
6. Pickling round-trip works (mirrors the legacy planner's invariant).
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters

from lexicone.planning.bicycle_model import NU, NX, discrete_dynamics
from lexicone.planning.lcp_mpc import (
    LCPLevelSpec,
    LCPLimits,
    LCPParameters,
    LCPRulePack,
    LCPTrajectoryPlanner,
    LinearisedRuleConstraint,
    make_inactive_constraint,
)
from lexicone.planning.slp_linearisation import BicycleLinearisation


WHEEL_BASE = 3.089
DT = 0.1
HORIZON_S = 1.0


def _build_planner(penalty_form: str = "l1", slots_per_level=(2, 2, 2)):
    """Construct a 3-level LCP planner with the requested penalty form."""
    level_specs = tuple(
        LCPLevelSpec(name=f"L{i+1}", slots_per_step=slots_per_level[i], epsilon_tolerance=1e-3)
        for i in range(len(slots_per_level))
    )
    params = LCPParameters(
        horizon_s=HORIZON_S,
        dt_s=DT,
        penalty_form=penalty_form,
        level_specs=level_specs,
        weights_per_level=(100.0, 10.0, 1.0),
    )
    return LCPTrajectoryPlanner(get_pacifica_parameters(), params)


def _identity_rule_pack(planner: LCPTrajectoryPlanner) -> LCPRulePack:
    """All slots inactive — the OCP reduces to pure tracking."""
    N = planner.horizon_steps
    levels = []
    for spec in planner._params.level_specs:
        steps = [[make_inactive_constraint() for _ in range(spec.slots_per_step)] for _ in range(N)]
        levels.append(steps)
    return LCPRulePack(constraints_by_level=levels)


def _straight_warm_start(planner: LCPTrajectoryPlanner, v0: float = 5.0):
    """Straight-line warm start: constant velocity, no turning, no acceleration."""
    N = planner.horizon_steps
    X_bar = np.zeros((NX, N + 1))
    U_bar = np.zeros((NU, N))
    step_fn = discrete_dynamics(WHEEL_BASE, DT)
    X_bar[:, 0] = np.array([0.0, 0.0, 0.0, v0])
    for k in range(N):
        X_bar[:, k + 1] = np.asarray(step_fn(X_bar[:, k], U_bar[:, k])).flatten()
    return X_bar, U_bar


def _straight_xref(N: int, v_target: float = 5.0) -> np.ndarray:
    """Reference along +x at constant velocity v_target."""
    xref = np.zeros((NX, N + 1))
    for k in range(N + 1):
        xref[0, k] = v_target * DT * k
        xref[3, k] = v_target
    return xref


def test_lcp_planner_builds_with_three_levels():
    planner = _build_planner()
    assert planner.n_levels == 3
    assert planner.horizon_steps == int(round(HORIZON_S / DT))


def test_lcp_planner_pure_tracking_no_rules(monkeypatch):
    """With all rules inactive, the planner should reproduce the warm-start
    (which already matches the reference)."""
    planner = _build_planner(penalty_form="l1")
    N = planner.horizon_steps

    X_bar, U_bar = _straight_warm_start(planner, v0=5.0)
    affine = planner.linearisation().linearise_trajectory(X_bar, U_bar)
    xref = _straight_xref(N, v_target=5.0)
    v_cap = np.full(N + 1, 25.0)

    planner.push_parameters(
        affine_steps=affine,
        x0_local=X_bar[:, 0],
        u_prev=np.zeros(NU),
        v_cap=v_cap,
        Xref_local=xref,
        rule_pack=_identity_rule_pack(planner),
    )
    planner.warm_start(X_bar, U_bar)
    X_sol, U_sol, T_sol = planner.solve_once()

    # Position tracking error should be tiny.
    assert np.max(np.abs(X_sol[:2, :] - xref[:2, :])) < 0.1
    # Velocity tracking error should be tiny.
    assert np.max(np.abs(X_sol[3, :] - xref[3, :])) < 0.2
    # All slacks should be ~0.
    for T_i in T_sol:
        assert np.max(T_i) < 1e-3


def test_lcp_planner_active_speed_rule_biases_trajectory():
    """An active "v <= v_target" rule at level 2 with mask=1 should clamp the
    speed: the L₁ cost on slack will push the MPC to track at exactly the cap."""
    planner = _build_planner(penalty_form="l1", slots_per_level=(1, 1, 1))
    N = planner.horizon_steps

    # Warm start at v0=5.0, but reference wants v=10.0.
    X_bar, U_bar = _straight_warm_start(planner, v0=5.0)
    affine = planner.linearisation().linearise_trajectory(X_bar, U_bar)
    xref = _straight_xref(N, v_target=10.0)
    v_cap = np.full(N + 1, 25.0)

    # Build a rule pack where level 2 slot 0 enforces v_k <= 6.0 (a 4 m/s gap
    # below the reference). a = [0, 0, 0, 1], b = [0, 0], e = -6.0 → v - 6 <= t.
    rule_pack_levels = []
    for i, spec in enumerate(planner._params.level_specs):
        steps = []
        for k in range(N):
            slots = []
            for j in range(spec.slots_per_step):
                if i == 1 and j == 0:
                    slots.append(
                        LinearisedRuleConstraint(
                            a=np.array([0.0, 0.0, 0.0, 1.0]),
                            b=np.zeros(NU),
                            e=-6.0,
                            mask=1.0,
                        )
                    )
                else:
                    slots.append(make_inactive_constraint())
            steps.append(slots)
        rule_pack_levels.append(steps)
    rule_pack = LCPRulePack(constraints_by_level=rule_pack_levels)

    planner.push_parameters(
        affine_steps=affine,
        x0_local=X_bar[:, 0],
        u_prev=np.zeros(NU),
        v_cap=v_cap,
        Xref_local=xref,
        rule_pack=rule_pack,
    )
    planner.warm_start(X_bar, U_bar)
    X_sol, U_sol, T_sol = planner.solve_once()

    # The active speed rule has weight w_2 = 10. The reference-speed cost has
    # weight w_speed = 1 against (v - 10)^2. The optimal v balances: every unit
    # of v above 6 pays w_2*1=10 in L1 cost; the reference saving is w_speed *
    # 2*(v-10). The minimum is at v ≈ 6 (rule binds) once w_2 > w_speed*|grad|.
    # With v_target = 10 and w_2 = 10 vs w_speed = 1, the rule should bind.
    terminal_v = X_sol[3, -1]
    assert terminal_v < 7.5, f"speed rule should bind; got v_terminal={terminal_v:.2f}"
    # The level-2 slack should be non-zero (the rule is active and forcing
    # the constraint). CasADi returns a 1-D array when slots_per_step == 1.
    assert np.max(np.asarray(T_sol[1]).flatten()) > 0.0


def test_lcp_planner_l1_and_l2_both_solve():
    """Both penalty forms should produce valid trajectories on the same problem."""
    for penalty in ("l1", "l2"):
        planner = _build_planner(penalty_form=penalty)
        N = planner.horizon_steps
        X_bar, U_bar = _straight_warm_start(planner, v0=5.0)
        affine = planner.linearisation().linearise_trajectory(X_bar, U_bar)
        xref = _straight_xref(N, v_target=5.0)
        v_cap = np.full(N + 1, 25.0)

        planner.push_parameters(
            affine_steps=affine,
            x0_local=X_bar[:, 0],
            u_prev=np.zeros(NU),
            v_cap=v_cap,
            Xref_local=xref,
            rule_pack=_identity_rule_pack(planner),
        )
        planner.warm_start(X_bar, U_bar)
        X_sol, U_sol, T_sol = planner.solve_once()
        assert X_sol.shape == (NX, N + 1)
        assert U_sol.shape == (NU, N)


def test_lcp_planner_pickle_roundtrip():
    """Pickling drops CasADi state; unpickling rebuilds via _build_problem."""
    planner = _build_planner()
    blob = pickle.dumps(planner)
    restored = pickle.loads(blob)
    assert restored.n_levels == planner.n_levels
    assert restored.horizon_steps == planner.horizon_steps

    # The restored planner should be functional.
    N = restored.horizon_steps
    X_bar, U_bar = _straight_warm_start(restored, v0=5.0)
    affine = restored.linearisation().linearise_trajectory(X_bar, U_bar)
    xref = _straight_xref(N, v_target=5.0)
    v_cap = np.full(N + 1, 25.0)
    restored.push_parameters(
        affine_steps=affine,
        x0_local=X_bar[:, 0],
        u_prev=np.zeros(NU),
        v_cap=v_cap,
        Xref_local=xref,
        rule_pack=_identity_rule_pack(restored),
    )
    restored.warm_start(X_bar, U_bar)
    X_sol, U_sol, T_sol = restored.solve_once()
    assert X_sol.shape == (NX, N + 1)


def test_per_level_violation_computation():
    """V_i should be sum-of-slacks (L₁) or sum-of-squared-slacks (L₂)."""
    planner_l1 = _build_planner(penalty_form="l1", slots_per_level=(1, 1, 1))
    planner_l2 = _build_planner(penalty_form="l2", slots_per_level=(1, 1, 1))
    T_sol = [
        np.array([[1.0, 2.0, 3.0]]),     # level 1: 1 slot × 3 steps
        np.array([[0.5, 0.0, 0.5]]),
        np.array([[0.0, 0.0, 0.0]]),
    ]
    V_l1 = planner_l1.per_level_violation(T_sol)
    V_l2 = planner_l2.per_level_violation(T_sol)
    np.testing.assert_allclose(V_l1, [6.0, 1.0, 0.0])
    np.testing.assert_allclose(V_l2, [14.0, 0.5, 0.0])  # 1 + 4 + 9; 0.25 + 0 + 0.25
