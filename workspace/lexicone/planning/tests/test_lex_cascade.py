"""Unit tests for the lexicographic cascade.

We exercise:

1. A trivial cascade with no active rule slots should produce ``V_i* = 0``
   for every level and converge on the reference-tracking trajectory.
2. A cascade with one boundary-binding safety rule should produce
   ``V_1* > 0`` (rule binds; can't be fully satisfied) and the lower-level
   stages should respect the budget.
3. The cascade returns a sensible active-set classification.

These tests run the *full* L+1 cascade so they are slower than other unit
tests in this package (multiple IPOPT solves). They run in seconds, not
minutes, but mark them appropriately.
"""

from __future__ import annotations

import numpy as np
import pytest

from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters

from lexicone.planning.bicycle_model import NU, NX, discrete_dynamics
from lexicone.planning.lcp_mpc import (
    LCPLevelSpec,
    LCPParameters,
    LCPRulePack,
    LinearisedRuleConstraint,
    make_inactive_constraint,
)
from lexicone.planning.lex_cascade import (
    CascadeResult,
    CascadeStageResult,
    run_cascade,
)
from lexicone.planning.slp_linearisation import BicycleLinearisation


WHEEL_BASE = 3.089
DT = 0.1
HORIZON_S = 1.0


def _base_params(penalty_form: str = "l1", slots=(1, 1, 1)) -> LCPParameters:
    specs = tuple(
        LCPLevelSpec(name=f"L{i+1}", slots_per_step=slots[i], epsilon_tolerance=1e-3)
        for i in range(len(slots))
    )
    return LCPParameters(
        horizon_s=HORIZON_S,
        dt_s=DT,
        penalty_form=penalty_form,
        level_specs=specs,
        weights_per_level=tuple(0.0 for _ in specs),  # cascade overrides
    )


def _identity_rule_pack(params: LCPParameters, N: int) -> LCPRulePack:
    levels = []
    for spec in params.level_specs:
        steps = [[make_inactive_constraint() for _ in range(spec.slots_per_step)] for _ in range(N)]
        levels.append(steps)
    return LCPRulePack(constraints_by_level=levels)


def _straight_warm_and_ref():
    """Build a straight-line warm-start trajectory and matching reference."""
    N = int(round(HORIZON_S / DT))
    step_fn = discrete_dynamics(WHEEL_BASE, DT)
    X_bar = np.zeros((NX, N + 1))
    U_bar = np.zeros((NU, N))
    X_bar[:, 0] = np.array([0.0, 0.0, 0.0, 5.0])
    for k in range(N):
        X_bar[:, k + 1] = np.asarray(step_fn(X_bar[:, k], U_bar[:, k])).flatten()
    xref = np.zeros((NX, N + 1))
    for k in range(N + 1):
        xref[0, k] = 5.0 * DT * k
        xref[3, k] = 5.0
    return X_bar, U_bar, xref


def test_cascade_no_rules_produces_zero_violations():
    """With no active rule constraints, every V_i* = 0."""
    params = _base_params(penalty_form="l1")
    lin = BicycleLinearisation(WHEEL_BASE, DT)
    X_bar, U_bar, xref = _straight_warm_and_ref()
    affine = lin.linearise_trajectory(X_bar, U_bar)
    N = X_bar.shape[1] - 1

    result = run_cascade(
        base_params=params,
        vehicle_parameters=get_pacifica_parameters(),
        affine_steps=affine,
        x0_local=X_bar[:, 0],
        u_prev=np.zeros(NU),
        v_cap=np.full(N + 1, 25.0),
        Xref_local=xref,
        rule_pack=_identity_rule_pack(params, N),
    )

    # Every V_i* is approximately zero.
    assert result.p_star[0] < 1e-3
    assert result.p_star[1] < 1e-3
    assert result.p_star[2] < 1e-3
    # J* is small (the trajectory tracks the reference).
    assert result.p_star[3] < 1.0
    # No active rule constraints.
    assert len(result.active_set) == 0


def test_cascade_records_one_stage_per_level_plus_performance():
    """A 3-level cascade should produce 4 CascadeStageResults."""
    params = _base_params(penalty_form="l1")
    lin = BicycleLinearisation(WHEEL_BASE, DT)
    X_bar, U_bar, xref = _straight_warm_and_ref()
    affine = lin.linearise_trajectory(X_bar, U_bar)
    N = X_bar.shape[1] - 1

    result = run_cascade(
        base_params=params,
        vehicle_parameters=get_pacifica_parameters(),
        affine_steps=affine,
        x0_local=X_bar[:, 0],
        u_prev=np.zeros(NU),
        v_cap=np.full(N + 1, 25.0),
        Xref_local=xref,
        rule_pack=_identity_rule_pack(params, N),
    )

    assert len(result.stage_results) == 4
    for i in range(3):
        assert result.stage_results[i].stage_index == i + 1
        assert not result.stage_results[i].is_performance_stage
    assert result.stage_results[3].is_performance_stage
    assert result.stage_results[3].stage_index == 4


def test_cascade_with_binding_safety_rule_records_nonzero_v_star():
    """A safety rule that the trajectory *cannot* fully satisfy should appear
    in ``V_1* > 0`` and the cascade should still terminate without crashing."""
    params = _base_params(penalty_form="l1")
    lin = BicycleLinearisation(WHEEL_BASE, DT)
    X_bar, U_bar, xref = _straight_warm_and_ref()
    affine = lin.linearise_trajectory(X_bar, U_bar)
    N = X_bar.shape[1] - 1

    # A safety rule that demands v <= -1 at every step. Impossible since v >= 0
    # is a hard box constraint; the slack will be at least 1 per step.
    rule_levels = []
    for i, spec in enumerate(params.level_specs):
        steps = []
        for k in range(N):
            if i == 0:
                steps.append(
                    [LinearisedRuleConstraint(a=np.array([0., 0., 0., 1.]), b=np.zeros(NU), e=1.0, mask=1.0)]
                )
            else:
                steps.append([make_inactive_constraint()])
        rule_levels.append(steps)
    rule_pack = LCPRulePack(constraints_by_level=rule_levels)

    result = run_cascade(
        base_params=params,
        vehicle_parameters=get_pacifica_parameters(),
        affine_steps=affine,
        x0_local=X_bar[:, 0],
        u_prev=np.zeros(NU),
        v_cap=np.full(N + 1, 25.0),
        Xref_local=xref,
        rule_pack=rule_pack,
    )

    # V_1* should be positive (the rule cannot be satisfied with v >= 0).
    assert result.p_star[0] > 1e-3
    # Lower levels and J should still be reasonable.
    assert result.p_star[1] < 1e-3 + result.p_star[0]
    # Active set should include the level-0 rule at most steps.
    active_level0 = [a for a in result.active_set if a[0] == 0]
    assert len(active_level0) > 0


def test_cascade_l2_runs_to_completion():
    """The cascade should run end-to-end under the L₂ penalty form."""
    params = _base_params(penalty_form="l2")
    lin = BicycleLinearisation(WHEEL_BASE, DT)
    X_bar, U_bar, xref = _straight_warm_and_ref()
    affine = lin.linearise_trajectory(X_bar, U_bar)
    N = X_bar.shape[1] - 1

    result = run_cascade(
        base_params=params,
        vehicle_parameters=get_pacifica_parameters(),
        affine_steps=affine,
        x0_local=X_bar[:, 0],
        u_prev=np.zeros(NU),
        v_cap=np.full(N + 1, 25.0),
        Xref_local=xref,
        rule_pack=_identity_rule_pack(params, N),
    )

    assert isinstance(result, CascadeResult)
    assert len(result.stage_results) == 4
    # Every V_i* approximately zero.
    np.testing.assert_array_less(result.p_star[:3], 1e-3)
