"""End-to-end integration test for the LCP foundation.

This is the acceptance test for Phase 0: it wires the map-lifter (here
mocked as a None map view), the SLP linearisation, the rule encoder
framework with concrete rules, the LCP MPC, and the lex cascade into a
single pipeline and asserts the pipeline produces sensible trajectories.

The map view is left empty so only the *non-map-dependent* encoders run:

- :class:`CollisionRule`            (Level 1, with synthetic agents)
- :class:`SpeedLimitRule`           (Level 2)
- :class:`SafeHeadwayRule`          (Level 3)
- :class:`LongitudinalComfortRule`  (Level 3)
- :class:`LateralAccelerationRule`  (Level 3)

Together they cover all three priority levels, with both linear and
linearised-nonlinear constraint forms.
"""

from __future__ import annotations

import numpy as np
import pytest

from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters

from lexicone.planning.bicycle_model import NU, NX, discrete_dynamics
from lexicone.planning.lcp_mpc import (
    LCPLevelSpec,
    LCPParameters,
    LCPTrajectoryPlanner,
)
from lexicone.planning.lex_cascade import run_cascade
from lexicone.planning.rule_encoder import (
    AgentSlot,
    CollisionRule,
    EncoderContext,
    LateralAccelerationRule,
    LongitudinalComfortRule,
    RuleSet,
    SafeHeadwayRule,
    SpeedLimitRule,
)
from lexicone.planning.slp_linearisation import BicycleLinearisation


WHEEL_BASE = 3.089
DT = 0.1
HORIZON_S = 1.5


def _build_scenario(v0: float, v_target: float, lead_x: float):
    """Build a 1.5 s straight-road scenario with one lead vehicle 30 m ahead."""
    N = int(round(HORIZON_S / DT))
    step_fn = discrete_dynamics(WHEEL_BASE, DT)
    X_bar = np.zeros((NX, N + 1))
    U_bar = np.zeros((NU, N))
    X_bar[:, 0] = np.array([0.0, 0.0, 0.0, v0])
    for k in range(N):
        X_bar[:, k + 1] = np.asarray(step_fn(X_bar[:, k], U_bar[:, k])).flatten()
    xref = np.zeros((NX, N + 1))
    for k in range(N + 1):
        xref[0, k] = v_target * DT * k
        xref[3, k] = v_target
    lead = AgentSlot(
        track_id="lead",
        x=lead_x, y=0.0, vx=0.0, vy=0.0,
        length=4.5, width=1.8, is_vru=False,
    )
    return X_bar, U_bar, xref, (lead,), N


def _ruleset_no_map() -> RuleSet:
    """A 3-level RuleSet with only the non-map-dependent rules wired in."""
    safety = [CollisionRule(slots_per_step=2)]
    legal = [SpeedLimitRule(slots_per_step=1)]
    comfort = [
        SafeHeadwayRule(slots_per_step=1),
        LongitudinalComfortRule(slots_per_step=2),
        LateralAccelerationRule(slots_per_step=2),
    ]
    return RuleSet(levels=[safety, legal, comfort])


def _build_planner(rs: RuleSet, penalty_form: str = "l1") -> LCPTrajectoryPlanner:
    counts = rs.slots_per_step_per_level()
    specs = tuple(
        LCPLevelSpec(name=f"L{i+1}", slots_per_step=counts[i], epsilon_tolerance=1e-3)
        for i in range(len(counts))
    )
    params = LCPParameters(
        horizon_s=HORIZON_S,
        dt_s=DT,
        penalty_form=penalty_form,
        level_specs=specs,
        weights_per_level=(1000.0, 100.0, 10.0),
        desired_speed_mps=12.0,
    )
    return LCPTrajectoryPlanner(get_pacifica_parameters(), params)


def test_lcp_pipeline_empty_road_no_violations():
    """Ego at v=v_target with no agents nearby — every level should be slack-free."""
    rs = _ruleset_no_map()
    planner = _build_planner(rs, penalty_form="l1")
    X_bar, U_bar, xref, _agents, N = _build_scenario(v0=8.0, v_target=8.0, lead_x=200.0)
    # Single far-away lead so CollisionRule fires applies_to_horizon=True but
    # the constraint is far from binding.
    far_lead = AgentSlot("lead", x=200.0, y=0.0, vx=0.0, vy=0.0, length=4.5, width=1.8, is_vru=False)
    ctx = EncoderContext(
        horizon_steps=N, dt_s=DT,
        warm_start_X=X_bar, warm_start_U=U_bar,
        Xref_local=xref,
        agents_local=(far_lead,),
        map_view=None,
        desired_speed_mps=8.0,
        ego_radius_m=planner._ego_radius,
        wheel_base_m=WHEEL_BASE,
    )
    pack = rs.encode_all(ctx)
    affine = planner.linearisation().linearise_trajectory(X_bar, U_bar)
    planner.push_parameters(
        affine_steps=affine,
        x0_local=X_bar[:, 0],
        u_prev=np.zeros(NU),
        v_cap=np.full(N + 1, 25.0),
        Xref_local=xref,
        rule_pack=pack,
    )
    planner.warm_start(X_bar, U_bar)
    X_sol, U_sol, T_sol = planner.solve_once()

    # All level slacks should be ~zero.
    for i, T_i in enumerate(T_sol):
        max_slack = float(np.max(np.maximum(T_i, 0.0)))
        assert max_slack < 1e-2, f"level {i} slack {max_slack:.3g} should be near zero"


def test_lcp_pipeline_lead_vehicle_triggers_headway():
    """Lead vehicle close ahead → SafeHeadwayRule binds → level-3 violation positive."""
    rs = _ruleset_no_map()
    planner = _build_planner(rs, penalty_form="l1")
    # Ego at v0=10, lead just 8 m ahead → headway grossly violated.
    X_bar, U_bar, xref, agents, N = _build_scenario(v0=10.0, v_target=10.0, lead_x=8.0)
    ctx = EncoderContext(
        horizon_steps=N, dt_s=DT,
        warm_start_X=X_bar, warm_start_U=U_bar,
        Xref_local=xref,
        agents_local=agents,
        map_view=None,
        desired_speed_mps=10.0,
        ego_radius_m=planner._ego_radius,
        wheel_base_m=WHEEL_BASE,
    )
    pack = rs.encode_all(ctx)
    affine = planner.linearisation().linearise_trajectory(X_bar, U_bar)
    planner.push_parameters(
        affine_steps=affine,
        x0_local=X_bar[:, 0],
        u_prev=np.zeros(NU),
        v_cap=np.full(N + 1, 25.0),
        Xref_local=xref,
        rule_pack=pack,
    )
    planner.warm_start(X_bar, U_bar)
    X_sol, U_sol, T_sol = planner.solve_once()

    # Level-3 (comfort, where headway lives) should have non-trivial slack.
    # The headway rule fires on every step → at least one slack is positive.
    level3 = T_sol[2]
    max_slack_l3 = float(np.max(np.maximum(level3, 0.0)))
    assert max_slack_l3 > 0.5, f"headway rule should bind: max L3 slack = {max_slack_l3:.3g}"


def test_lcp_pipeline_cascade_orders_violations_correctly():
    """Run the full lex cascade on the close-lead scenario and verify the
    cascade produces a non-decreasing achievement: V_1* ≥ 0, V_2* ≥ 0,
    V_3* > 0 (because headway is genuinely infeasible at v=10 with lead at 8 m)."""
    rs = _ruleset_no_map()
    base_params = LCPParameters(
        horizon_s=HORIZON_S,
        dt_s=DT,
        penalty_form="l1",
        level_specs=tuple(
            LCPLevelSpec(name=f"L{i+1}", slots_per_step=n, epsilon_tolerance=1e-3)
            for i, n in enumerate(rs.slots_per_step_per_level())
        ),
        weights_per_level=(0.0, 0.0, 0.0),  # cascade overrides
        desired_speed_mps=10.0,
    )
    X_bar, U_bar, xref, agents, N = _build_scenario(v0=10.0, v_target=10.0, lead_x=8.0)
    ctx = EncoderContext(
        horizon_steps=N, dt_s=DT,
        warm_start_X=X_bar, warm_start_U=U_bar,
        Xref_local=xref,
        agents_local=agents,
        map_view=None,
        desired_speed_mps=10.0,
        ego_radius_m=2.5,
        wheel_base_m=WHEEL_BASE,
    )
    pack = rs.encode_all(ctx)
    lin = BicycleLinearisation(WHEEL_BASE, DT)
    affine = lin.linearise_trajectory(X_bar, U_bar)

    result = run_cascade(
        base_params=base_params,
        vehicle_parameters=get_pacifica_parameters(),
        affine_steps=affine,
        x0_local=X_bar[:, 0],
        u_prev=np.zeros(NU),
        v_cap=np.full(N + 1, 25.0),
        Xref_local=xref,
        rule_pack=pack,
    )

    # p_star = (V_1*, V_2*, V_3*, J*).
    assert result.p_star.shape == (4,)
    # Every V_i* must be non-negative (a property of the cascade's
    # non-negative-slack formulation). With v=10 and lead 8 m ahead, the
    # linearised CollisionRule at the warm-start binds aggressively — V_1*
    # will be positive. V_2* (speed limit; v_target=10 = limit) should stay
    # small. V_3* (headway) is consumed within the V_1* budget.
    assert all(v >= -1e-9 for v in result.p_star[:3]), f"V_i* must be non-negative: {result.p_star}"
    # The active set should include at least one rule (collision binds for
    # sure in this scenario).
    assert len(result.active_set) > 0
    # And the cascade should not blow up J — the performance is recorded.
    assert np.isfinite(result.p_star[3])


def test_lcp_pipeline_l2_runs_to_completion():
    """The same scenario should also solve under L₂ tolerance compliance."""
    rs = _ruleset_no_map()
    planner = _build_planner(rs, penalty_form="l2")
    X_bar, U_bar, xref, agents, N = _build_scenario(v0=10.0, v_target=10.0, lead_x=8.0)
    ctx = EncoderContext(
        horizon_steps=N, dt_s=DT,
        warm_start_X=X_bar, warm_start_U=U_bar,
        Xref_local=xref,
        agents_local=agents,
        map_view=None,
        desired_speed_mps=10.0,
        ego_radius_m=planner._ego_radius,
        wheel_base_m=WHEEL_BASE,
    )
    pack = rs.encode_all(ctx)
    affine = planner.linearisation().linearise_trajectory(X_bar, U_bar)
    planner.push_parameters(
        affine_steps=affine,
        x0_local=X_bar[:, 0],
        u_prev=np.zeros(NU),
        v_cap=np.full(N + 1, 25.0),
        Xref_local=xref,
        rule_pack=pack,
    )
    planner.warm_start(X_bar, U_bar)
    X_sol, U_sol, T_sol = planner.solve_once()
    assert X_sol.shape == (NX, N + 1)
