"""Tests for the rule encoder framework and concrete rule implementations."""

from __future__ import annotations

import math

import numpy as np
import pytest

from lexicone.planning.bicycle_model import NU, NX
from lexicone.planning.lcp_mpc import LinearisedRuleConstraint, make_inactive_constraint
from lexicone.planning.rule_encoder import (
    AgentSlot,
    CollisionRule,
    DrivableBoundaryRule,
    EncoderContext,
    LaneCorridorRule,
    LateralAccelerationRule,
    LongitudinalComfortRule,
    RuleSet,
    SafeHeadwayRule,
    SidewalkDriveRule,
    SpeedLimitRule,
    StubRule,
    make_default_ruleset,
)


def _ctx(
    N: int = 10,
    dt: float = 0.1,
    agents=(),
    map_view=None,
    desired_speed: float = 12.0,
    v0: float = 5.0,
) -> EncoderContext:
    X = np.zeros((NX, N + 1))
    U = np.zeros((NU, N))
    X[:, 0] = np.array([0.0, 0.0, 0.0, v0])
    for k in range(N):
        X[0, k + 1] = X[0, k] + v0 * dt
        X[3, k + 1] = v0
    Xref = X.copy()
    Xref[3, :] = desired_speed
    return EncoderContext(
        horizon_steps=N,
        dt_s=dt,
        warm_start_X=X,
        warm_start_U=U,
        Xref_local=Xref,
        agents_local=tuple(agents),
        map_view=map_view,
        desired_speed_mps=desired_speed,
        ego_radius_m=2.5,
        wheel_base_m=3.089,
    )


# ----------------------------------------------------------------------
# Speed-limit rule (simplest case)
# ----------------------------------------------------------------------


def test_speed_limit_rule_always_applies_and_has_one_slot_per_step():
    rule = SpeedLimitRule()
    ctx = _ctx(N=5, desired_speed=12.0)
    assert rule.applies_to_horizon(ctx)
    encoded = rule.encode(ctx)
    assert len(encoded) == 5
    for k in range(5):
        assert len(encoded[k]) == 1
        c = encoded[k][0]
        # a should pick up only the velocity component
        assert c.a[3] == pytest.approx(1.0)
        assert c.a[0] == 0.0 and c.a[1] == 0.0 and c.a[2] == 0.0
        # e = -(v_lim + tol) — Xref's v=12, tol=1 → e=-13
        assert c.e == pytest.approx(-13.0)
        assert c.mask == 1.0


# ----------------------------------------------------------------------
# Longitudinal comfort rule (linear in u, 2 slots)
# ----------------------------------------------------------------------


def test_longitudinal_comfort_two_slots_symmetric_box():
    rule = LongitudinalComfortRule(a_max_comf_mps2=1.8)
    ctx = _ctx(N=3)
    assert rule.applies_to_horizon(ctx)
    encoded = rule.encode(ctx)
    assert len(encoded) == 3
    for slots in encoded:
        assert len(slots) == 2
        c_plus, c_minus = slots
        # +a slot: b = (1, 0), e = -1.8
        np.testing.assert_allclose(c_plus.b, [1.0, 0.0])
        assert c_plus.e == pytest.approx(-1.8)
        # -a slot: b = (-1, 0), e = -1.8
        np.testing.assert_allclose(c_minus.b, [-1.0, 0.0])
        assert c_minus.e == pytest.approx(-1.8)


# ----------------------------------------------------------------------
# Lateral acceleration rule (nonlinear → linearised)
# ----------------------------------------------------------------------


def test_lateral_acceleration_zero_at_zero_warm_start():
    """At v̄=0, δ̄=0 the lateral accel is zero and gradients vanish; the
    linearised inequality reduces to the constant ``-a_max ≤ t``."""
    rule = LateralAccelerationRule(a_y_max_comf_mps2=2.0)
    ctx = _ctx(N=2, v0=0.0)  # warm-start velocity 0
    encoded = rule.encode(ctx)
    c_plus, c_minus = encoded[0]
    assert c_plus.e == pytest.approx(-2.0)
    assert c_minus.e == pytest.approx(-2.0)


def test_lateral_acceleration_nonzero_at_high_velocity():
    """At v̄=10 m/s, δ̄=0, the linearised constraint couples δ into the
    inequality with slope v̄²/L = 100/3.089 ≈ 32.4."""
    rule = LateralAccelerationRule(a_y_max_comf_mps2=2.0)
    N = 2
    ctx = _ctx(N=N, v0=10.0)
    encoded = rule.encode(ctx)
    c_plus, _ = encoded[0]
    expected_slope_d = 10.0 * 10.0 / 3.089
    assert c_plus.b[1] == pytest.approx(expected_slope_d, rel=1e-3)


# ----------------------------------------------------------------------
# Collision rule
# ----------------------------------------------------------------------


def test_collision_rule_no_agents_means_no_apply():
    rule = CollisionRule(slots_per_step=4)
    ctx = _ctx(agents=())
    assert not rule.applies_to_horizon(ctx)


def test_collision_rule_with_close_agent_applies_and_encodes():
    rule = CollisionRule(slots_per_step=4, collision_buffer_m=0.5, max_distance_m=40.0)
    ag = AgentSlot(
        track_id="lead",
        x=10.0, y=0.0, vx=0.0, vy=0.0,
        length=4.5, width=1.8, is_vru=False,
    )
    ctx = _ctx(agents=(ag,))
    assert rule.applies_to_horizon(ctx)
    encoded = rule.encode(ctx)
    assert len(encoded) == ctx.horizon_steps
    for slots in encoded:
        assert len(slots) == 4
        # First slot is the close agent; remaining are inactive padding.
        active = slots[0]
        assert active.mask == 1.0
        assert slots[1].mask == 0.0  # padded


def test_collision_rule_vru_inflates_radius():
    """A pedestrian close to the ego should produce a constraint with a
    *larger* offset ``e`` (more violation room) than a same-distance vehicle
    of the same size, because the VRU radius is inflated."""
    pedestrian = AgentSlot(
        "ped", x=10.0, y=0.0, vx=0.0, vy=0.0, length=0.6, width=0.6, is_vru=True
    )
    vehicle = AgentSlot(
        "veh", x=10.0, y=0.0, vx=0.0, vy=0.0, length=0.6, width=0.6, is_vru=False
    )
    rule = CollisionRule(slots_per_step=1, vru_inflate_m=0.5, collision_buffer_m=0.0)
    ped_e = rule.encode(_ctx(agents=(pedestrian,)))[0][0].e
    veh_e = rule.encode(_ctx(agents=(vehicle,)))[0][0].e
    # VRU's r_min is larger → f_bar = r_min² - dx² is larger → e is larger.
    assert ped_e > veh_e


# ----------------------------------------------------------------------
# Safe headway rule
# ----------------------------------------------------------------------


def test_safe_headway_applies_when_lead_in_lane():
    rule = SafeHeadwayRule(time_headway_s=1.2, min_gap_m=2.0)
    lead = AgentSlot("lead", x=15.0, y=0.0, vx=0.0, vy=0.0, length=4.5, width=1.8, is_vru=False)
    ctx = _ctx(agents=(lead,))
    assert rule.applies_to_horizon(ctx)


def test_safe_headway_does_not_apply_when_lead_far_off_lane():
    rule = SafeHeadwayRule(time_headway_s=1.2, lateral_tol_m=1.6)
    lead = AgentSlot("lead", x=15.0, y=5.0, vx=0.0, vy=0.0, length=4.5, width=1.8, is_vru=False)
    ctx = _ctx(agents=(lead,))
    assert not rule.applies_to_horizon(ctx)


def test_safe_headway_linearised_constraint_couples_x_and_v():
    rule = SafeHeadwayRule(time_headway_s=1.2, min_gap_m=2.0)
    lead = AgentSlot("lead", x=15.0, y=0.0, vx=0.0, vy=0.0, length=4.5, width=1.8, is_vru=False)
    ctx = _ctx(agents=(lead,))
    encoded = rule.encode(ctx)
    c = encoded[0][0]
    # a couples position (x) and velocity (v)
    assert c.a[0] == pytest.approx(1.0)
    assert c.a[3] == pytest.approx(1.2)


# ----------------------------------------------------------------------
# RuleSet aggregation
# ----------------------------------------------------------------------


def test_ruleset_aggregates_per_level_slot_counts():
    rs = RuleSet(levels=[
        [SpeedLimitRule(), LongitudinalComfortRule()],     # 1 + 2 = 3 slots
        [LateralAccelerationRule()],                       # 2 slots
        [],
    ])
    counts = rs.slots_per_step_per_level()
    assert counts == [3, 2, 0]


def test_ruleset_encode_all_produces_correct_shape():
    rs = make_default_ruleset()
    counts = rs.slots_per_step_per_level()
    # Default ruleset slot budgets:
    #   safety = 8 (collision) + 2 (LaneCorridor) + 4 (stub 7r5) + 2 (stub 10r5) = 16
    #   legal  = 1 (SpeedLimit) + 2 + 2 + 1 + 1 stubs = 7
    #   comfort= 1 (Headway) + 2 (LonComf) + 2 (LatAcc) + 2 + 2 + 1 stubs = 10
    assert counts == [16, 7, 10]
    pack = rs.encode_all(_ctx())
    assert len(pack.constraints_by_level) == 3
    for i, level_constraints in enumerate(pack.constraints_by_level):
        assert len(level_constraints) == 10  # horizon_steps
        for step_constraints in level_constraints:
            assert len(step_constraints) == counts[i]


def test_default_ruleset_only_safe_subset_is_active():
    """Under the test context (empty agent list, no map view), the always-on
    rules are: CollisionRule (only applies when an agent is in ROI — here it
    does not), LaneCorridorRule, SpeedLimitRule, LongitudinalComfortRule,
    LateralAccelerationRule. SafeHeadwayRule only fires when there is an
    in-lane lead — here none. With the test context, level-2's only enabled
    rule is SpeedLimitRule (slot 0); the other 6 slots are stub-inactive."""
    rs = make_default_ruleset()
    pack = rs.encode_all(_ctx())
    # Inspect level-2 (legal): the only enabled rule is SpeedLimitRule
    # (slot 0). Slots 1-6 are stubs → all-inactive (mask=0).
    legal_constraints_first_step = pack.constraints_by_level[1][0]
    assert legal_constraints_first_step[0].mask == 1.0  # SpeedLimitRule active
    for c in legal_constraints_first_step[1:]:
        assert c.mask == 0.0  # 7r2 / 7r3 / 7r1 / 7r4 are stubs


# ----------------------------------------------------------------------
# Lane corridor rule (7r0)
# ----------------------------------------------------------------------


def test_lane_corridor_rule_two_slots_per_step():
    rule = LaneCorridorRule(slots_per_step=2, half_width_m=1.0)
    ctx = _ctx(N=5)
    assert rule.applies_to_horizon(ctx)
    encoded = rule.encode(ctx)
    assert len(encoded) == 5
    for slots in encoded:
        assert len(slots) == 2


def test_lane_corridor_constraints_at_straight_reference_along_x():
    """For a reference along +x at the origin (no rotation), the corridor
    constraints reduce to ``|y| <= half_width``. ``a`` should isolate the
    y-coordinate; ``e`` should equal ``-half_width``."""
    rule = LaneCorridorRule(slots_per_step=2, half_width_m=1.0)
    # Build a context whose Xref_local is along +x with psi_ref = 0.
    N = 3
    ctx = _ctx(N=N, v0=5.0)
    # Override Xref to point straight along +x with psi=0 and y=0.
    ctx.Xref_local[:] = 0.0
    for k in range(N + 1):
        ctx.Xref_local[0, k] = 5.0 * 0.1 * k
        ctx.Xref_local[2, k] = 0.0
        ctx.Xref_local[3, k] = 5.0
    encoded = rule.encode(ctx)
    c_left, c_right = encoded[0]
    # Left constraint: y <= +half_width  →  a=(0,1,0,0), e=-half_width
    np.testing.assert_allclose(c_left.a, [0.0, 1.0, 0.0, 0.0], atol=1e-9)
    assert c_left.e == pytest.approx(-1.0, abs=1e-9)
    # Right constraint: -y <= +half_width  →  a=(0,-1,0,0), e=-half_width
    np.testing.assert_allclose(c_right.a, [0.0, -1.0, 0.0, 0.0], atol=1e-9)
    assert c_right.e == pytest.approx(-1.0, abs=1e-9)


def test_lane_corridor_constraints_satisfied_at_reference_centreline():
    """Evaluated at any point on the reference centreline, the corridor
    constraint slack is exactly ``-half_width`` (no slack needed; the ego is
    at the centre of the corridor)."""
    rule = LaneCorridorRule(slots_per_step=2, half_width_m=1.0)
    N = 3
    ctx = _ctx(N=N, v0=5.0)
    # Reference along +x at y=0; warm-start is also along +x at y=0 (the
    # default _ctx setup), so the ego sits on the centreline.
    encoded = rule.encode(ctx)
    for k in range(N):
        c_left, c_right = encoded[k]
        x_bar = ctx.warm_start_X[0, k]
        y_bar = ctx.warm_start_X[1, k]
        # Slack required at warm-start: a^T x + e (without the t).
        lhs_left = c_left.a[0] * x_bar + c_left.a[1] * y_bar + c_left.e
        lhs_right = c_right.a[0] * x_bar + c_right.a[1] * y_bar + c_right.e
        # Both should be ≤ 0 since the warm-start is inside the corridor.
        assert lhs_left <= 1e-9
        assert lhs_right <= 1e-9


def test_drivable_boundary_rule_now_disabled():
    """The legacy :class:`DrivableBoundaryRule` is preserved for
    documentation but ``applies_to_horizon`` always returns False so it
    cannot accidentally be re-wired into a working ruleset."""
    rule = DrivableBoundaryRule()
    ctx = _ctx()
    assert not rule.applies_to_horizon(ctx)


def test_stub_rule_returns_only_inactive_slots():
    stub = StubRule("test_stub", priority_level=2, slots_per_step=3, doc="test")
    ctx = _ctx()
    assert not stub.applies_to_horizon(ctx)
    inactive = stub.all_inactive(ctx)
    assert len(inactive) == ctx.horizon_steps
    for step in inactive:
        assert len(step) == 3
        for c in step:
            assert c.mask == 0.0
