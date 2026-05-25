"""Tests for the LCP-mode opt-in branch of :class:`TwoLevelMPCPlanner`.

We don't exercise the full simulator path here — that's the demo's job.
Instead we verify that:

1. The constructor accepts the new LCP-mode parameters without breaking the
   legacy constructor signature.
2. With ``penalty_form=None`` (default) the planner is backward-compatible
   and ``_penalty_form`` stays None.
3. With ``penalty_form="l1"`` or ``"l2"`` the LCP-mode flags are stored and
   the resolver populates the right downstream containers.
4. The Hydra YAML keys for the LCP mode match what the constructor expects.
"""

from __future__ import annotations

from typing import List, Optional

import pytest

from lexicone.planning.two_level_planner import TwoLevelMPCPlanner


def test_legacy_construction_keeps_penalty_form_none():
    """Default construction (no LCP keys) must remain backward-compatible."""
    planner = TwoLevelMPCPlanner(
        mpc_horizon_s=2.0,
        mpc_dt_s=0.1,
        desired_speed_mps=12.0,
    )
    assert planner._penalty_form is None
    assert planner._lcp_planner is None
    assert planner._lcp_ruleset is None
    assert planner._lcp_map_lifter is None
    assert planner._lcp_cache is None
    assert planner._lcp_compliance is None


def test_l1_construction_stores_lcp_flags_but_defers_full_init():
    """With ``penalty_form="l1"``, the LCP flags are stored; the heavy
    LCP plumbing (planner, map lifter, etc.) is not built until
    :meth:`initialize` is called — keeping construction cheap."""
    planner = TwoLevelMPCPlanner(
        mpc_horizon_s=2.0,
        mpc_dt_s=0.1,
        desired_speed_mps=12.0,
        penalty_form="l1",
        weights_per_level=["auto", "auto", "auto"],
        epsilon_per_level=[1e-4, 4e-2, 5e-1],
        scenario_class_hint="following_lane_with_lead",
    )
    assert planner._penalty_form == "l1"
    assert planner._lcp_weights_spec == ["auto", "auto", "auto"]
    assert planner._lcp_epsilon_per_level == [1e-4, 4e-2, 5e-1]
    assert planner._scenario_class_hint == "following_lane_with_lead"
    # Before initialize(), LCP-side containers are still None.
    assert planner._lcp_planner is None


def test_l2_construction_accepts_epsilon_vector():
    planner = TwoLevelMPCPlanner(
        mpc_horizon_s=3.0,
        penalty_form="l2",
        weights_per_level=[100.0, 10.0, 1.0],
        epsilon_per_level=[1e-2, 1e-1, 1e-1],
    )
    assert planner._penalty_form == "l2"
    assert planner._lcp_weights_spec == [100.0, 10.0, 1.0]
    assert planner._lcp_epsilon_per_level == [1e-2, 1e-1, 1e-1]


def test_invalid_penalty_form_rejected():
    with pytest.raises(ValueError, match="penalty_form"):
        TwoLevelMPCPlanner(penalty_form="huber")
    with pytest.raises(ValueError, match="penalty_form"):
        TwoLevelMPCPlanner(penalty_form="L1")  # case-sensitive — paper uses lowercase


def test_yaml_compatible_construction_with_full_lcp_keys():
    """The constructor must accept every keyword the YAML emits."""
    kwargs = dict(
        mpc_horizon_s=3.0,
        mpc_dt_s=0.1,
        replan_period_s=8.0,
        desired_speed_mps=12.0,
        occupancy_map_radius_m=40.0,
        global_lookahead_m=200.0,
        obstacle_slot_count=6,
        collision_buffer_m=0.4,
        max_accel_mps2=2.5,
        max_decel_mps2=3.5,
        max_speed_mps=25.0,
        max_steer_rad=0.5,
        max_steer_rate_radps=0.7,
        max_jerk_mps3=12.0,
        weight_pos=4.0,
        weight_heading=10.0,
        weight_speed=1.0,
        weight_control=0.05,
        weight_control_rate=0.2,
        weight_slack=500.0,
        # LCP-mode opt-in.
        penalty_form="l1",
        weights_per_level=["auto", "auto", "auto"],
        epsilon_per_level=[1.0e-4, 4.0e-2, 5.0e-1],
        scenario_class_hint="default",
        lcp_map_radius_m=80.0,
    )
    planner = TwoLevelMPCPlanner(**kwargs)
    assert planner.name() == "TwoLevelMPCPlanner"
    assert planner._penalty_form == "l1"


def test_legacy_yaml_keys_still_work():
    """Existing YAML configs (no LCP keys) construct successfully."""
    kwargs = dict(
        mpc_horizon_s=3.0,
        mpc_dt_s=0.1,
        replan_period_s=8.0,
        desired_speed_mps=12.0,
        occupancy_map_radius_m=40.0,
        global_lookahead_m=200.0,
        obstacle_slot_count=6,
        collision_buffer_m=0.4,
        max_accel_mps2=2.5,
        max_decel_mps2=3.5,
        max_speed_mps=25.0,
        max_steer_rad=0.5,
        max_steer_rate_radps=0.7,
        max_jerk_mps3=12.0,
        weight_pos=4.0,
        weight_heading=10.0,
        weight_speed=1.0,
        weight_control=0.05,
        weight_control_rate=0.2,
        weight_slack=500.0,
    )
    planner = TwoLevelMPCPlanner(**kwargs)
    assert planner._penalty_form is None
