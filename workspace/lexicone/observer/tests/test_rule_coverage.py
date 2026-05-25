"""Per-rule coverage tests.

One test per rule that was previously exercised only by the end-to-end
pipeline run. Each test constructs a minimal scene that makes the target
rule applicable, then asserts (a) the rule's applicability gate fires and
(b) where a clear violating configuration exists, the violation rate is
positive.
"""

from __future__ import annotations

import math

from lexicone.observer import RuleEngine
from lexicone.observer.types import (
    CrosswalkSnapshot,
    IntersectionSnapshot,
)
from lexicone.observer.types import AgentType
from lexicone.observer.tests.synthetic import (
    build_straight_road_scene,
    make_agent,
    rectangle_polygon,
    straight_lane,
)


def _eval(engine: RuleEngine, snap, rule_id: str):
    engine.step(snap)
    return next(e for e in engine.history[-1] if e.rule_id == rule_id)


# ----- 0r2 longitudinal comfort -----


def test_longitudinal_comfort_violates_on_high_ax():
    eng = RuleEngine()
    snap = build_straight_road_scene(ego_speed=10.0, ax=5.0)
    e = _eval(eng, snap, "0r2")
    assert e.applies
    assert e.is_violated
    assert e.details["violation"]["ax_mps2"] == 5.0


def test_longitudinal_comfort_no_violation_at_low_ax():
    eng = RuleEngine()
    snap = build_straight_road_scene(ego_speed=10.0, ax=0.5)
    e = _eval(eng, snap, "0r2")
    assert e.applies
    assert not e.is_violated


# ----- 0r3 lateral comfort -----


def test_lateral_comfort_violates_on_high_ay():
    eng = RuleEngine()
    snap = build_straight_road_scene(ego_speed=10.0, ay=4.0)
    e = _eval(eng, snap, "0r3")
    assert e.applies
    assert e.is_violated


def test_lateral_comfort_not_applicable_when_stationary():
    eng = RuleEngine()
    snap = build_straight_road_scene(ego_speed=0.05)
    e = _eval(eng, snap, "0r3")
    assert not e.applies


# ----- 1r0 yield priority -----


def test_yield_priority_close_pedestrian_with_closing_ego():
    ped = make_agent("p1", AgentType.PEDESTRIAN, x=4.0, y=0.5)
    eng = RuleEngine()
    snap = build_straight_road_scene(ego_speed=8.0, agents=[ped])
    e = _eval(eng, snap, "1r0")
    assert e.applies
    assert e.is_violated


# ----- 1r2 block the box -----


def test_block_the_box_stopped_in_intersection_with_blocker():
    intersection = IntersectionSnapshot(
        intersection_id="i1",
        polygon=rectangle_polygon(0, 0, 10, 10),
    )
    blocker = make_agent("b1", AgentType.VEHICLE, x=6.0, y=0.0, heading=0.0, speed=0.0)
    eng = RuleEngine()
    snap = build_straight_road_scene(
        ego_speed=0.1,
        intersections=[intersection],
        agents=[blocker],
    )
    e = _eval(eng, snap, "1r2")
    assert e.applies
    assert e.is_violated


def test_block_the_box_not_applicable_when_moving():
    intersection = IntersectionSnapshot(
        intersection_id="i1",
        polygon=rectangle_polygon(0, 0, 10, 10),
    )
    eng = RuleEngine()
    snap = build_straight_road_scene(ego_speed=5.0, intersections=[intersection])
    e = _eval(eng, snap, "1r2")
    assert not e.applies


# ----- 1r5 uncontrolled intersection -----


def test_uncontrolled_intersection_with_cross_traffic():
    intersection = IntersectionSnapshot(
        intersection_id="i1",
        polygon=rectangle_polygon(0, 0, 12, 12),
    )
    # Cross-traffic vehicle to the ego's right (lat<0 in ego frame), close
    # enough to the conflict point that it reaches it well before the ego.
    cross = make_agent(
        "c1", AgentType.VEHICLE, x=6.0, y=-2.0, heading=math.pi / 2.0, speed=8.0
    )
    eng = RuleEngine()
    snap = build_straight_road_scene(
        ego_speed=2.0,
        intersections=[intersection],
        agents=[cross],
    )
    e = _eval(eng, snap, "1r5")
    assert e.applies
    assert e.is_violated


# ----- 2r2 route adherence -----


def test_route_adherence_off_route_lane():
    eng = RuleEngine()
    # Ego is in the default "lane0" but the route says only "lane_X" is valid.
    snap = build_straight_road_scene(ego_speed=10.0, route_lane_ids=["lane_X"])
    e = _eval(eng, snap, "2r2")
    assert e.applies
    assert e.is_violated


def test_route_adherence_on_route():
    eng = RuleEngine()
    snap = build_straight_road_scene(ego_speed=10.0, route_lane_ids=["lane0"])
    e = _eval(eng, snap, "2r2")
    assert e.applies
    assert not e.is_violated


def test_route_adherence_not_applicable_without_route():
    eng = RuleEngine()
    snap = build_straight_road_scene(ego_speed=10.0)
    e = _eval(eng, snap, "2r2")
    assert not e.applies


# ----- 3r5 lateral clearance -----


def test_lateral_clearance_violates_with_close_neighbor():
    neighbor = make_agent("n1", AgentType.VEHICLE, x=0.0, y=1.5, heading=0.0, speed=5.0)
    eng = RuleEngine()
    snap = build_straight_road_scene(ego_speed=5.0, agents=[neighbor])
    e = _eval(eng, snap, "3r5")
    assert e.applies
    assert e.is_violated


# ----- 3r6 lane intrusion -----


def test_lane_intrusion_violates_when_neighbor_closing_laterally():
    # Neighbor on the ego's left moving right toward ego.
    neighbor = make_agent(
        "n1", AgentType.VEHICLE, x=0.0, y=2.5, heading=-math.pi / 2.0, speed=1.0
    )
    eng = RuleEngine()
    snap = build_straight_road_scene(ego_speed=5.0, agents=[neighbor])
    e = _eval(eng, snap, "3r6")
    assert e.applies
    assert e.is_violated


# ----- 7r0 drivable boundary -----


def test_drivable_boundary_violates_when_off_road():
    # Default drivable area is 200x12 centered at origin (y in [-6, 6]).
    eng = RuleEngine()
    snap = build_straight_road_scene(ego_y=8.0, ego_speed=5.0)
    e = _eval(eng, snap, "7r0")
    assert e.applies
    assert e.is_violated


# ----- 7r2 opposing lane -----


def test_opposing_lane_violates_when_ego_in_opposite_lane():
    # Add an opposite-direction lane that contains the ego.
    opp = straight_lane(lane_id="opp", y_center=0.0, width=3.5, heading=math.pi, speed_limit_mps=None)
    eng = RuleEngine()
    snap = build_straight_road_scene(ego_speed=5.0, extra_lanes=[opp])
    e = _eval(eng, snap, "7r2")
    assert e.applies
    assert e.is_violated


# ----- 7r3 one-way direction -----


def test_one_way_direction_violates_when_wrong_way():
    opp = straight_lane(lane_id="opp", y_center=0.0, width=3.5, heading=math.pi, speed_limit_mps=None)
    eng = RuleEngine()
    snap = build_straight_road_scene(ego_speed=5.0, extra_lanes=[opp])
    e = _eval(eng, snap, "7r3")
    assert e.applies
    assert e.is_violated


# ----- 7r4 stop in crosswalk -----


def test_stop_in_crosswalk_violates_when_dwelling():
    crosswalk = CrosswalkSnapshot(crosswalk_id="cw1", polygon=rectangle_polygon(0, 0, 4, 4))
    eng = RuleEngine()
    snap = build_straight_road_scene(ego_speed=0.1, crosswalks=[crosswalk])
    e = _eval(eng, snap, "7r4")
    assert e.applies
    assert e.is_violated


# ----- 8r1 crosswalk pedestrian yield -----


def test_crosswalk_pedestrian_yield_violates_when_speeding_through():
    # Place the crosswalk so the ego's footprint already overlaps it, with a
    # pedestrian on it. The rule's swept-forward projection then sees a
    # conflict and applies; ego speed above the yield threshold triggers a
    # violation.
    crosswalk = CrosswalkSnapshot(crosswalk_id="cw1", polygon=rectangle_polygon(3, 0, 4, 4))
    ped = make_agent("p1", AgentType.PEDESTRIAN, x=3.0, y=0.0)
    eng = RuleEngine()
    snap = build_straight_road_scene(ego_speed=8.0, crosswalks=[crosswalk], agents=[ped])
    e = _eval(eng, snap, "8r1")
    assert e.applies
    assert e.is_violated


# ----- 9r1 non-traversable surface -----


def test_non_traversable_surface_violates_when_off_drivable():
    eng = RuleEngine()
    snap = build_straight_road_scene(ego_y=8.0, ego_speed=5.0)
    e = _eval(eng, snap, "9r1")
    assert e.applies
    assert e.is_violated


# ----- 10r3 unmarked crosswalk yield -----


def test_unmarked_crosswalk_yield_in_intersection_with_pedestrian():
    intersection = IntersectionSnapshot(
        intersection_id="i1",
        polygon=rectangle_polygon(0, 0, 12, 12),
    )
    ped = make_agent("p1", AgentType.PEDESTRIAN, x=4.0, y=2.0)
    eng = RuleEngine()
    snap = build_straight_road_scene(
        ego_speed=3.0,
        intersections=[intersection],
        agents=[ped],
    )
    e = _eval(eng, snap, "10r3")
    assert e.applies
    assert e.is_violated


# ----- 10r4 cyclist passing -----


def test_cyclist_passing_violates_with_small_lateral_gap():
    cyclist = make_agent("b1", AgentType.BICYCLE, x=2.0, y=1.0, heading=0.0, speed=2.0)
    eng = RuleEngine()
    snap = build_straight_road_scene(ego_speed=8.0, agents=[cyclist])
    e = _eval(eng, snap, "10r4")
    assert e.applies
    assert e.is_violated
