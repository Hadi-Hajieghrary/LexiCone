"""Smoke tests for the rule engine.

These tests use synthetic scenes (no nuplan-devkit) to verify:
- the engine can be constructed with the default 25-rule set;
- each rule's applicability gate fires under the conditions it was designed
  for and does not fire when conditions don't match;
- violation rates are non-zero for clearly non-compliant scenes;
- the windowed summary correctly integrates per-tick rates.
"""

from __future__ import annotations

import pytest

from lexicone.observer import RuleEngine, build_default_rules
from lexicone.observer.types import (
    AgentType,
    IntersectionSnapshot,
    StopLineSnapshot,
    StopType,
    TrafficLightState,
    TrafficLightStatus,
    WalkwaySnapshot,
)
from lexicone.observer.tests.synthetic import (
    build_straight_road_scene,
    make_agent,
    rectangle_polygon,
    straight_lane,
)


def test_registry_has_25_rules():
    rules = build_default_rules()
    assert len(rules) == 25
    ids = {r.id for r in rules}
    assert "10r0" in ids
    assert "0r3" in ids


def test_engine_runs_empty_scene_without_errors():
    eng = RuleEngine()
    snap = build_straight_road_scene()
    evals = eng.step(snap)
    assert len(evals) == 25
    # No violations on a clean empty scene at posted-limit speed.
    assert eng.current_violations() == []


def test_speed_limit_violation_triggers():
    eng = RuleEngine()
    snap = build_straight_road_scene(ego_speed=30.0, speed_limit_mps=13.4)
    eng.step(snap)
    by_id = {e.rule_id: e for e in eng.history[-1]}
    e = by_id["3r0"]
    assert e.applies
    assert e.is_violated
    assert e.violation_rate > 0
    assert e.details["violation"]["overshoot_mps"] == pytest.approx(30.0 - (13.4 + 1.0), rel=1e-3)


def test_speed_limit_not_applicable_without_lane_limit():
    eng = RuleEngine()
    snap = build_straight_road_scene(ego_speed=20.0, speed_limit_mps=None)
    eng.step(snap)
    e = next(e for e in eng.history[-1] if e.rule_id == "3r0")
    assert not e.applies


def test_vru_collision_when_pedestrian_under_wheel():
    ped = make_agent("p1", AgentType.PEDESTRIAN, x=1.0, y=0.0, length=0.6, width=0.6)
    eng = RuleEngine()
    snap = build_straight_road_scene(agents=[ped])
    eng.step(snap)
    e = next(e for e in eng.history[-1] if e.rule_id == "10r0")
    assert e.applies
    assert e.is_violated
    assert e.violation_rate > 0.0


def test_vru_collision_not_violated_at_safe_distance():
    ped = make_agent("p1", AgentType.PEDESTRIAN, x=20.0, y=10.0)
    eng = RuleEngine()
    snap = build_straight_road_scene(agents=[ped])
    eng.step(snap)
    e = next(e for e in eng.history[-1] if e.rule_id == "10r0")
    assert e.applies
    assert not e.is_violated


def test_vehicle_collision_triggers_on_overlap():
    other = make_agent("c1", AgentType.VEHICLE, x=2.0, y=0.0, heading=0.0)
    eng = RuleEngine()
    snap = build_straight_road_scene(agents=[other])
    eng.step(snap)
    e = next(e for e in eng.history[-1] if e.rule_id == "9r0")
    assert e.applies and e.is_violated


def test_lateral_acceleration_comfort_violation():
    eng = RuleEngine()
    snap = build_straight_road_scene(ego_speed=10.0, yaw_rate=0.6, ay=6.0)
    eng.step(snap)
    e = next(e for e in eng.history[-1] if e.rule_id == "1r11")
    assert e.applies and e.is_violated


def test_safe_headway_violation():
    leader = make_agent("L", AgentType.VEHICLE, x=6.0, y=0.0, heading=0.0, speed=2.0)
    eng = RuleEngine()
    snap = build_straight_road_scene(ego_speed=10.0, agents=[leader])
    eng.step(snap)
    e = next(e for e in eng.history[-1] if e.rule_id == "3r3")
    assert e.applies
    assert e.is_violated
    assert e.details["violation"]["thw_s"] < 1.5


def test_safe_headway_not_applicable_without_leader():
    eng = RuleEngine()
    snap = build_straight_road_scene(ego_speed=10.0)
    eng.step(snap)
    e = next(e for e in eng.history[-1] if e.rule_id == "3r3")
    assert not e.applies


def test_traffic_light_red_in_intersection():
    intersection = IntersectionSnapshot(
        intersection_id="i1",
        polygon=rectangle_polygon(0, 0, 10, 10),
    )
    lc = straight_lane(
        lane_id="lc1",
        is_lane_connector=True,
        speed_limit_mps=13.4,
    )
    snap = build_straight_road_scene(
        ego_speed=3.0,
        intersections=[intersection],
        lane_connectors=[lc],
        traffic_lights=[TrafficLightStatus(lane_connector_id="lc1", state=TrafficLightState.RED)],
    )
    eng = RuleEngine()
    eng.step(snap)
    e = next(e for e in eng.history[-1] if e.rule_id == "7r1")
    assert e.applies
    assert e.is_violated


def test_mandatory_stop_rolling_through():
    stop_polyline = [(4.0, -2.0), (4.0, 2.0)]
    sl = StopLineSnapshot(
        stop_line_id="sl1",
        polyline=stop_polyline,
        stop_type=StopType.STOP_SIGN,
        associated_lane_id="lane0",
    )
    snap = build_straight_road_scene(ego_x=3.5, ego_speed=4.0, stop_lines=[sl])
    eng = RuleEngine()
    eng.step(snap)
    e = next(e for e in eng.history[-1] if e.rule_id == "8r0")
    assert e.applies
    assert e.is_violated


def test_mandatory_stop_complies_when_stopped():
    stop_polyline = [(4.0, -2.0), (4.0, 2.0)]
    sl = StopLineSnapshot(
        stop_line_id="sl1",
        polyline=stop_polyline,
        stop_type=StopType.STOP_SIGN,
        associated_lane_id="lane0",
    )
    snap = build_straight_road_scene(ego_x=3.0, ego_speed=0.05, stop_lines=[sl])
    eng = RuleEngine()
    eng.step(snap)
    e = next(e for e in eng.history[-1] if e.rule_id == "8r0")
    assert e.applies
    assert not e.is_violated


def test_window_summary_integration():
    eng = RuleEngine()
    for k in range(11):
        snap = build_straight_road_scene(
            ego_speed=13.4 + 1.0 + 5.0,
            timestamp_us=k * 100_000,
        )
        eng.step(snap)
    summary = eng.summary()
    s = summary.rule_summaries["3r0"]
    assert s.n_steps_applicable == 11
    assert s.n_steps_violated == 11
    assert s.integrated_violation == pytest.approx(11 * 25.0 * 0.1, rel=1e-2)


def test_window_subset():
    eng = RuleEngine()
    for k in range(10):
        speed = 25.0 if k < 5 else 12.0
        snap = build_straight_road_scene(ego_speed=speed, timestamp_us=k * 100_000)
        eng.step(snap)
    full = eng.summary()
    early = eng.summary(window_s=(0.0, 0.5))
    late = eng.summary(window_s=(0.5, 1.0))
    assert full.rule_summaries["3r0"].n_steps_violated == 5
    assert early.rule_summaries["3r0"].n_steps_violated >= 4
    assert late.rule_summaries["3r0"].n_steps_violated == 0


def test_sidewalk_drive_when_on_walkway():
    walkway = WalkwaySnapshot(walkway_id="w1", polygon=rectangle_polygon(0, 5, 10, 4))
    snap = build_straight_road_scene(ego_y=5.0, ego_speed=2.0, walkways=[walkway])
    eng = RuleEngine()
    eng.step(snap)
    e = next(e for e in eng.history[-1] if e.rule_id == "7r5")
    assert e.applies and e.is_violated


def test_bike_lane_encroachment():
    bike = straight_lane(lane_id="bk1", y_center=2.5, width=1.5, speed_limit_mps=None, is_bike_lane=True)
    snap = build_straight_road_scene(ego_y=2.5, ego_speed=5.0, bike_lanes=[bike])
    eng = RuleEngine()
    eng.step(snap)
    e = next(e for e in eng.history[-1] if e.rule_id == "10r5")
    assert e.applies and e.is_violated


def test_no_violations_in_clean_replay():
    eng = RuleEngine()
    for k in range(5):
        snap = build_straight_road_scene(ego_speed=12.0, timestamp_us=k * 100_000)
        eng.step(snap)
    summary = eng.summary()
    violators = [s for s in summary.rule_summaries.values() if s.n_steps_violated > 0]
    assert violators == []


def test_current_applicable_rules_filters_to_active_set():
    eng = RuleEngine()
    snap = build_straight_road_scene(ego_speed=10.0)
    eng.step(snap)
    applicable = eng.current_applicable_rules()
    ids = {e.rule_id for e in applicable}
    assert "3r0" in ids
    assert "0r2" in ids
