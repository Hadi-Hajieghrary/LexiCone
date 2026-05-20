"""Crosswalk-related rules.

- 8r1 — Yield right-of-way to pedestrians at crosswalks.
- 10r3 — Yield right-of-way at unmarked crosswalks.
- 7r4 — Do not stop inside crosswalks.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional, Tuple

from shapely.geometry import Point

from ..geometry import (
    agent_footprint,
    ego_center,
    ego_footprint,
    planar_distance,
    polygon_from_points,
)
from ..rule import ObserverRule
from ..types import AgentType, SceneSnapshot
from ._common import VRU_TYPES, ego_in_intersection


def _conflicting_crosswalks(snap: SceneSnapshot, marked: bool):
    """Yield (crosswalk, polygon) pairs whose polygon intersects the ego footprint
    (or a short forward projection), filtered by ``is_marked``.
    """
    fp = ego_footprint(snap.ego)
    # Project the ego forward by a couple of seconds to detect upcoming crosswalks.
    cos_h = math.cos(snap.ego.pose.heading)
    sin_h = math.sin(snap.ego.pose.heading)
    forward_t = 2.0
    forward_d = max(snap.ego.speed, 5.0) * forward_t
    ec = ego_center(snap.ego)
    forward_point = Point(ec[0] + cos_h * forward_d, ec[1] + sin_h * forward_d)
    swept = fp.union(forward_point.buffer(snap.ego.width))
    for cw in snap.map.crosswalks:
        if cw.is_marked != marked:
            continue
        poly = polygon_from_points(cw.polygon)
        if poly is None:
            continue
        if swept.intersects(poly):
            yield cw, poly


def _pedestrians_on_or_near(polygon, snap: SceneSnapshot, near_m: float = 1.5):
    """Pedestrians (and cyclists treated as VRUs) on, or within ``near_m`` of, polygon."""
    inflated = polygon.buffer(near_m)
    out = []
    for a in snap.agents:
        ot = a.object_type.value if isinstance(a.object_type, AgentType) else a.object_type
        if ot not in VRU_TYPES:
            continue
        if inflated.contains(Point(a.pose.x, a.pose.y)):
            out.append(a)
    return out


class CrosswalkPedestrianYieldRule(ObserverRule):
    """8r1 — yield to pedestrians at marked crosswalks."""

    id = "8r1"
    level = 8
    name = "Yield right-of-way to pedestrians at crosswalks"
    description = (
        "Conflict between ego and any pedestrian/cyclist on or approaching a "
        "marked crosswalk in the ego's path."
    )

    def __init__(self, near_m: float = 1.5, yield_speed_mps: float = 1.0):
        self.near_m = near_m
        self.yield_speed_mps = yield_speed_mps

    def applies_at(self, snap: SceneSnapshot) -> Tuple[bool, Mapping[str, Any]]:
        for _, poly in _conflicting_crosswalks(snap, marked=True):
            peds = _pedestrians_on_or_near(poly, snap, self.near_m)
            if peds:
                return True, {"n_peds": len(peds)}
        return False, {}

    def violation_at(self, snap: SceneSnapshot) -> Tuple[float, Mapping[str, Any]]:
        worst_speed = 0.0
        worst_min_dist = math.inf
        for _, poly in _conflicting_crosswalks(snap, marked=True):
            peds = _pedestrians_on_or_near(poly, snap, self.near_m)
            if not peds:
                continue
            ec = ego_center(snap.ego)
            for p in peds:
                d = planar_distance(ec, (p.pose.x, p.pose.y))
                if d < worst_min_dist:
                    worst_min_dist = d
            if snap.ego.speed > self.yield_speed_mps:
                worst_speed = max(worst_speed, snap.ego.speed)
        rate = max(0.0, worst_speed - self.yield_speed_mps)
        return rate, {
            "ego_speed_mps": snap.ego.speed,
            "yield_threshold_mps": self.yield_speed_mps,
            "min_ped_distance_m": (
                worst_min_dist if worst_min_dist != math.inf else None
            ),
        }


class UnmarkedCrosswalkYieldRule(ObserverRule):
    """10r3 — at intersections, treat unmarked crossings as crosswalks."""

    id = "10r3"
    level = 10
    name = "Yield at unmarked crosswalks"
    description = (
        "At any intersection without a marked crosswalk, the ego must yield to "
        "pedestrians as if a crosswalk were present."
    )

    def __init__(self, near_m: float = 2.0, yield_speed_mps: float = 1.0):
        self.near_m = near_m
        self.yield_speed_mps = yield_speed_mps

    def applies_at(self, snap: SceneSnapshot) -> Tuple[bool, Mapping[str, Any]]:
        in_int, int_id = ego_in_intersection(snap)
        if not in_int:
            return False, {}
        # Check that no marked crosswalk covers the intersection footprint.
        # If a marked crosswalk exists *in* the intersection, prefer 8r1 over 10r3.
        for cw in snap.map.crosswalks:
            if cw.is_marked:
                poly = polygon_from_points(cw.polygon)
                if poly is None:
                    continue
                for inter in snap.map.intersections:
                    ipoly = polygon_from_points(inter.polygon)
                    if ipoly is not None and ipoly.intersects(poly):
                        return False, {"reason": "marked_crosswalk_present"}
        # Are pedestrians intending to cross?
        ec = ego_center(snap.ego)
        peds = [
            a for a in snap.agents
            if (a.object_type.value if isinstance(a.object_type, AgentType) else a.object_type)
            == AgentType.PEDESTRIAN.value
            and planar_distance(ec, (a.pose.x, a.pose.y)) <= 15.0
        ]
        return bool(peds), {"intersection_id": int_id, "n_peds_nearby": len(peds)}

    def violation_at(self, snap: SceneSnapshot) -> Tuple[float, Mapping[str, Any]]:
        ec = ego_center(snap.ego)
        worst_dist = math.inf
        for a in snap.agents:
            ot = a.object_type.value if isinstance(a.object_type, AgentType) else a.object_type
            if ot != AgentType.PEDESTRIAN.value:
                continue
            d = planar_distance(ec, (a.pose.x, a.pose.y))
            if d < worst_dist:
                worst_dist = d
        rate = max(0.0, snap.ego.speed - self.yield_speed_mps) if snap.ego.speed > self.yield_speed_mps else 0.0
        # If a pedestrian is very close and ego is moving, escalate.
        if worst_dist != math.inf and worst_dist < 3.0 and snap.ego.speed > 0.2:
            rate += (3.0 - worst_dist) * snap.ego.speed
        return rate, {
            "ego_speed_mps": snap.ego.speed,
            "min_ped_distance_m": worst_dist if worst_dist != math.inf else None,
        }


class StopInCrosswalkRule(ObserverRule):
    """7r4 — do not stop inside a crosswalk."""

    id = "7r4"
    level = 7
    name = "Do not stop inside crosswalks"
    description = (
        "Penalises dwell time with the ego footprint overlapping a crosswalk "
        "while the ego is essentially stopped."
    )

    def __init__(self, stop_speed_mps: float = 0.5):
        self.stop_speed_mps = stop_speed_mps

    def applies_at(self, snap: SceneSnapshot) -> Tuple[bool, Mapping[str, Any]]:
        if snap.ego.speed > self.stop_speed_mps:
            return False, {"ego_speed_mps": snap.ego.speed}
        fp = ego_footprint(snap.ego)
        for cw in snap.map.crosswalks:
            poly = polygon_from_points(cw.polygon)
            if poly is not None and fp.intersects(poly):
                return True, {"crosswalk_id": cw.crosswalk_id, "ego_speed_mps": snap.ego.speed}
        return False, {}

    def violation_at(self, snap: SceneSnapshot) -> Tuple[float, Mapping[str, Any]]:
        fp = ego_footprint(snap.ego)
        total = 0.0
        for cw in snap.map.crosswalks:
            poly = polygon_from_points(cw.polygon)
            if poly is not None and fp.intersects(poly):
                total += float(fp.intersection(poly).area)
        # Rate is overlap area — the observer multiplies by dt to get dwell-area-time.
        return total, {"overlap_area_m2": total}
