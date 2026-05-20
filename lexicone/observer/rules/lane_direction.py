"""Lane-direction rules.

- 7r2 — Avoid opposing lane when oncoming traffic is present.
- 7r3 — Obey one-way directionality.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional, Tuple

from shapely.geometry import Point

from ..geometry import (
    ego_center,
    ego_footprint,
    heading_difference,
    is_opposite_direction,
    is_same_direction,
    lane_heading,
    lane_polygon,
)
from ..rule import ObserverRule
from ..types import AgentType, SceneSnapshot
from ._common import VEHICLE_LIKE_TYPES


def _lane_containing_ego(snap: SceneSnapshot):
    ec = ego_center(snap.ego)
    pt = Point(ec)
    candidates = list(snap.map.lanes) + list(snap.map.lane_connectors)
    for lane in candidates:
        poly = lane_polygon(lane)
        if poly is not None and poly.contains(pt):
            yield lane


class OpposingLaneRule(ObserverRule):
    id = "7r2"
    level = 7
    name = "Avoid opposing lane with oncoming traffic"
    description = (
        "Penalises ego footprint area overlapping a lane whose direction is "
        "opposite to ego, scaled by speed and by the presence of oncoming "
        "vehicles in that lane."
    )

    def __init__(self, opposing_search_m: float = 60.0):
        self.opposing_search_m = opposing_search_m

    def applies_at(self, snap: SceneSnapshot) -> Tuple[bool, Mapping[str, Any]]:
        fp = ego_footprint(snap.ego)
        for lane in list(snap.map.lanes) + list(snap.map.lane_connectors):
            if not is_opposite_direction(lane_heading(lane), snap.ego.pose.heading):
                continue
            poly = lane_polygon(lane)
            if poly is not None and fp.intersects(poly):
                return True, {"opposing_lane_id": lane.lane_id}
        return False, {}

    def violation_at(self, snap: SceneSnapshot) -> Tuple[float, Mapping[str, Any]]:
        fp = ego_footprint(snap.ego)
        overlap_total = 0.0
        oncoming_present = False
        ec = ego_center(snap.ego)
        for lane in list(snap.map.lanes) + list(snap.map.lane_connectors):
            if not is_opposite_direction(lane_heading(lane), snap.ego.pose.heading):
                continue
            poly = lane_polygon(lane)
            if poly is None or not fp.intersects(poly):
                continue
            overlap_total += float(fp.intersection(poly).area)
            for a in snap.agents:
                ot = a.object_type.value if isinstance(a.object_type, AgentType) else a.object_type
                if ot not in VEHICLE_LIKE_TYPES:
                    continue
                if not is_opposite_direction(a.pose.heading, snap.ego.pose.heading):
                    continue
                if poly.contains(Point(a.pose.x, a.pose.y)):
                    # Within search corridor
                    dx = a.pose.x - ec[0]
                    dy = a.pose.y - ec[1]
                    cos_h = math.cos(snap.ego.pose.heading)
                    sin_h = math.sin(snap.ego.pose.heading)
                    lon = dx * cos_h + dy * sin_h
                    if 0 <= lon <= self.opposing_search_m:
                        oncoming_present = True
                        break
        # Higher rate when oncoming traffic is actually present.
        weight = 1.0 + (4.0 if oncoming_present else 0.0)
        rate = overlap_total * max(snap.ego.speed, 1e-3) * weight
        return rate, {
            "overlap_area_m2": overlap_total,
            "oncoming_present": oncoming_present,
            "ego_speed_mps": snap.ego.speed,
        }


class OneWayDirectionRule(ObserverRule):
    id = "7r3"
    level = 7
    name = "Obey one-way street directionality"
    description = (
        "Penalises forward travel on a lane whose direction opposes the ego "
        "heading (wrong-way driving)."
    )

    def applies_at(self, snap: SceneSnapshot) -> Tuple[bool, Mapping[str, Any]]:
        for lane in _lane_containing_ego(snap):
            if is_opposite_direction(lane_heading(lane), snap.ego.pose.heading):
                return True, {"lane_id": lane.lane_id}
            if not is_same_direction(lane_heading(lane), snap.ego.pose.heading, math.radians(60)):
                return True, {"lane_id": lane.lane_id, "reason": "lateral_heading"}
        return False, {}

    def violation_at(self, snap: SceneSnapshot) -> Tuple[float, Mapping[str, Any]]:
        v = snap.ego.speed
        if v < 0.3:
            return 0.0, {"ego_speed_mps": v, "reason": "stationary"}
        for lane in _lane_containing_ego(snap):
            dh = abs(heading_difference(lane_heading(lane), snap.ego.pose.heading))
            if dh > math.radians(90):
                rate = v * (dh - math.radians(90)) / math.pi
                return rate, {
                    "lane_id": lane.lane_id,
                    "heading_difference_rad": dh,
                    "ego_speed_mps": v,
                }
        return 0.0, {"ego_speed_mps": v}
