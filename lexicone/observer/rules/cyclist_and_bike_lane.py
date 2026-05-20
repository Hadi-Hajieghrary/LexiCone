"""Cyclist-related rules.

- 10r4 — Provide safe passing distance for cyclists.
- 10r5 — Do not encroach into designated bicycle lanes.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Tuple

from ..geometry import ego_center, ego_footprint, lane_polygon, planar_distance
from ..rule import ObserverRule
from ..types import AgentType, SceneSnapshot
from ._common import ego_overlaps_bike_lane


def _ego_lat_relative(snap: SceneSnapshot, x: float, y: float) -> Tuple[float, float]:
    """Return (longitudinal, lateral) offset of (x, y) in the ego frame."""
    ec = ego_center(snap.ego)
    cos_h = math.cos(snap.ego.pose.heading)
    sin_h = math.sin(snap.ego.pose.heading)
    dx = x - ec[0]
    dy = y - ec[1]
    return dx * cos_h + dy * sin_h, -dx * sin_h + dy * cos_h


class CyclistPassingRule(ObserverRule):
    id = "10r4"
    level = 10
    name = "Safe passing distance for cyclists"
    description = (
        "Penalises overtaking a cyclist with lateral clearance below a "
        "minimum, scaled by ego speed."
    )

    def __init__(self, min_lateral_m: float = 1.5, search_long_m: float = 8.0):
        self.min_lateral_m = min_lateral_m
        self.search_long_m = search_long_m

    def applies_at(self, snap: SceneSnapshot) -> Tuple[bool, Mapping[str, Any]]:
        for a in snap.agents:
            ot = a.object_type.value if isinstance(a.object_type, AgentType) else a.object_type
            if ot != AgentType.BICYCLE.value:
                continue
            lon, lat = _ego_lat_relative(snap, a.pose.x, a.pose.y)
            if abs(lon) <= self.search_long_m and abs(lat) <= 6.0 and snap.ego.speed > a.speed + 0.5:
                return True, {"cyclist_track_id": a.track_id, "ego_long_offset_m": lon, "lateral_m": lat}
        return False, {}

    def violation_at(self, snap: SceneSnapshot) -> Tuple[float, Mapping[str, Any]]:
        worst = 0.0
        worst_id = None
        for a in snap.agents:
            ot = a.object_type.value if isinstance(a.object_type, AgentType) else a.object_type
            if ot != AgentType.BICYCLE.value:
                continue
            lon, lat = _ego_lat_relative(snap, a.pose.x, a.pose.y)
            if abs(lon) > self.search_long_m or abs(lat) > 6.0 or snap.ego.speed <= a.speed + 0.5:
                continue
            shortfall = max(0.0, self.min_lateral_m - abs(lat))
            if shortfall > 0:
                # Severity scales with shortfall and ego speed.
                rate = shortfall * max(snap.ego.speed, 1.0)
                if rate > worst:
                    worst = rate
                    worst_id = a.track_id
        return worst, {"worst_track_id": worst_id, "min_lateral_threshold_m": self.min_lateral_m}


class BikeLaneEncroachmentRule(ObserverRule):
    id = "10r5"
    level = 10
    name = "Do not encroach into bicycle lanes"
    description = (
        "Penalises footprint overlap with any designated bicycle lane polygon."
    )

    def applies_at(self, snap: SceneSnapshot) -> Tuple[bool, Mapping[str, Any]]:
        n = len(snap.map.bike_lanes)
        return n > 0, {"n_bike_lanes": n}

    def violation_at(self, snap: SceneSnapshot) -> Tuple[float, Mapping[str, Any]]:
        _, overlap = ego_overlaps_bike_lane(snap)
        # Overlap area weighted by speed so brief encroachment at low speed is
        # less penalised than sustained drift at high speed.
        return overlap * max(snap.ego.speed, 1e-3), {"overlap_area_m2": overlap}
