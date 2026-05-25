"""7r2 — Avoid opposing lane when oncoming traffic is present."""

from __future__ import annotations

import math
from typing import Any, Mapping, Tuple

from shapely.geometry import Point

from ..context import VEHICLE_LIKE_TYPES, SceneContext, _type_value
from ..geometry import is_opposite_direction, lane_heading, lane_polygon
from ..rule import ObserverRule


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

    def applies(self, ctx: SceneContext) -> Tuple[bool, Mapping[str, Any]]:
        fp = ctx.ego_footprint
        for lane in ctx.all_lanes:
            if not is_opposite_direction(lane_heading(lane), ctx.ego.pose.heading):
                continue
            poly = lane_polygon(lane)
            if poly is not None and fp.intersects(poly):
                return True, {"opposing_lane_id": lane.lane_id}
        return False, {}

    def violation(self, ctx: SceneContext) -> Tuple[float, Mapping[str, Any]]:
        fp = ctx.ego_footprint
        ec = ctx.ego_center
        cos_h = math.cos(ctx.ego.pose.heading)
        sin_h = math.sin(ctx.ego.pose.heading)
        overlap_total = 0.0
        oncoming_present = False
        for lane in ctx.all_lanes:
            if not is_opposite_direction(lane_heading(lane), ctx.ego.pose.heading):
                continue
            poly = lane_polygon(lane)
            if poly is None or not fp.intersects(poly):
                continue
            overlap_total += float(fp.intersection(poly).area)
            for a in ctx.snapshot.agents:
                if _type_value(a.object_type) not in VEHICLE_LIKE_TYPES:
                    continue
                if not is_opposite_direction(a.pose.heading, ctx.ego.pose.heading):
                    continue
                if poly.contains(Point(a.pose.x, a.pose.y)):
                    dx = a.pose.x - ec[0]
                    dy = a.pose.y - ec[1]
                    lon = dx * cos_h + dy * sin_h
                    if 0 <= lon <= self.opposing_search_m:
                        oncoming_present = True
                        break
        weight = 1.0 + (4.0 if oncoming_present else 0.0)
        rate = overlap_total * max(ctx.ego.speed, 1e-3) * weight
        return rate, {
            "overlap_area_m2": overlap_total,
            "oncoming_present": oncoming_present,
            "ego_speed_mps": ctx.ego.speed,
        }
