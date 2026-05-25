"""10r4 — Provide safe passing distance for cyclists."""

from __future__ import annotations

import math
from typing import Any, Mapping, Tuple

from ..context import SceneContext, _type_value
from ..rule import ObserverRule
from ..types import AgentType


def _ego_frame_offset(ctx: SceneContext, x: float, y: float) -> Tuple[float, float]:
    ec = ctx.ego_center
    cos_h = math.cos(ctx.ego.pose.heading)
    sin_h = math.sin(ctx.ego.pose.heading)
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

    def applies(self, ctx: SceneContext) -> Tuple[bool, Mapping[str, Any]]:
        for a in ctx.snapshot.agents:
            if _type_value(a.object_type) != AgentType.BICYCLE.value:
                continue
            lon, lat = _ego_frame_offset(ctx, a.pose.x, a.pose.y)
            if abs(lon) <= self.search_long_m and abs(lat) <= 6.0 and ctx.ego.speed > a.speed + 0.5:
                return True, {
                    "cyclist_track_id": a.track_id,
                    "ego_long_offset_m": lon,
                    "lateral_m": lat,
                }
        return False, {}

    def violation(self, ctx: SceneContext) -> Tuple[float, Mapping[str, Any]]:
        worst = 0.0
        worst_id = None
        for a in ctx.snapshot.agents:
            if _type_value(a.object_type) != AgentType.BICYCLE.value:
                continue
            lon, lat = _ego_frame_offset(ctx, a.pose.x, a.pose.y)
            if abs(lon) > self.search_long_m or abs(lat) > 6.0 or ctx.ego.speed <= a.speed + 0.5:
                continue
            shortfall = max(0.0, self.min_lateral_m - abs(lat))
            if shortfall > 0:
                rate = shortfall * max(ctx.ego.speed, 1.0)
                if rate > worst:
                    worst = rate
                    worst_id = a.track_id
        return worst, {"worst_track_id": worst_id, "min_lateral_threshold_m": self.min_lateral_m}
