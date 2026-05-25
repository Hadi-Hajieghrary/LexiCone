"""1r2 — Don't block the box (intersection)."""

from __future__ import annotations

import math
from typing import Any, Mapping, Tuple

from ..context import VEHICLE_LIKE_TYPES, SceneContext, _type_value
from ..geometry import is_same_direction
from ..rule import ObserverRule


class BlockTheBoxRule(ObserverRule):
    id = "1r2"
    level = 1
    name = "Don't block the box"
    description = (
        "Penalises being essentially stopped inside an intersection without "
        "a clear downstream gap to exit."
    )

    def __init__(self, stop_speed_mps: float = 0.5, gap_lookahead_m: float = 10.0):
        self.stop_speed_mps = stop_speed_mps
        self.gap_lookahead_m = gap_lookahead_m

    def applies(self, ctx: SceneContext) -> Tuple[bool, Mapping[str, Any]]:
        in_int, int_id = ctx.ego_in_intersection
        return in_int and ctx.ego.speed <= self.stop_speed_mps, {
            "intersection_id": int_id,
            "ego_speed_mps": ctx.ego.speed,
        }

    def violation(self, ctx: SceneContext) -> Tuple[float, Mapping[str, Any]]:
        ec = ctx.ego_center
        cos_h = math.cos(ctx.ego.pose.heading)
        sin_h = math.sin(ctx.ego.pose.heading)
        blocked = False
        blocking_track = None
        for a in ctx.snapshot.agents:
            if _type_value(a.object_type) not in VEHICLE_LIKE_TYPES:
                continue
            dx = a.pose.x - ec[0]
            dy = a.pose.y - ec[1]
            lon = dx * cos_h + dy * sin_h
            lat = -dx * sin_h + dy * cos_h
            if 0 < lon <= self.gap_lookahead_m and abs(lat) <= 2.0 and is_same_direction(
                a.pose.heading, ctx.ego.pose.heading
            ) and a.speed < 0.5:
                blocked = True
                blocking_track = a.track_id
                break
        rate = 1.0 if blocked else 0.0
        return rate, {"downstream_blocked": blocked, "blocking_track_id": blocking_track}
