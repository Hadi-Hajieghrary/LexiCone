"""10r3 — Yield right-of-way at unmarked crosswalks (intersections without
marked crossings)."""

from __future__ import annotations

import math
from typing import Any, Mapping, Tuple

from ..context import SceneContext
from ..geometry import planar_distance, polygon_from_points
from ..rule import ObserverRule
from ..types import AgentType


class UnmarkedCrosswalkYieldRule(ObserverRule):
    id = "10r3"
    level = 10
    name = "Yield at unmarked crosswalks"
    description = (
        "At any intersection without a marked crosswalk, the ego must yield "
        "to pedestrians as if a crosswalk were present."
    )

    def __init__(self, near_m: float = 2.0, yield_speed_mps: float = 1.0):
        self.near_m = near_m
        self.yield_speed_mps = yield_speed_mps

    def applies(self, ctx: SceneContext) -> Tuple[bool, Mapping[str, Any]]:
        in_int, int_id = ctx.ego_in_intersection
        if not in_int:
            return False, {}
        # If a marked crosswalk exists in the intersection, 8r1 handles it.
        for cw in ctx.snapshot.map.crosswalks:
            if not cw.is_marked:
                continue
            poly = polygon_from_points(cw.polygon)
            if poly is None:
                continue
            for inter in ctx.snapshot.map.intersections:
                ipoly = polygon_from_points(inter.polygon)
                if ipoly is not None and ipoly.intersects(poly):
                    return False, {"reason": "marked_crosswalk_present"}
        peds = ctx.agents_by_type([AgentType.PEDESTRIAN], radius_m=15.0)
        return bool(peds), {"intersection_id": int_id, "n_peds_nearby": len(peds)}

    def violation(self, ctx: SceneContext) -> Tuple[float, Mapping[str, Any]]:
        ec = ctx.ego_center
        worst_dist = math.inf
        for a in ctx.agents_by_type([AgentType.PEDESTRIAN]):
            d = planar_distance(ec, (a.pose.x, a.pose.y))
            if d < worst_dist:
                worst_dist = d
        rate = max(0.0, ctx.ego.speed - self.yield_speed_mps) if ctx.ego.speed > self.yield_speed_mps else 0.0
        if worst_dist != math.inf and worst_dist < 3.0 and ctx.ego.speed > 0.2:
            rate += (3.0 - worst_dist) * ctx.ego.speed
        return rate, {
            "ego_speed_mps": ctx.ego.speed,
            "min_ped_distance_m": worst_dist if worst_dist != math.inf else None,
        }
