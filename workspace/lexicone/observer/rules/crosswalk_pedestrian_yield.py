"""8r1 — Yield right-of-way to pedestrians at marked crosswalks."""

from __future__ import annotations

import math
from typing import Any, Iterable, List, Mapping, Tuple

from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry

from ..context import VRU_TYPES, SceneContext, _type_value
from ..geometry import planar_distance, polygon_from_points
from ..rule import ObserverRule
from ..types import AgentSnapshot, CrosswalkSnapshot


def _conflicting_crosswalks(ctx: SceneContext, marked: bool) -> Iterable[Tuple[CrosswalkSnapshot, BaseGeometry]]:
    """Yield (crosswalk, polygon) pairs intersecting a forward-projected ego sweep."""
    fp = ctx.ego_footprint
    cos_h = math.cos(ctx.ego.pose.heading)
    sin_h = math.sin(ctx.ego.pose.heading)
    forward_d = max(ctx.ego.speed, 5.0) * 2.0
    ec = ctx.ego_center
    forward_point = Point(ec[0] + cos_h * forward_d, ec[1] + sin_h * forward_d)
    swept = fp.union(forward_point.buffer(ctx.ego.width))
    for cw in ctx.snapshot.map.crosswalks:
        if cw.is_marked != marked:
            continue
        poly = polygon_from_points(cw.polygon)
        if poly is None:
            continue
        if swept.intersects(poly):
            yield cw, poly


def _pedestrians_on_or_near(polygon: BaseGeometry, ctx: SceneContext, near_m: float) -> List[AgentSnapshot]:
    inflated = polygon.buffer(near_m)
    out: List[AgentSnapshot] = []
    for a in ctx.snapshot.agents:
        if _type_value(a.object_type) not in VRU_TYPES:
            continue
        if inflated.contains(Point(a.pose.x, a.pose.y)):
            out.append(a)
    return out


class CrosswalkPedestrianYieldRule(ObserverRule):
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

    def applies(self, ctx: SceneContext) -> Tuple[bool, Mapping[str, Any]]:
        for _, poly in _conflicting_crosswalks(ctx, marked=True):
            peds = _pedestrians_on_or_near(poly, ctx, self.near_m)
            if peds:
                return True, {"n_peds": len(peds)}
        return False, {}

    def violation(self, ctx: SceneContext) -> Tuple[float, Mapping[str, Any]]:
        worst_speed = 0.0
        worst_min_dist = math.inf
        for _, poly in _conflicting_crosswalks(ctx, marked=True):
            peds = _pedestrians_on_or_near(poly, ctx, self.near_m)
            if not peds:
                continue
            ec = ctx.ego_center
            for p in peds:
                d = planar_distance(ec, (p.pose.x, p.pose.y))
                if d < worst_min_dist:
                    worst_min_dist = d
            if ctx.ego.speed > self.yield_speed_mps:
                worst_speed = max(worst_speed, ctx.ego.speed)
        rate = max(0.0, worst_speed - self.yield_speed_mps)
        return rate, {
            "ego_speed_mps": ctx.ego.speed,
            "yield_threshold_mps": self.yield_speed_mps,
            "min_ped_distance_m": (worst_min_dist if worst_min_dist != math.inf else None),
        }
