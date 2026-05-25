"""7r3 — Obey one-way street directionality (no wrong-way driving)."""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Tuple

from shapely.geometry import Point

from ..context import SceneContext
from ..geometry import (
    heading_difference,
    is_opposite_direction,
    is_same_direction,
    lane_heading,
    lane_polygon,
)
from ..rule import ObserverRule
from ..types import LaneSnapshot


def _lanes_containing_ego(ctx: SceneContext) -> Iterable[LaneSnapshot]:
    pt = Point(ctx.ego_center)
    for lane in ctx.all_lanes:
        poly = lane_polygon(lane)
        if poly is not None and poly.contains(pt):
            yield lane


class OneWayDirectionRule(ObserverRule):
    id = "7r3"
    level = 7
    name = "Obey one-way street directionality"
    description = (
        "Penalises forward travel on a lane whose direction opposes the ego "
        "heading (wrong-way driving)."
    )

    def applies(self, ctx: SceneContext) -> Tuple[bool, Mapping[str, Any]]:
        for lane in _lanes_containing_ego(ctx):
            if is_opposite_direction(lane_heading(lane), ctx.ego.pose.heading):
                return True, {"lane_id": lane.lane_id}
            if not is_same_direction(lane_heading(lane), ctx.ego.pose.heading, math.radians(60)):
                return True, {"lane_id": lane.lane_id, "reason": "lateral_heading"}
        return False, {}

    def violation(self, ctx: SceneContext) -> Tuple[float, Mapping[str, Any]]:
        v = ctx.ego.speed
        if v < 0.3:
            return 0.0, {"ego_speed_mps": v, "reason": "stationary"}
        for lane in _lanes_containing_ego(ctx):
            dh = abs(heading_difference(lane_heading(lane), ctx.ego.pose.heading))
            if dh > math.radians(90):
                rate = v * (dh - math.radians(90)) / math.pi
                return rate, {
                    "lane_id": lane.lane_id,
                    "heading_difference_rad": dh,
                    "ego_speed_mps": v,
                }
        return 0.0, {"ego_speed_mps": v}
