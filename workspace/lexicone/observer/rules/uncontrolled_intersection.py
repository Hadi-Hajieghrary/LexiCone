"""1r5 — Negotiate uncontrolled intersections safely."""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional, Tuple

from shapely.geometry import Point

from ..context import VEHICLE_LIKE_TYPES, SceneContext, _type_value
from ..geometry import is_same_direction, lane_polygon, planar_distance
from ..rule import ObserverRule


class UncontrolledIntersectionRule(ObserverRule):
    id = "1r5"
    level = 1
    name = "Negotiate uncontrolled intersections safely"
    description = (
        "At intersections without a TL or stop sign for the ego's lane, "
        "penalises advancing through when another road user has priority "
        "(e.g. yield-to-right, first-to-stop)."
    )

    def __init__(self, t_advance_margin_s: float = 1.0):
        self.t_advance_margin_s = t_advance_margin_s

    def _is_uncontrolled(self, ctx: SceneContext, int_id: Optional[str]) -> bool:
        if int_id is None:
            return False
        ego_lane = ctx.ego_lane
        if ego_lane is not None:
            for sl in ctx.snapshot.map.stop_lines:
                if sl.associated_lane_id == ego_lane.lane_id:
                    return False
        for lc in ctx.snapshot.map.lane_connectors:
            poly = lane_polygon(lc)
            if poly is None:
                continue
            if poly.contains(Point(*ctx.ego_center)):
                active, _ = ctx.red_or_yellow_for_lane(lc.lane_id)
                if active:
                    return False
        return True

    def applies(self, ctx: SceneContext) -> Tuple[bool, Mapping[str, Any]]:
        in_int, int_id = ctx.ego_in_intersection
        if not in_int:
            return False, {}
        if not self._is_uncontrolled(ctx, int_id):
            return False, {"reason": "controlled"}
        peers = 0
        ec = ctx.ego_center
        for a in ctx.snapshot.agents:
            if _type_value(a.object_type) not in VEHICLE_LIKE_TYPES:
                continue
            if planar_distance(ec, (a.pose.x, a.pose.y)) <= 20.0 and not is_same_direction(
                a.pose.heading, ctx.ego.pose.heading, math.radians(60)
            ):
                peers += 1
        return peers > 0, {"intersection_id": int_id, "n_cross_traffic": peers}

    def violation(self, ctx: SceneContext) -> Tuple[float, Mapping[str, Any]]:
        ec = ctx.ego_center
        cos_h = math.cos(ctx.ego.pose.heading)
        sin_h = math.sin(ctx.ego.pose.heading)
        worst = 0.0
        worst_track = None
        for a in ctx.snapshot.agents:
            if _type_value(a.object_type) not in VEHICLE_LIKE_TYPES:
                continue
            dx = a.pose.x - ec[0]
            dy = a.pose.y - ec[1]
            lat = -dx * sin_h + dy * cos_h
            lon = dx * cos_h + dy * sin_h
            if lat >= 0:
                continue
            if not (0 < lon < 20.0 and a.speed > 0.5):
                continue
            d_ego_to_conflict = max(lon - 2.0, 0.0)
            t_ego = d_ego_to_conflict / max(ctx.ego.speed, 0.5)
            d_a = math.hypot(dx, dy)
            t_a = d_a / max(a.speed, 0.5)
            if t_a + self.t_advance_margin_s <= t_ego:
                rate = max(0.0, ctx.ego.speed - 0.5)
                if rate > worst:
                    worst = rate
                    worst_track = a.track_id
        return worst, {"yielding_to_track_id": worst_track}
