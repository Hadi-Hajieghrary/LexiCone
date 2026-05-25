"""2r2 — Adhere to the planned global route."""

from __future__ import annotations

import math
from typing import Any, Mapping, Tuple

from ..context import SceneContext
from ..geometry import project_onto_polyline
from ..rule import ObserverRule


class RouteAdherenceRule(ObserverRule):
    id = "2r2"
    level = 2
    name = "Adhere to the planned global route"
    description = (
        "Penalises geometric drift outside the planned-route lane corridor "
        "and topological deviations (ego in a lane outside the route)."
    )

    def __init__(self, corridor_lateral_m: float = 2.0):
        self.corridor_lateral_m = corridor_lateral_m

    def applies(self, ctx: SceneContext) -> Tuple[bool, Mapping[str, Any]]:
        if ctx.snapshot.route_lane_ids is None or not ctx.snapshot.route_lane_ids:
            return False, {"reason": "no_route"}
        return True, {"n_route_lanes": len(ctx.snapshot.route_lane_ids)}

    def violation(self, ctx: SceneContext) -> Tuple[float, Mapping[str, Any]]:
        route = set(ctx.snapshot.route_lane_ids or [])
        ego_lane = ctx.ego_lane
        in_route = ego_lane is not None and ego_lane.lane_id in route
        if in_route:
            return 0.0, {"ego_lane_id": ego_lane.lane_id, "on_route": True}
        ec = ctx.ego_center
        best_lat = math.inf
        nearest_id = None
        for lane in ctx.all_lanes:
            if lane.lane_id not in route:
                continue
            if not lane.centerline or len(lane.centerline) < 2:
                continue
            _, lat, _ = project_onto_polyline(ec, lane.centerline)
            if abs(lat) < best_lat:
                best_lat = abs(lat)
                nearest_id = lane.lane_id
        drift = max(0.0, (best_lat if best_lat != math.inf else 5.0) - self.corridor_lateral_m)
        return drift, {
            "ego_lane_id": ego_lane.lane_id if ego_lane is not None else None,
            "on_route": False,
            "nearest_route_lane_id": nearest_id,
            "lateral_to_route_m": best_lat if best_lat != math.inf else None,
        }
