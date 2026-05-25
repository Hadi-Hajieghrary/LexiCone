"""7r1 — Obey traffic-light states inside intersections."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Tuple

from ..context import SceneContext
from ..geometry import polygon_from_points
from ..rule import ObserverRule
from ..types import LaneSnapshot, TrafficLightState


class TrafficLightComplianceRule(ObserverRule):
    id = "7r1"
    level = 7
    name = "Obey traffic-light states in intersections"
    description = (
        "Counts time inside (or entering) an intersection while the "
        "associated lane-connector traffic light is RED (or YELLOW with "
        "safe stopping feasible)."
    )

    def __init__(self, safe_stop_decel_mps2: float = 3.0):
        self.safe_stop_decel_mps2 = safe_stop_decel_mps2

    def _relevant_connector(self, ctx: SceneContext) -> Optional[LaneSnapshot]:
        fp = ctx.ego_footprint
        for lc in ctx.snapshot.map.lane_connectors:
            poly = polygon_from_points(lc.polygon)
            if poly is not None and fp.intersects(poly):
                return lc
        return None

    def applies(self, ctx: SceneContext) -> Tuple[bool, Mapping[str, Any]]:
        lc = self._relevant_connector(ctx)
        if lc is None:
            return False, {}
        active, state = ctx.red_or_yellow_for_lane(lc.lane_id)
        if not active:
            return False, {"lc_id": lc.lane_id, "state": state}
        return True, {"lc_id": lc.lane_id, "state": state}

    def violation(self, ctx: SceneContext) -> Tuple[float, Mapping[str, Any]]:
        lc = self._relevant_connector(ctx)
        if lc is None:
            return 0.0, {}
        _, state = ctx.red_or_yellow_for_lane(lc.lane_id)
        in_int, _ = ctx.ego_in_intersection
        v = ctx.ego.speed
        rate = 0.0
        details: dict[str, Any] = {"state": state, "in_intersection": in_int, "ego_speed_mps": v}
        if state == TrafficLightState.RED.value and in_int and v > 0.3:
            rate = v
            details["red_in_intersection"] = True
        elif state == TrafficLightState.YELLOW.value and not in_int and v > 0.3:
            sl = ctx.stop_polygon_for_ego()
            if sl is not None:
                _, d = sl
                if d > 0:
                    stop_dist = (v * v) / (2.0 * self.safe_stop_decel_mps2)
                    if stop_dist <= d:
                        rate = v - 0.3
                        details["yellow_runnable"] = True
        return max(0.0, rate), details
