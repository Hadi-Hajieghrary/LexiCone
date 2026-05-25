"""8r0 — Comply with mandatory stops (stop signs, red-before-turn)."""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional, Tuple

from ..context import SceneContext
from ..rule import ObserverRule
from ..types import StopType


class MandatoryStopRule(ObserverRule):
    id = "8r0"
    level = 8
    name = "Comply with mandatory stops"
    description = (
        "At a stop sign or red-light controlled stop line, the ego must come "
        "to a complete stop at/before the line. Penalises: (i) crossing while "
        "moving above stop tolerance, (ii) failing to stop within a short "
        "approach window."
    )

    def __init__(self, stop_speed_mps: float = 0.3, approach_dist_m: float = 8.0):
        self.stop_speed_mps = stop_speed_mps
        self.approach_dist_m = approach_dist_m
        self._approach_state: dict[str, float] = {}
        self._last_stop_id: Optional[str] = None

    def applies(self, ctx: SceneContext) -> Tuple[bool, Mapping[str, Any]]:
        ego_lane = ctx.ego_lane
        if ego_lane is None:
            return False, {}
        result = ctx.stop_polygon_for_ego(ego_lane)
        if result is None:
            return False, {}
        _, signed_dist = result
        stop_id = None
        controlling_type: Optional[str] = None
        for sl in ctx.snapshot.map.stop_lines:
            if sl.associated_lane_id == ego_lane.lane_id:
                stop_id = sl.stop_line_id
                controlling_type = (
                    sl.stop_type.value if isinstance(sl.stop_type, StopType) else sl.stop_type
                )
                break
        if controlling_type not in (StopType.STOP_SIGN.value, StopType.GENERIC.value):
            return False, {"stop_type": controlling_type}
        if abs(signed_dist) > self.approach_dist_m:
            return False, {"distance_to_stop_m": signed_dist}
        return True, {
            "stop_line_id": stop_id,
            "distance_to_stop_m": signed_dist,
            "stop_type": controlling_type,
        }

    def violation(self, ctx: SceneContext) -> Tuple[float, Mapping[str, Any]]:
        ego_lane = ctx.ego_lane
        if ego_lane is None:
            return 0.0, {}
        result = ctx.stop_polygon_for_ego(ego_lane)
        if result is None:
            return 0.0, {}
        poly, signed_dist = result
        v = ctx.ego.speed
        rate = 0.0
        details: dict[str, Any] = {"distance_to_stop_m": signed_dist, "ego_speed_mps": v}
        if ctx.ego_footprint.intersects(poly) and v > self.stop_speed_mps:
            rate = max(rate, v - self.stop_speed_mps)
            details["overrunning_stop_line"] = True
        stop_id = ego_lane.lane_id
        prev_min = self._approach_state.get(stop_id, math.inf)
        new_min = min(prev_min, v)
        self._approach_state[stop_id] = new_min
        if signed_dist < -1.0 and new_min > self.stop_speed_mps:
            rate = max(rate, new_min - self.stop_speed_mps)
            details["min_approach_speed_mps"] = new_min
            details["rolling_stop"] = True
        return rate, details
