"""3r6 — Manage lane intrusions from adjacent vehicles."""

from __future__ import annotations

from typing import Any, Mapping, Tuple

from ..context import SceneContext, relative_velocity
from ..rule import ObserverRule


class LaneIntrusionRule(ObserverRule):
    id = "3r6"
    level = 3
    name = "Manage lane intrusions from adjacent vehicles"
    description = (
        "Penalises low lateral time-to-collision with adjacent vehicles, "
        "encouraging early gap creation."
    )

    def __init__(self, min_lat_ttc_s: float = 2.0):
        self.min_lat_ttc_s = min_lat_ttc_s

    def applies(self, ctx: SceneContext) -> Tuple[bool, Mapping[str, Any]]:
        for n in ctx.lateral_neighbors(max_long_m=10.0, lateral_band_m=5.0):
            _, lat_rel = relative_velocity(ctx.ego, n.agent)
            v_close = -lat_rel if n.lateral_offset_m > 0 else lat_rel
            if v_close > 0.2:
                return True, {"intruding_track_id": n.agent.track_id}
        return False, {}

    def violation(self, ctx: SceneContext) -> Tuple[float, Mapping[str, Any]]:
        worst = 0.0
        worst_track = None
        for n in ctx.lateral_neighbors(max_long_m=10.0, lateral_band_m=5.0):
            lat_gap = max(0.0, abs(n.lateral_offset_m) - (ctx.ego.width + n.agent.width) / 2.0)
            _, lat_rel = relative_velocity(ctx.ego, n.agent)
            v_close = -lat_rel if n.lateral_offset_m > 0 else lat_rel
            if v_close <= 0.05:
                continue
            ttc = lat_gap / v_close
            shortfall = max(0.0, self.min_lat_ttc_s - ttc)
            if shortfall > worst:
                worst = shortfall
                worst_track = n.agent.track_id
        return worst, {"worst_track_id": worst_track, "min_lat_ttc_s": self.min_lat_ttc_s}
