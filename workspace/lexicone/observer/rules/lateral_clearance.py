"""3r5 — Maintain lateral clearance to adjacent agents."""

from __future__ import annotations

from typing import Any, Mapping, Tuple

from ..context import SceneContext, relative_velocity
from ..rule import ObserverRule


class LateralClearanceRule(ObserverRule):
    id = "3r5"
    level = 3
    name = "Maintain lateral clearance"
    description = (
        "Penalises lateral distance to adjacent agents below a dynamic safe "
        "minimum that grows with relative lateral velocity."
    )

    def __init__(self, min_lateral_m: float = 1.0, vrel_coef_s: float = 0.5):
        self.min_lateral_m = min_lateral_m
        self.vrel_coef_s = vrel_coef_s

    def applies(self, ctx: SceneContext) -> Tuple[bool, Mapping[str, Any]]:
        neighbors = ctx.lateral_neighbors(max_long_m=8.0, lateral_band_m=4.0)
        return bool(neighbors), {"n_neighbors": len(neighbors)}

    def violation(self, ctx: SceneContext) -> Tuple[float, Mapping[str, Any]]:
        worst = 0.0
        worst_track = None
        for n in ctx.lateral_neighbors(max_long_m=8.0, lateral_band_m=4.0):
            lat_gap = max(0.0, abs(n.lateral_offset_m) - (ctx.ego.width + n.agent.width) / 2.0)
            _, lat_rel = relative_velocity(ctx.ego, n.agent)
            v_close = max(0.0, -lat_rel if n.lateral_offset_m > 0 else lat_rel)
            d_safe = self.min_lateral_m + self.vrel_coef_s * v_close
            shortfall = max(0.0, d_safe - lat_gap)
            if shortfall > worst:
                worst = shortfall
                worst_track = n.agent.track_id
        return worst, {
            "worst_track_id": worst_track,
            "min_lateral_threshold_m": self.min_lateral_m,
        }
