"""3r3 — Maintain safe following headway (THW and TTC)."""

from __future__ import annotations

import math
from typing import Any, Mapping, Tuple

from ..context import SceneContext, relative_velocity
from ..rule import ObserverRule


class SafeHeadwayRule(ObserverRule):
    id = "3r3"
    level = 3
    name = "Maintain safe following headway"
    description = (
        "Penalises shortfalls in time headway (THW) and time-to-collision "
        "(TTC) below safe minimums when an in-lane lead vehicle is present."
    )

    def __init__(
        self,
        min_thw_s: float = 1.5,
        min_ttc_s: float = 3.0,
        min_distance_m: float = 2.0,
    ):
        self.min_thw_s = min_thw_s
        self.min_ttc_s = min_ttc_s
        self.min_distance_m = min_distance_m

    def applies(self, ctx: SceneContext) -> Tuple[bool, Mapping[str, Any]]:
        lead = ctx.lead_agent()
        if lead is None:
            return False, {}
        return True, {
            "lead_track_id": lead.agent.track_id,
            "longitudinal_distance_m": lead.longitudinal_distance_m,
        }

    def violation(self, ctx: SceneContext) -> Tuple[float, Mapping[str, Any]]:
        lead = ctx.lead_agent()
        if lead is None:
            return 0.0, {}
        ego = ctx.ego
        v = max(ego.speed, 1e-3)
        gap = max(0.0, lead.longitudinal_distance_m - (ego.length + lead.agent.length) / 2.0)
        thw = gap / v
        rel_lon, _ = relative_velocity(ego, lead.agent)
        closing = -rel_lon
        ttc = gap / closing if closing > 0.1 else math.inf
        thw_shortfall = max(0.0, self.min_thw_s - thw)
        ttc_shortfall = max(0.0, self.min_ttc_s - ttc) if ttc != math.inf else 0.0
        too_close = max(0.0, self.min_distance_m - gap) * 2.0
        rate = thw_shortfall + ttc_shortfall + too_close
        return rate, {
            "lead_track_id": lead.agent.track_id,
            "gap_m": gap,
            "thw_s": thw,
            "ttc_s": None if ttc == math.inf else ttc,
            "closing_speed_mps": closing,
        }
