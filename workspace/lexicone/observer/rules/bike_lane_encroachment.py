"""10r5 — Do not encroach into designated bicycle lanes."""

from __future__ import annotations

from typing import Any, Mapping, Tuple

from ..context import SceneContext
from ..rule import ObserverRule


class BikeLaneEncroachmentRule(ObserverRule):
    id = "10r5"
    level = 10
    name = "Do not encroach into bicycle lanes"
    description = (
        "Penalises footprint overlap with any designated bicycle lane polygon."
    )

    def applies(self, ctx: SceneContext) -> Tuple[bool, Mapping[str, Any]]:
        n = len(ctx.snapshot.map.bike_lanes)
        return n > 0, {"n_bike_lanes": n}

    def violation(self, ctx: SceneContext) -> Tuple[float, Mapping[str, Any]]:
        overlap = ctx.bike_lane_overlap_m2
        return overlap * max(ctx.ego.speed, 1e-3), {"overlap_area_m2": overlap}
