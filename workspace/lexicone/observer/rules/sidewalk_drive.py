"""7r5 — Do not drive on sidewalks or pedestrian areas."""

from __future__ import annotations

from typing import Any, Mapping, Tuple

from ..context import SceneContext
from ..rule import ObserverRule


class SidewalkDriveRule(ObserverRule):
    id = "7r5"
    level = 7
    name = "Do not drive on sidewalks or pedestrian areas"
    description = (
        "Penalises footprint overlap with walkway polygons weighted by speed."
    )

    def applies(self, ctx: SceneContext) -> Tuple[bool, Mapping[str, Any]]:
        n = len(ctx.snapshot.map.walkways)
        return n > 0, {"n_walkways": n}

    def violation(self, ctx: SceneContext) -> Tuple[float, Mapping[str, Any]]:
        overlap = ctx.walkway_overlap_m2
        return overlap * max(ctx.ego.speed, 1e-3), {
            "overlap_area_m2": overlap,
            "ego_speed_mps": ctx.ego.speed,
        }
