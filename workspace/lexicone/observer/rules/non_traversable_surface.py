"""9r1 — Avoid driving into areas with no traversable surface."""

from __future__ import annotations

from typing import Any, Mapping, Tuple

from ..context import SceneContext
from ..geometry import footprint_outside_drivable
from ..rule import ObserverRule


class NonTraversableSurfaceRule(ObserverRule):
    id = "9r1"
    level = 9
    name = "Avoid non-traversable surface"
    description = (
        "Penalises any portion of the ego footprint that is outside the "
        "mapped drivable area (medians, gores, off-road)."
    )

    def applies(self, ctx: SceneContext) -> Tuple[bool, Mapping[str, Any]]:
        polys = ctx.drivable_polygons
        return bool(polys), {"n_drivable_polygons": len(polys)}

    def violation(self, ctx: SceneContext) -> Tuple[float, Mapping[str, Any]]:
        outside_area = footprint_outside_drivable(ctx.ego_footprint, ctx.drivable_polygons)
        return outside_area, {
            "outside_area_m2": outside_area,
            "ego_speed_mps": ctx.ego.speed,
        }
