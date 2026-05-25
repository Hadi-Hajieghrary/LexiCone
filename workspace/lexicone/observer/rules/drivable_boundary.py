"""7r0 — Stay within drivable surface boundaries (excluding intersection
crossings, which are handled by 9r1's safety-critical formulation)."""

from __future__ import annotations

from typing import Any, Mapping, Tuple

from ..context import SceneContext
from ..geometry import footprint_outside_drivable
from ..rule import ObserverRule


class DrivableBoundaryRule(ObserverRule):
    id = "7r0"
    level = 7
    name = "Stay within drivable surface boundaries"
    description = (
        "Penalises distance-time spent with any part of the ego footprint "
        "beyond the legal road edge / lane boundary (excluding intentional "
        "intersection traversal)."
    )

    def applies(self, ctx: SceneContext) -> Tuple[bool, Mapping[str, Any]]:
        polys = ctx.drivable_polygons
        in_int, _ = ctx.ego_in_intersection
        return bool(polys) and not in_int, {
            "n_drivable_polygons": len(polys),
            "in_intersection": in_int,
        }

    def violation(self, ctx: SceneContext) -> Tuple[float, Mapping[str, Any]]:
        outside_area = footprint_outside_drivable(ctx.ego_footprint, ctx.drivable_polygons)
        return outside_area * max(ctx.ego.speed, 0.0), {
            "outside_area_m2": outside_area,
            "ego_speed_mps": ctx.ego.speed,
        }
