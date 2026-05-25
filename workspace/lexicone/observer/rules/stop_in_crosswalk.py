"""7r4 — Do not stop inside crosswalks."""

from __future__ import annotations

from typing import Any, Mapping, Tuple

from ..context import SceneContext
from ..geometry import polygon_from_points
from ..rule import ObserverRule


class StopInCrosswalkRule(ObserverRule):
    id = "7r4"
    level = 7
    name = "Do not stop inside crosswalks"
    description = (
        "Penalises dwell time with the ego footprint overlapping a crosswalk "
        "while the ego is essentially stopped."
    )

    def __init__(self, stop_speed_mps: float = 0.5):
        self.stop_speed_mps = stop_speed_mps

    def applies(self, ctx: SceneContext) -> Tuple[bool, Mapping[str, Any]]:
        if ctx.ego.speed > self.stop_speed_mps:
            return False, {"ego_speed_mps": ctx.ego.speed}
        fp = ctx.ego_footprint
        for cw in ctx.snapshot.map.crosswalks:
            poly = polygon_from_points(cw.polygon)
            if poly is not None and fp.intersects(poly):
                return True, {"crosswalk_id": cw.crosswalk_id, "ego_speed_mps": ctx.ego.speed}
        return False, {}

    def violation(self, ctx: SceneContext) -> Tuple[float, Mapping[str, Any]]:
        fp = ctx.ego_footprint
        total = 0.0
        for cw in ctx.snapshot.map.crosswalks:
            poly = polygon_from_points(cw.polygon)
            if poly is not None and fp.intersects(poly):
                total += float(fp.intersection(poly).area)
        return total, {"overlap_area_m2": total}
