"""10r0 — Avoid collision with VRUs."""

from __future__ import annotations

from typing import Any, Mapping, Tuple

from ..context import VRU_TYPES, SceneContext, _type_value
from ..geometry import agent_footprint, planar_distance
from ..rule import ObserverRule


class VRUCollisionRule(ObserverRule):
    id = "10r0"
    level = 10
    name = "Avoid collision with VRUs"
    description = (
        "Penalises any spatial overlap between the ego footprint and the "
        "(inflated) footprint of any pedestrian or cyclist."
    )

    def __init__(self, vru_inflate_m: float = 0.10, search_radius_m: float = 30.0):
        self.vru_inflate_m = vru_inflate_m
        self.search_radius_m = search_radius_m

    def applies(self, ctx: SceneContext) -> Tuple[bool, Mapping[str, Any]]:
        ec = ctx.ego_center
        nearby = 0
        for a in ctx.snapshot.agents:
            if _type_value(a.object_type) not in VRU_TYPES:
                continue
            if planar_distance(ec, (a.pose.x, a.pose.y)) <= self.search_radius_m:
                nearby += 1
        return nearby > 0, {"n_vrus_in_radius": nearby, "radius_m": self.search_radius_m}

    def violation(self, ctx: SceneContext) -> Tuple[float, Mapping[str, Any]]:
        ego_fp = ctx.ego_footprint
        total_overlap = 0.0
        worst_id = None
        worst_overlap = 0.0
        for a in ctx.snapshot.agents:
            if _type_value(a.object_type) not in VRU_TYPES:
                continue
            af = agent_footprint(a)
            if self.vru_inflate_m > 0:
                af = af.buffer(self.vru_inflate_m)
            if ego_fp.intersects(af):
                ov = float(ego_fp.intersection(af).area)
                total_overlap += ov
                if ov > worst_overlap:
                    worst_overlap = ov
                    worst_id = a.track_id
        return total_overlap, {
            "overlap_area_m2": total_overlap,
            "worst_track_id": worst_id,
            "worst_overlap_m2": worst_overlap,
        }
