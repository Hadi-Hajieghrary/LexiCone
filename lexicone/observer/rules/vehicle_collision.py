"""9r0 — Avoid collision with non-VRU vehicles or obstacles."""

from __future__ import annotations

from typing import Any, Mapping, Tuple

from ..geometry import agent_footprint, ego_center, ego_footprint, planar_distance
from ..rule import ObserverRule
from ..types import AgentType, SceneSnapshot
from ._common import VRU_TYPES


class VehicleCollisionRule(ObserverRule):
    id = "9r0"
    level = 9
    name = "Avoid collision with non-VRU vehicles or obstacles"
    description = (
        "Penalises overlap between ego footprint and any non-VRU object: "
        "vehicles, motorcycles, barriers, cones, generic objects."
    )

    def __init__(self, search_radius_m: float = 40.0):
        self.search_radius_m = search_radius_m

    def applies_at(self, snap: SceneSnapshot) -> Tuple[bool, Mapping[str, Any]]:
        ec = ego_center(snap.ego)
        nearby = 0
        for a in snap.agents:
            ot = a.object_type.value if isinstance(a.object_type, AgentType) else a.object_type
            if ot in VRU_TYPES:
                continue
            if planar_distance(ec, (a.pose.x, a.pose.y)) <= self.search_radius_m:
                nearby += 1
        return nearby > 0, {"n_nonvru_in_radius": nearby}

    def violation_at(self, snap: SceneSnapshot) -> Tuple[float, Mapping[str, Any]]:
        ego_fp = ego_footprint(snap.ego)
        total = 0.0
        worst_id = None
        worst = 0.0
        for a in snap.agents:
            ot = a.object_type.value if isinstance(a.object_type, AgentType) else a.object_type
            if ot in VRU_TYPES:
                continue
            af = agent_footprint(a)
            if ego_fp.intersects(af):
                ov = float(ego_fp.intersection(af).area)
                total += ov
                if ov > worst:
                    worst = ov
                    worst_id = a.track_id
        return total, {
            "overlap_area_m2": total,
            "worst_track_id": worst_id,
            "worst_overlap_m2": worst,
        }
