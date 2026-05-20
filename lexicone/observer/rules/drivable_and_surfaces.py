"""Drivable-surface and pedestrian-surface rules.

- 9r1 — Avoid driving into areas with no traversable surface.
- 7r0 — Stay within drivable surface boundaries.
- 7r5 — Do not drive on sidewalks or pedestrian areas.

9r1 and 7r0 share an underlying detector (ego footprint outside drivable
polygons); 9r1 is the safety-critical formulation that always applies whenever
a drivable-area layer is present, while 7r0 is the lighter operational form
that excludes intersection crossings (lane connectors).
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Tuple

from ..geometry import ego_footprint, footprint_outside_drivable, polygon_from_points
from ..rule import ObserverRule
from ..types import SceneSnapshot
from ._common import ego_in_intersection, ego_overlaps_walkway


def _drivable_polygons(snap: SceneSnapshot):
    polys = []
    for da in snap.map.drivable_area:
        p = polygon_from_points(da.polygon)
        if p is not None:
            polys.append(p)
    return polys


class NonTraversableSurfaceRule(ObserverRule):
    id = "9r1"
    level = 9
    name = "Avoid non-traversable surface"
    description = (
        "Penalises any portion of the ego footprint that is outside the mapped "
        "drivable area (medians, gores, off-road)."
    )

    def applies_at(self, snap: SceneSnapshot) -> Tuple[bool, Mapping[str, Any]]:
        polys = _drivable_polygons(snap)
        return bool(polys), {"n_drivable_polygons": len(polys)}

    def violation_at(self, snap: SceneSnapshot) -> Tuple[float, Mapping[str, Any]]:
        polys = _drivable_polygons(snap)
        fp = ego_footprint(snap.ego)
        outside_area = footprint_outside_drivable(fp, polys)
        return outside_area, {"outside_area_m2": outside_area, "ego_speed_mps": snap.ego.speed}


class DrivableBoundaryRule(ObserverRule):
    id = "7r0"
    level = 7
    name = "Stay within drivable surface boundaries"
    description = (
        "Penalises distance-time spent with any part of the ego footprint "
        "beyond the legal road edge / lane boundary (excluding intentional "
        "intersection traversal)."
    )

    def applies_at(self, snap: SceneSnapshot) -> Tuple[bool, Mapping[str, Any]]:
        polys = _drivable_polygons(snap)
        in_int, _ = ego_in_intersection(snap)
        return bool(polys) and not in_int, {
            "n_drivable_polygons": len(polys),
            "in_intersection": in_int,
        }

    def violation_at(self, snap: SceneSnapshot) -> Tuple[float, Mapping[str, Any]]:
        polys = _drivable_polygons(snap)
        fp = ego_footprint(snap.ego)
        outside_area = footprint_outside_drivable(fp, polys)
        # Weight by current speed to express "distance-time" beyond boundary.
        return outside_area * max(snap.ego.speed, 0.0), {
            "outside_area_m2": outside_area,
            "ego_speed_mps": snap.ego.speed,
        }


class SidewalkDriveRule(ObserverRule):
    id = "7r5"
    level = 7
    name = "Do not drive on sidewalks or pedestrian areas"
    description = (
        "Penalises footprint overlap with walkway polygons weighted by speed."
    )

    def applies_at(self, snap: SceneSnapshot) -> Tuple[bool, Mapping[str, Any]]:
        n = len(snap.map.walkways)
        return n > 0, {"n_walkways": n}

    def violation_at(self, snap: SceneSnapshot) -> Tuple[float, Mapping[str, Any]]:
        _, overlap = ego_overlaps_walkway(snap)
        return overlap * max(snap.ego.speed, 1e-3), {
            "overlap_area_m2": overlap,
            "ego_speed_mps": snap.ego.speed,
        }
