"""Geometry helpers for the observer.

All polygon/footprint operations go through Shapely. Footprints are oriented
oriented bounding boxes in world coordinates.
"""

from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence, Tuple

from shapely.affinity import rotate, translate
from shapely.geometry import LineString, Point, Polygon
from shapely.geometry.base import BaseGeometry

from .types import (
    AgentSnapshot,
    CrosswalkSnapshot,
    EgoSnapshot,
    IntersectionSnapshot,
    LaneSnapshot,
    Pose2D,
    WalkwaySnapshot,
)


def _box_polygon(length: float, width: float) -> Polygon:
    """Axis-aligned box centered at origin: [-L/2, +L/2] x [-W/2, +W/2]."""
    L2, W2 = length / 2.0, width / 2.0
    return Polygon([(-L2, -W2), (L2, -W2), (L2, W2), (-L2, W2)])


def _oriented_box(center: Tuple[float, float], heading: float, length: float, width: float) -> Polygon:
    box = _box_polygon(length, width)
    rotated = rotate(box, math.degrees(heading), origin=(0.0, 0.0))
    return translate(rotated, center[0], center[1])


def ego_footprint(ego: EgoSnapshot) -> Polygon:
    """Oriented bounding box for the ego in world frame.

    Accounts for the ego pose being expressed at the rear axle by default.
    """
    cx, cy = ego.pose.x, ego.pose.y
    if not ego.pose_at_center and ego.rear_axle_to_center:
        cx += ego.rear_axle_to_center * math.cos(ego.pose.heading)
        cy += ego.rear_axle_to_center * math.sin(ego.pose.heading)
    return _oriented_box((cx, cy), ego.pose.heading, ego.length, ego.width)


def ego_center(ego: EgoSnapshot) -> Tuple[float, float]:
    """World-frame geometric center of the ego footprint."""
    if ego.pose_at_center or not ego.rear_axle_to_center:
        return (ego.pose.x, ego.pose.y)
    cx = ego.pose.x + ego.rear_axle_to_center * math.cos(ego.pose.heading)
    cy = ego.pose.y + ego.rear_axle_to_center * math.sin(ego.pose.heading)
    return (cx, cy)


def agent_footprint(agent: AgentSnapshot) -> Polygon:
    return _oriented_box((agent.pose.x, agent.pose.y), agent.pose.heading, agent.length, agent.width)


def polygon_from_points(points: Sequence[Tuple[float, float]]) -> Optional[Polygon]:
    if points is None or len(points) < 3:
        return None
    poly = Polygon(points)
    if not poly.is_valid:
        poly = poly.buffer(0.0)
    if poly.is_empty:
        return None
    return poly


def linestring_from_points(points: Sequence[Tuple[float, float]]) -> Optional[LineString]:
    if points is None or len(points) < 2:
        return None
    return LineString(points)


def planar_distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def signed_lateral_offset(ref_a: Tuple[float, float], ref_b: Tuple[float, float], point: Tuple[float, float]) -> float:
    """Signed lateral offset of ``point`` from the directed line ref_a→ref_b.

    Positive = left of the direction of travel; negative = right.
    """
    dx, dy = ref_b[0] - ref_a[0], ref_b[1] - ref_a[1]
    norm = math.hypot(dx, dy)
    if norm < 1e-9:
        return planar_distance(ref_a, point)
    nx, ny = -dy / norm, dx / norm
    return (point[0] - ref_a[0]) * nx + (point[1] - ref_a[1]) * ny


def heading_difference(a: float, b: float) -> float:
    """Signed minimal difference (a - b) wrapped to (-pi, pi]."""
    d = (a - b) % (2 * math.pi)
    if d > math.pi:
        d -= 2 * math.pi
    return d


def is_same_direction(h1: float, h2: float, tol_rad: float = math.radians(45.0)) -> bool:
    return abs(heading_difference(h1, h2)) <= tol_rad


def is_opposite_direction(h1: float, h2: float, tol_rad: float = math.radians(45.0)) -> bool:
    return abs(abs(heading_difference(h1, h2)) - math.pi) <= tol_rad


def project_onto_polyline(
    point: Tuple[float, float], polyline: Sequence[Tuple[float, float]]
) -> Tuple[float, float, float]:
    """Project point onto polyline.

    Returns (arclength_at_projection, signed_lateral_offset, segment_heading).
    """
    if polyline is None or len(polyline) < 2:
        raise ValueError("polyline needs at least 2 points")
    best = None
    cumulative = 0.0
    for i in range(len(polyline) - 1):
        a = polyline[i]
        b = polyline[i + 1]
        seg_dx, seg_dy = b[0] - a[0], b[1] - a[1]
        seg_len = math.hypot(seg_dx, seg_dy)
        if seg_len < 1e-9:
            continue
        # Parametric projection clamped to [0, 1]
        t = ((point[0] - a[0]) * seg_dx + (point[1] - a[1]) * seg_dy) / (seg_len * seg_len)
        t_clamped = max(0.0, min(1.0, t))
        proj_x = a[0] + seg_dx * t_clamped
        proj_y = a[1] + seg_dy * t_clamped
        d = math.hypot(point[0] - proj_x, point[1] - proj_y)
        if best is None or d < best[0]:
            heading = math.atan2(seg_dy, seg_dx)
            lateral = signed_lateral_offset(a, b, point)
            arclen = cumulative + seg_len * t_clamped
            best = (d, arclen, lateral, heading)
        cumulative += seg_len
    if best is None:
        raise ValueError("degenerate polyline")
    return best[1], best[2], best[3]


def lane_heading(lane: LaneSnapshot) -> float:
    if lane.heading_at_start is not None:
        return lane.heading_at_start
    if lane.centerline is None or len(lane.centerline) < 2:
        return 0.0
    a, b = lane.centerline[0], lane.centerline[1]
    return math.atan2(b[1] - a[1], b[0] - a[0])


def lane_polygon(lane: LaneSnapshot) -> Optional[Polygon]:
    return polygon_from_points(lane.polygon)


def find_ego_lane(ego: EgoSnapshot, lanes: Iterable[LaneSnapshot]) -> Optional[LaneSnapshot]:
    """Return the lane whose polygon contains the ego center, if any.

    On a tie (two lanes' polygons overlap the center), picks the lane whose
    heading best matches the ego heading.
    """
    ex, ey = ego_center(ego)
    pt = Point(ex, ey)
    best: Optional[LaneSnapshot] = None
    best_score = -math.inf
    for lane in lanes:
        poly = lane_polygon(lane)
        if poly is None or not poly.contains(pt):
            continue
        score = -abs(heading_difference(lane_heading(lane), ego.pose.heading))
        if score > best_score:
            best_score = score
            best = lane
    return best


def footprint_overlaps_polygon(footprint: Polygon, polygon: Polygon) -> float:
    """Area (m^2) of overlap between ``footprint`` and ``polygon``; 0 if disjoint."""
    if not footprint.intersects(polygon):
        return 0.0
    return float(footprint.intersection(polygon).area)


def footprint_outside_drivable(footprint: Polygon, drivable_polygons: Iterable[Polygon]) -> float:
    """Area of ``footprint`` outside the union of drivable polygons.

    If no drivable polygons are supplied, returns 0 (i.e. assume drivable).
    """
    polys = [p for p in drivable_polygons if p is not None and not p.is_empty]
    if not polys:
        return 0.0
    union: BaseGeometry = polys[0]
    for p in polys[1:]:
        union = union.union(p)
    outside = footprint.difference(union)
    return float(outside.area) if not outside.is_empty else 0.0


def closest_agent(
    ego: EgoSnapshot, agents: Iterable[AgentSnapshot]
) -> Tuple[Optional[AgentSnapshot], float]:
    """Return the closest agent to the ego and the planar center-to-center distance."""
    ec = ego_center(ego)
    best: Optional[AgentSnapshot] = None
    best_d = math.inf
    for a in agents:
        d = planar_distance(ec, (a.pose.x, a.pose.y))
        if d < best_d:
            best_d = d
            best = a
    return best, best_d
