"""Shared helpers used by multiple observer rules."""

from __future__ import annotations

import math
from typing import Iterable, List, Optional, Sequence, Tuple

from shapely.geometry import Polygon

from ..geometry import (
    agent_footprint,
    ego_center,
    ego_footprint,
    find_ego_lane,
    heading_difference,
    is_opposite_direction,
    is_same_direction,
    lane_heading,
    lane_polygon,
    planar_distance,
    polygon_from_points,
    project_onto_polyline,
)
from ..types import (
    AgentSnapshot,
    AgentType,
    EgoSnapshot,
    LaneSnapshot,
    SceneSnapshot,
    TrafficLightState,
)


VRU_TYPES = {AgentType.PEDESTRIAN.value, AgentType.BICYCLE.value}
VEHICLE_LIKE_TYPES = {
    AgentType.VEHICLE.value,
    AgentType.MOTORCYCLE.value,
    AgentType.BICYCLE.value,
}


def agents_by_type(
    snap: SceneSnapshot, types: Iterable[str | AgentType], radius_m: Optional[float] = None
) -> List[AgentSnapshot]:
    """Return agents matching ``types`` (optionally within ``radius_m`` of ego)."""
    wanted = {t.value if isinstance(t, AgentType) else t for t in types}
    ec = ego_center(snap.ego)
    out: List[AgentSnapshot] = []
    for a in snap.agents:
        ot = a.object_type.value if isinstance(a.object_type, AgentType) else a.object_type
        if ot not in wanted:
            continue
        if radius_m is not None and planar_distance(ec, (a.pose.x, a.pose.y)) > radius_m:
            continue
        out.append(a)
    return out


def ego_in_intersection(snap: SceneSnapshot) -> Tuple[bool, Optional[str]]:
    """Whether any part of the ego footprint overlaps any intersection polygon."""
    fp = ego_footprint(snap.ego)
    for inter in snap.map.intersections:
        poly = polygon_from_points(inter.polygon)
        if poly is None:
            continue
        if fp.intersects(poly):
            return True, inter.intersection_id
    return False, None


def lead_agent_in_lane(
    snap: SceneSnapshot, max_distance_m: float = 80.0, lateral_tol_m: float = 1.6
) -> Tuple[Optional[AgentSnapshot], Optional[float], Optional[LaneSnapshot]]:
    """Find the nearest in-lane lead vehicle/bicycle ahead of ego.

    Returns (agent, longitudinal_distance, ego_lane). ``longitudinal_distance``
    is measured along the ego lane centerline if available, otherwise the
    Euclidean distance along the ego heading.
    """
    ego = snap.ego
    ec = ego_center(ego)
    ego_lane = find_ego_lane(ego, list(snap.map.lanes) + list(snap.map.lane_connectors))
    centerline = ego_lane.centerline if ego_lane is not None else None

    best: Tuple[Optional[AgentSnapshot], Optional[float]] = (None, None)
    best_lon = math.inf
    for a in snap.agents:
        ot = a.object_type.value if isinstance(a.object_type, AgentType) else a.object_type
        if ot not in VEHICLE_LIKE_TYPES:
            continue
        if not is_same_direction(a.pose.heading, ego.pose.heading, math.radians(45.0)):
            continue
        if planar_distance(ec, (a.pose.x, a.pose.y)) > max_distance_m + 5.0:
            continue
        if centerline is not None and len(centerline) >= 2:
            ego_s, _, _ = project_onto_polyline(ec, centerline)
            ag_s, ag_lat, _ = project_onto_polyline((a.pose.x, a.pose.y), centerline)
            lon = ag_s - ego_s
            if lon < 0.5 or abs(ag_lat) > lateral_tol_m:
                continue
        else:
            # Fallback: project onto ego heading.
            dx = a.pose.x - ec[0]
            dy = a.pose.y - ec[1]
            cos_h = math.cos(ego.pose.heading)
            sin_h = math.sin(ego.pose.heading)
            lon = dx * cos_h + dy * sin_h
            lat = -dx * sin_h + dy * cos_h
            if lon < 0.5 or abs(lat) > lateral_tol_m:
                continue
        if lon < best_lon and lon <= max_distance_m:
            best_lon = lon
            best = (a, lon)
    return best[0], best[1], ego_lane


def lateral_neighbors(
    snap: SceneSnapshot, max_long_m: float = 8.0, lateral_band_m: float = 4.0
) -> List[Tuple[AgentSnapshot, float, float]]:
    """Agents currently alongside the ego (within a short longitudinal band).

    Returns a list of (agent, signed_lateral_offset, longitudinal_offset)
    triples, all in the ego frame. Positive lateral = left of ego.
    """
    out: List[Tuple[AgentSnapshot, float, float]] = []
    ec = ego_center(snap.ego)
    cos_h = math.cos(snap.ego.pose.heading)
    sin_h = math.sin(snap.ego.pose.heading)
    for a in snap.agents:
        ot = a.object_type.value if isinstance(a.object_type, AgentType) else a.object_type
        if ot not in VEHICLE_LIKE_TYPES and ot not in VRU_TYPES:
            continue
        dx = a.pose.x - ec[0]
        dy = a.pose.y - ec[1]
        lon = dx * cos_h + dy * sin_h
        lat = -dx * sin_h + dy * cos_h
        if abs(lon) <= max_long_m and abs(lat) <= lateral_band_m:
            out.append((a, lat, lon))
    return out


def relative_velocity(ego: EgoSnapshot, agent: AgentSnapshot) -> Tuple[float, float]:
    """Velocity of ``agent`` minus ``ego`` in the ego frame.

    Returns (longitudinal, lateral); positive longitudinal = agent moving forward
    relative to ego.
    """
    rvx = agent.vx - ego.vx
    rvy = agent.vy - ego.vy
    cos_h = math.cos(ego.pose.heading)
    sin_h = math.sin(ego.pose.heading)
    lon = rvx * cos_h + rvy * sin_h
    lat = -rvx * sin_h + rvy * cos_h
    return lon, lat


def red_or_yellow_for_lane(
    snap: SceneSnapshot, lane_id: Optional[str]
) -> Tuple[bool, Optional[str]]:
    """Whether the supplied lane connector ID is showing red or yellow.

    Returns (active_red_or_yellow, state_string).
    """
    if lane_id is None:
        return False, None
    for tl in snap.traffic_lights:
        if tl.lane_connector_id == lane_id:
            state = tl.state.value if isinstance(tl.state, TrafficLightState) else tl.state
            return state in ("RED", "YELLOW"), state
    return False, None


def stop_line_for_ego(
    snap: SceneSnapshot, ego_lane: Optional[LaneSnapshot]
) -> Optional[Tuple[Polygon, float]]:
    """Find a stop polygon associated with the ego's current lane.

    Returns a buffered polygon around the stop polyline and the ego's distance
    to the polyline (negative when ego has passed it along the lane direction).
    """
    if ego_lane is None or not snap.map.stop_lines:
        return None
    centerline = ego_lane.centerline
    if not centerline or len(centerline) < 2:
        return None
    ec = ego_center(snap.ego)
    ego_s, _, _ = project_onto_polyline(ec, centerline)
    best: Optional[Tuple[Polygon, float]] = None
    best_dist = math.inf
    for sl in snap.map.stop_lines:
        if sl.associated_lane_id is not None and sl.associated_lane_id != ego_lane.lane_id:
            continue
        if not sl.polyline or len(sl.polyline) < 2:
            continue
        midx = sum(p[0] for p in sl.polyline) / len(sl.polyline)
        midy = sum(p[1] for p in sl.polyline) / len(sl.polyline)
        sl_s, _, _ = project_onto_polyline((midx, midy), centerline)
        dist = sl_s - ego_s  # positive if ahead
        if abs(dist) > 35.0:
            continue
        if abs(dist) < best_dist:
            best_dist = abs(dist)
            # Buffer the stop polyline into a small polygon for footprint checks.
            from shapely.geometry import LineString

            poly = LineString(sl.polyline).buffer(0.5)
            best = (poly, dist)
    return best


def speed_limit_for_ego(snap: SceneSnapshot) -> Optional[float]:
    ego_lane = find_ego_lane(snap.ego, list(snap.map.lanes) + list(snap.map.lane_connectors))
    if ego_lane is None or ego_lane.speed_limit_mps is None:
        return None
    return float(ego_lane.speed_limit_mps)


def ego_overlaps_walkway(snap: SceneSnapshot) -> Tuple[bool, float]:
    fp = ego_footprint(snap.ego)
    total = 0.0
    for w in snap.map.walkways:
        poly = polygon_from_points(w.polygon)
        if poly is None:
            continue
        if fp.intersects(poly):
            total += float(fp.intersection(poly).area)
    return total > 0.0, total


def ego_overlaps_bike_lane(snap: SceneSnapshot) -> Tuple[bool, float]:
    fp = ego_footprint(snap.ego)
    total = 0.0
    for bl in snap.map.bike_lanes:
        poly = lane_polygon(bl)
        if poly is None:
            continue
        if fp.intersects(poly):
            total += float(fp.intersection(poly).area)
    return total > 0.0, total
