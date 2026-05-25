"""Per-tick situational context for the rule engine.

The :class:`SceneContext` packages a raw :class:`SceneSnapshot` together with
all derivations that more than one rule would otherwise recompute: ego
footprint, ego-containing lane, intersection / walkway / bike-lane overlaps,
nearest in-lane lead vehicle, drivable polygons, and so on.

Heavy derivations are :func:`functools.cached_property`s so a rule that never
asks for them pays nothing. The :class:`RuleEngine` builds one context per
tick and passes it to every rule; rules call ``ctx.<field>`` instead of
re-deriving from the raw snapshot.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Iterable, List, Optional, Sequence, Tuple

from shapely.geometry import LineString, Polygon
from shapely.geometry.base import BaseGeometry

from .geometry import (
    agent_footprint,
    ego_center,
    ego_footprint,
    find_ego_lane,
    is_same_direction,
    lane_polygon,
    planar_distance,
    polygon_from_points,
    project_onto_polyline,
)
from .types import (
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


def _type_value(t: Any) -> str:
    return t.value if isinstance(t, AgentType) else str(t)


@dataclass
class LeadAgent:
    """Result of an in-lane lead-vehicle query."""

    agent: AgentSnapshot
    longitudinal_distance_m: float
    ego_lane: Optional[LaneSnapshot]


@dataclass
class LateralNeighbor:
    """An agent currently alongside the ego in the ego frame."""

    agent: AgentSnapshot
    lateral_offset_m: float  # +left, -right
    longitudinal_offset_m: float  # +ahead


class SceneContext:
    """All the situational information rules need for one tick.

    The context wraps a :class:`SceneSnapshot` and exposes both eagerly cached
    derivations (``ego_footprint``, ``ego_lane``, ``drivable_polygons``, …)
    and parameterised queries (``lead_agent``, ``lateral_neighbors``,
    ``red_or_yellow_for_lane``, …). All members are read-only with respect to
    the underlying snapshot; the engine constructs a fresh context per tick.
    """

    __slots__ = ("snapshot", "__dict__")

    def __init__(self, snapshot: SceneSnapshot) -> None:
        self.snapshot = snapshot

    # ----- convenience pass-throughs -----

    @property
    def ego(self) -> EgoSnapshot:
        return self.snapshot.ego

    @property
    def timestamp_us(self) -> int:
        return self.snapshot.timestamp_us

    # ----- ego derivations -----

    @cached_property
    def ego_footprint(self) -> Polygon:
        return ego_footprint(self.snapshot.ego)

    @cached_property
    def ego_center(self) -> Tuple[float, float]:
        return ego_center(self.snapshot.ego)

    @cached_property
    def all_lanes(self) -> Tuple[LaneSnapshot, ...]:
        return tuple(self.snapshot.map.lanes) + tuple(self.snapshot.map.lane_connectors)

    @cached_property
    def ego_lane(self) -> Optional[LaneSnapshot]:
        return find_ego_lane(self.snapshot.ego, self.all_lanes)

    @cached_property
    def ego_speed_limit_mps(self) -> Optional[float]:
        lane = self.ego_lane
        if lane is None or lane.speed_limit_mps is None:
            return None
        return float(lane.speed_limit_mps)

    # ----- map overlap derivations -----

    @cached_property
    def drivable_polygons(self) -> List[Polygon]:
        polys: List[Polygon] = []
        for da in self.snapshot.map.drivable_area:
            p = polygon_from_points(da.polygon)
            if p is not None:
                polys.append(p)
        return polys

    @cached_property
    def ego_in_intersection(self) -> Tuple[bool, Optional[str]]:
        fp = self.ego_footprint
        for inter in self.snapshot.map.intersections:
            poly = polygon_from_points(inter.polygon)
            if poly is None:
                continue
            if fp.intersects(poly):
                return True, inter.intersection_id
        return False, None

    @cached_property
    def walkway_overlap_m2(self) -> float:
        fp = self.ego_footprint
        total = 0.0
        for w in self.snapshot.map.walkways:
            poly = polygon_from_points(w.polygon)
            if poly is None:
                continue
            if fp.intersects(poly):
                total += float(fp.intersection(poly).area)
        return total

    @cached_property
    def bike_lane_overlap_m2(self) -> float:
        fp = self.ego_footprint
        total = 0.0
        for bl in self.snapshot.map.bike_lanes:
            poly = lane_polygon(bl)
            if poly is None:
                continue
            if fp.intersects(poly):
                total += float(fp.intersection(poly).area)
        return total

    # ----- parameterised queries -----

    def agents_by_type(
        self,
        types: Iterable[Any],
        radius_m: Optional[float] = None,
    ) -> List[AgentSnapshot]:
        """Agents whose ``object_type`` matches ``types`` (optionally near ego)."""
        wanted = {_type_value(t) for t in types}
        ec = self.ego_center
        out: List[AgentSnapshot] = []
        for a in self.snapshot.agents:
            if _type_value(a.object_type) not in wanted:
                continue
            if radius_m is not None and planar_distance(ec, (a.pose.x, a.pose.y)) > radius_m:
                continue
            out.append(a)
        return out

    def lead_agent(
        self,
        max_distance_m: float = 80.0,
        lateral_tol_m: float = 1.6,
    ) -> Optional[LeadAgent]:
        """Nearest in-lane lead vehicle/bicycle ahead of ego (or ``None``)."""
        ego = self.snapshot.ego
        ec = self.ego_center
        ego_lane = self.ego_lane
        centerline = ego_lane.centerline if ego_lane is not None else None

        best_agent: Optional[AgentSnapshot] = None
        best_lon = math.inf
        for a in self.snapshot.agents:
            if _type_value(a.object_type) not in VEHICLE_LIKE_TYPES:
                continue
            if not _is_same_dir(a.pose.heading, ego.pose.heading):
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
                cos_h = math.cos(ego.pose.heading)
                sin_h = math.sin(ego.pose.heading)
                dx = a.pose.x - ec[0]
                dy = a.pose.y - ec[1]
                lon = dx * cos_h + dy * sin_h
                lat = -dx * sin_h + dy * cos_h
                if lon < 0.5 or abs(lat) > lateral_tol_m:
                    continue
            if lon < best_lon and lon <= max_distance_m:
                best_lon = lon
                best_agent = a
        if best_agent is None:
            return None
        return LeadAgent(agent=best_agent, longitudinal_distance_m=best_lon, ego_lane=ego_lane)

    def lateral_neighbors(
        self,
        max_long_m: float = 8.0,
        lateral_band_m: float = 4.0,
    ) -> List[LateralNeighbor]:
        """Agents currently alongside the ego (within a longitudinal band)."""
        out: List[LateralNeighbor] = []
        ec = self.ego_center
        cos_h = math.cos(self.snapshot.ego.pose.heading)
        sin_h = math.sin(self.snapshot.ego.pose.heading)
        for a in self.snapshot.agents:
            ot = _type_value(a.object_type)
            if ot not in VEHICLE_LIKE_TYPES and ot not in VRU_TYPES:
                continue
            dx = a.pose.x - ec[0]
            dy = a.pose.y - ec[1]
            lon = dx * cos_h + dy * sin_h
            lat = -dx * sin_h + dy * cos_h
            if abs(lon) <= max_long_m and abs(lat) <= lateral_band_m:
                out.append(LateralNeighbor(agent=a, lateral_offset_m=lat, longitudinal_offset_m=lon))
        return out

    def red_or_yellow_for_lane(self, lane_id: Optional[str]) -> Tuple[bool, Optional[str]]:
        """Whether the given lane connector is showing red or yellow."""
        if lane_id is None:
            return False, None
        for tl in self.snapshot.traffic_lights:
            if tl.lane_connector_id == lane_id:
                state = (
                    tl.state.value if isinstance(tl.state, TrafficLightState) else tl.state
                )
                return state in ("RED", "YELLOW"), state
        return False, None

    def stop_polygon_for_ego(
        self,
        ego_lane: Optional[LaneSnapshot] = None,
    ) -> Optional[Tuple[Polygon, float]]:
        """Buffered stop polygon associated with the ego's lane and the signed
        longitudinal distance to it (negative = passed)."""
        lane = ego_lane if ego_lane is not None else self.ego_lane
        if lane is None or not self.snapshot.map.stop_lines:
            return None
        centerline = lane.centerline
        if not centerline or len(centerline) < 2:
            return None
        ec = self.ego_center
        ego_s, _, _ = project_onto_polyline(ec, centerline)
        best: Optional[Tuple[Polygon, float]] = None
        best_abs = math.inf
        for sl in self.snapshot.map.stop_lines:
            if sl.associated_lane_id is not None and sl.associated_lane_id != lane.lane_id:
                continue
            if not sl.polyline or len(sl.polyline) < 2:
                continue
            midx = sum(p[0] for p in sl.polyline) / len(sl.polyline)
            midy = sum(p[1] for p in sl.polyline) / len(sl.polyline)
            sl_s, _, _ = project_onto_polyline((midx, midy), centerline)
            signed = sl_s - ego_s
            if abs(signed) > 35.0:
                continue
            if abs(signed) < best_abs:
                best_abs = abs(signed)
                poly = LineString(sl.polyline).buffer(0.5)
                best = (poly, signed)
        return best


def _is_same_dir(h1: float, h2: float) -> bool:
    return is_same_direction(h1, h2, math.radians(45.0))


def relative_velocity(ego: EgoSnapshot, agent: AgentSnapshot) -> Tuple[float, float]:
    """Velocity of ``agent`` minus ``ego`` in the ego frame.

    Returns (longitudinal, lateral); positive longitudinal = moving forward
    relative to ego.
    """
    rvx = agent.vx - ego.vx
    rvy = agent.vy - ego.vy
    cos_h = math.cos(ego.pose.heading)
    sin_h = math.sin(ego.pose.heading)
    lon = rvx * cos_h + rvy * sin_h
    lat = -rvx * sin_h + rvy * cos_h
    return lon, lat
