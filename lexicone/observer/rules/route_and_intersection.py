"""Route, yielding, and intersection-etiquette rules.

- 2r2 — Adhere to the planned global route.
- 1r0 — Yield to higher-priority road users.
- 1r2 — Don't block the box (intersection).
- 1r5 — Negotiate uncontrolled intersections safely.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional, Tuple

from shapely.geometry import Point

from ..geometry import (
    ego_center,
    ego_footprint,
    find_ego_lane,
    is_same_direction,
    lane_polygon,
    planar_distance,
    project_onto_polyline,
)
from ..rule import ObserverRule
from ..types import AgentType, SceneSnapshot, TrafficLightState
from ._common import (
    VEHICLE_LIKE_TYPES,
    ego_in_intersection,
    red_or_yellow_for_lane,
    relative_velocity,
)


class RouteAdherenceRule(ObserverRule):
    id = "2r2"
    level = 2
    name = "Adhere to the planned global route"
    description = (
        "Penalises geometric drift outside the planned-route lane corridor "
        "and topological deviations (ego in a lane outside the route)."
    )

    def __init__(self, corridor_lateral_m: float = 2.0):
        self.corridor_lateral_m = corridor_lateral_m

    def applies_at(self, snap: SceneSnapshot) -> Tuple[bool, Mapping[str, Any]]:
        if snap.route_lane_ids is None or not snap.route_lane_ids:
            return False, {"reason": "no_route"}
        return True, {"n_route_lanes": len(snap.route_lane_ids)}

    def violation_at(self, snap: SceneSnapshot) -> Tuple[float, Mapping[str, Any]]:
        route = set(snap.route_lane_ids or [])
        ego_lane = find_ego_lane(
            snap.ego, list(snap.map.lanes) + list(snap.map.lane_connectors)
        )
        in_route = ego_lane is not None and ego_lane.lane_id in route
        if in_route:
            return 0.0, {"ego_lane_id": ego_lane.lane_id, "on_route": True}
        # Compute lateral distance to nearest route lane centerline.
        ec = ego_center(snap.ego)
        best_lat = math.inf
        nearest_id = None
        for lane in list(snap.map.lanes) + list(snap.map.lane_connectors):
            if lane.lane_id not in route:
                continue
            if not lane.centerline or len(lane.centerline) < 2:
                continue
            _, lat, _ = project_onto_polyline(ec, lane.centerline)
            if abs(lat) < best_lat:
                best_lat = abs(lat)
                nearest_id = lane.lane_id
        drift = max(0.0, (best_lat if best_lat != math.inf else 5.0) - self.corridor_lateral_m)
        return drift, {
            "ego_lane_id": ego_lane.lane_id if ego_lane is not None else None,
            "on_route": False,
            "nearest_route_lane_id": nearest_id,
            "lateral_to_route_m": best_lat if best_lat != math.inf else None,
        }


class YieldPriorityRule(ObserverRule):
    id = "1r0"
    level = 1
    name = "Yield to higher-priority road users"
    description = (
        "Penalises encroachment that forces a prioritized agent (e.g. a "
        "through vehicle on a major road, a pedestrian with right-of-way) to "
        "brake beyond comfort or face critically low TTC."
    )

    def __init__(self, comfort_decel_mps2: float = 2.0, ttc_critical_s: float = 2.0):
        self.comfort_decel_mps2 = comfort_decel_mps2
        self.ttc_critical_s = ttc_critical_s

    def _priority_agents(self, snap: SceneSnapshot):
        """Agents whose ROW may be affected by ego encroachment.

        For now: pedestrians/cyclists within 20m, and vehicles whose path
        crosses the ego footprint forward projection.
        """
        ec = ego_center(snap.ego)
        for a in snap.agents:
            ot = a.object_type.value if isinstance(a.object_type, AgentType) else a.object_type
            if ot == AgentType.PEDESTRIAN.value:
                if planar_distance(ec, (a.pose.x, a.pose.y)) <= 15.0:
                    yield a
            elif ot in VEHICLE_LIKE_TYPES:
                # Cross-traffic check: agent moving roughly perpendicular and close.
                cos_h = math.cos(snap.ego.pose.heading)
                sin_h = math.sin(snap.ego.pose.heading)
                dx = a.pose.x - ec[0]
                dy = a.pose.y - ec[1]
                lon = dx * cos_h + dy * sin_h
                lat = -dx * sin_h + dy * cos_h
                if 0 < lon < 25.0 and abs(lat) < 15.0 and not is_same_direction(
                    a.pose.heading, snap.ego.pose.heading, math.radians(45)
                ):
                    yield a

    def applies_at(self, snap: SceneSnapshot) -> Tuple[bool, Mapping[str, Any]]:
        peers = list(self._priority_agents(snap))
        return bool(peers), {"n_priority_agents": len(peers)}

    def violation_at(self, snap: SceneSnapshot) -> Tuple[float, Mapping[str, Any]]:
        worst = 0.0
        worst_track = None
        ec = ego_center(snap.ego)
        for a in self._priority_agents(snap):
            d = planar_distance(ec, (a.pose.x, a.pose.y))
            rel_lon, _ = relative_velocity(snap.ego, a)
            closing = -rel_lon
            ttc = d / closing if closing > 0.3 else math.inf
            forced_decel = (closing * closing) / (2.0 * max(d, 0.3)) if closing > 0 else 0.0
            score = max(0.0, forced_decel - self.comfort_decel_mps2)
            if ttc != math.inf and ttc < self.ttc_critical_s:
                score += (self.ttc_critical_s - ttc)
            if score > worst:
                worst = score
                worst_track = a.track_id
        return worst, {
            "worst_track_id": worst_track,
            "comfort_decel_mps2": self.comfort_decel_mps2,
            "ttc_critical_s": self.ttc_critical_s,
        }


class BlockTheBoxRule(ObserverRule):
    id = "1r2"
    level = 1
    name = "Don't block the box"
    description = (
        "Penalises being essentially stopped inside an intersection without a "
        "clear downstream gap to exit."
    )

    def __init__(self, stop_speed_mps: float = 0.5, gap_lookahead_m: float = 10.0):
        self.stop_speed_mps = stop_speed_mps
        self.gap_lookahead_m = gap_lookahead_m

    def applies_at(self, snap: SceneSnapshot) -> Tuple[bool, Mapping[str, Any]]:
        in_int, int_id = ego_in_intersection(snap)
        return in_int and snap.ego.speed <= self.stop_speed_mps, {
            "intersection_id": int_id,
            "ego_speed_mps": snap.ego.speed,
        }

    def violation_at(self, snap: SceneSnapshot) -> Tuple[float, Mapping[str, Any]]:
        # Look for a downstream vehicle blocking the next gap_lookahead_m.
        ec = ego_center(snap.ego)
        cos_h = math.cos(snap.ego.pose.heading)
        sin_h = math.sin(snap.ego.pose.heading)
        blocked = False
        blocking_track = None
        for a in snap.agents:
            ot = a.object_type.value if isinstance(a.object_type, AgentType) else a.object_type
            if ot not in VEHICLE_LIKE_TYPES:
                continue
            dx = a.pose.x - ec[0]
            dy = a.pose.y - ec[1]
            lon = dx * cos_h + dy * sin_h
            lat = -dx * sin_h + dy * cos_h
            if 0 < lon <= self.gap_lookahead_m and abs(lat) <= 2.0 and is_same_direction(
                a.pose.heading, snap.ego.pose.heading
            ) and a.speed < 0.5:
                blocked = True
                blocking_track = a.track_id
                break
        rate = 1.0 if blocked else 0.0
        return rate, {"downstream_blocked": blocked, "blocking_track_id": blocking_track}


class UncontrolledIntersectionRule(ObserverRule):
    id = "1r5"
    level = 1
    name = "Negotiate uncontrolled intersections safely"
    description = (
        "At intersections without a TL or stop sign for the ego's lane, "
        "penalises advancing through when another road user has priority (e.g. "
        "yield-to-right, first-to-stop)."
    )

    def __init__(self, t_advance_margin_s: float = 1.0):
        self.t_advance_margin_s = t_advance_margin_s

    def _intersection_is_uncontrolled(self, snap: SceneSnapshot, int_id: Optional[str]) -> bool:
        if int_id is None:
            return False
        # Any stop line whose associated lane is the ego's lane → controlled.
        ego_lane = find_ego_lane(snap.ego, list(snap.map.lanes) + list(snap.map.lane_connectors))
        if ego_lane is not None:
            for sl in snap.map.stop_lines:
                if sl.associated_lane_id == ego_lane.lane_id:
                    return False
        # Any TL state we can match for the connector → controlled.
        for lc in snap.map.lane_connectors:
            poly = lane_polygon(lc)
            if poly is None:
                continue
            if poly.contains(Point(*ego_center(snap.ego))):
                active, _ = red_or_yellow_for_lane(snap, lc.lane_id)
                if active:
                    return False
        return True

    def applies_at(self, snap: SceneSnapshot) -> Tuple[bool, Mapping[str, Any]]:
        in_int, int_id = ego_in_intersection(snap)
        if not in_int:
            return False, {}
        if not self._intersection_is_uncontrolled(snap, int_id):
            return False, {"reason": "controlled"}
        # Any cross-traffic candidate within range?
        peers = 0
        ec = ego_center(snap.ego)
        for a in snap.agents:
            ot = a.object_type.value if isinstance(a.object_type, AgentType) else a.object_type
            if ot not in VEHICLE_LIKE_TYPES:
                continue
            if planar_distance(ec, (a.pose.x, a.pose.y)) <= 20.0 and not is_same_direction(
                a.pose.heading, snap.ego.pose.heading, math.radians(60)
            ):
                peers += 1
        return peers > 0, {"intersection_id": int_id, "n_cross_traffic": peers}

    def violation_at(self, snap: SceneSnapshot) -> Tuple[float, Mapping[str, Any]]:
        # Yield-to-right convention: any cross-traffic vehicle to the ego's
        # right that is closer to the conflict point than the ego, moving,
        # earns priority. If ego is still advancing, that's a violation.
        ec = ego_center(snap.ego)
        cos_h = math.cos(snap.ego.pose.heading)
        sin_h = math.sin(snap.ego.pose.heading)
        worst = 0.0
        worst_track = None
        for a in snap.agents:
            ot = a.object_type.value if isinstance(a.object_type, AgentType) else a.object_type
            if ot not in VEHICLE_LIKE_TYPES:
                continue
            dx = a.pose.x - ec[0]
            dy = a.pose.y - ec[1]
            lat = -dx * sin_h + dy * cos_h  # ego frame
            lon = dx * cos_h + dy * sin_h
            if lat >= 0:  # not to the ego's right
                continue
            if not (0 < lon < 20.0 and a.speed > 0.5):
                continue
            # Time advantage: who reaches the conflict first? Approximate
            # using current speeds and distances to ec.
            d_ego_to_conflict = max(lon - 2.0, 0.0)
            t_ego = d_ego_to_conflict / max(snap.ego.speed, 0.5)
            d_a = math.hypot(dx, dy)
            t_a = d_a / max(a.speed, 0.5)
            if t_a + self.t_advance_margin_s <= t_ego:
                # Other vehicle arrives first → ego must yield.
                # If ego is still moving forward, that's a violation.
                rate = max(0.0, snap.ego.speed - 0.5)
                if rate > worst:
                    worst = rate
                    worst_track = a.track_id
        return worst, {"yielding_to_track_id": worst_track}
