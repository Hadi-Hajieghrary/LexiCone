"""Adapter from a NuPlan ``AbstractScenario`` to :class:`SceneSnapshot`.

The module imports nuplan-devkit lazily so the rest of the observer package
can be used (and unit-tested) without nuplan installed. Use:

    >>> from lexicone.observer.nuplan_adapter import NuPlanSceneSource
    >>> source = NuPlanSceneSource(scenario, radius_m=80.0)
    >>> for snap in source:
    ...     evals = observer.step(snap)
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Iterator, List, Optional, Tuple

from .types import (
    AgentSnapshot,
    AgentType,
    CrosswalkSnapshot,
    DrivableAreaSnapshot,
    EgoSnapshot,
    IntersectionSnapshot,
    LaneSnapshot,
    MapSnapshot,
    Pose2D,
    SceneSnapshot,
    StopLineSnapshot,
    StopType,
    TrafficLightState,
    TrafficLightStatus,
    WalkwaySnapshot,
)


# Map NuPlan TrackedObjectType -> our AgentType.
_NUPLAN_TYPE_TO_AGENT = {
    "VEHICLE": AgentType.VEHICLE,
    "PEDESTRIAN": AgentType.PEDESTRIAN,
    "BICYCLE": AgentType.BICYCLE,
    "TRAFFIC_CONE": AgentType.TRAFFIC_CONE,
    "BARRIER": AgentType.BARRIER,
    "CZONE_SIGN": AgentType.CZONE_SIGN,
    "GENERIC_OBJECT": AgentType.GENERIC_OBJECT,
    "EGO": AgentType.VEHICLE,
}

_NUPLAN_TL_TO_STATE = {
    "GREEN": TrafficLightState.GREEN,
    "YELLOW": TrafficLightState.YELLOW,
    "RED": TrafficLightState.RED,
    "UNKNOWN": TrafficLightState.UNKNOWN,
}


def _agent_type(nuplan_type: Any) -> AgentType:
    name = getattr(nuplan_type, "name", None) or getattr(nuplan_type, "fullname", None) or str(nuplan_type)
    return _NUPLAN_TYPE_TO_AGENT.get(name, AgentType.UNKNOWN)


def _tl_state(nuplan_state: Any) -> TrafficLightState:
    name = getattr(nuplan_state, "name", None) or str(nuplan_state)
    return _NUPLAN_TL_TO_STATE.get(name, TrafficLightState.UNKNOWN)


class NuPlanSceneSource:
    """Iterate ``SceneSnapshot``s over a NuPlan scenario."""

    def __init__(
        self,
        scenario: Any,
        radius_m: float = 80.0,
        planner_predictions: Optional[Iterable[Any]] = None,
        route_lane_ids: Optional[Iterable[str]] = None,
        include_lane_connectors: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        scenario:
            A ``nuplan.planning.scenario_builder.abstract_scenario.AbstractScenario``.
        radius_m:
            Region-of-interest radius around the ego for selecting map objects
            and tracked agents.
        planner_predictions:
            Optional iterable, one element per iteration, of the planner's
            planned future trajectory (a sequence of ``EgoState``-like
            objects). If omitted, ``planned_trajectory`` is left None.
        route_lane_ids:
            Optional global-route lane IDs (constant across the scenario).
        include_lane_connectors:
            If True, query lane connectors in addition to lanes.
        """
        self.scenario = scenario
        self.radius_m = float(radius_m)
        self._predictions = list(planner_predictions) if planner_predictions is not None else None
        self._route_lane_ids = list(route_lane_ids) if route_lane_ids is not None else None
        self.include_lane_connectors = include_lane_connectors

        # Lazy-import the SemanticMapLayer enum.
        from nuplan.common.maps.maps_datatypes import SemanticMapLayer  # type: ignore

        self._SL = SemanticMapLayer

    def __iter__(self) -> Iterator[SceneSnapshot]:
        n = self.scenario.get_number_of_iterations()
        for i in range(n):
            yield self.snapshot_at(i)

    def snapshot_at(self, iteration: int) -> SceneSnapshot:
        ego_state = self.scenario.get_ego_state_at_iteration(iteration)
        detections = self.scenario.get_tracked_objects_at_iteration(iteration)
        tls = self.scenario.get_traffic_light_status_at_iteration(iteration)

        ego = _ego_to_snapshot(ego_state)
        agents = _detections_to_agents(detections)
        map_api = self.scenario.map_api
        map_snap = _map_to_snapshot(map_api, ego, self.radius_m, self._SL, self.include_lane_connectors)
        traffic_lights = _tls_to_status(tls)

        planned = None
        if self._predictions is not None and iteration < len(self._predictions):
            planned = [_ego_to_snapshot(s) for s in self._predictions[iteration] or []]

        return SceneSnapshot(
            timestamp_us=int(ego_state.time_point.time_us),
            ego=ego,
            agents=agents,
            map=map_snap,
            traffic_lights=traffic_lights,
            planned_trajectory=planned,
            route_lane_ids=self._route_lane_ids,
        )


# --- helpers -----------------------------------------------------------------


def _ego_to_snapshot(ego_state: Any) -> EgoSnapshot:
    """Convert a nuplan ``EgoState`` to :class:`EgoSnapshot`.

    Uses ``rear_axle`` (NuPlan convention) for pose and ``dynamic_car_state`` for
    velocities/accelerations. Dimensions come from ``car_footprint``.
    """
    pose = ego_state.rear_axle
    dyn = ego_state.dynamic_car_state
    fp = ego_state.car_footprint
    # NuPlan velocities/accelerations are in ego (rear-axle) frame.
    # Rotate to world frame using the heading.
    heading = pose.heading
    cos_h = math.cos(heading)
    sin_h = math.sin(heading)
    vx_body = getattr(dyn, "rear_axle_velocity_2d", None)
    if vx_body is not None and hasattr(vx_body, "x"):
        v_lon = vx_body.x
        v_lat = vx_body.y
    else:
        v_lon = float(getattr(dyn, "speed", 0.0))
        v_lat = 0.0
    vx_world = v_lon * cos_h - v_lat * sin_h
    vy_world = v_lon * sin_h + v_lat * cos_h
    ax_body = getattr(dyn, "rear_axle_acceleration_2d", None)
    if ax_body is not None and hasattr(ax_body, "x"):
        a_lon = ax_body.x
        a_lat = ax_body.y
    else:
        a_lon = 0.0
        a_lat = 0.0
    ax_world = a_lon * cos_h - a_lat * sin_h
    ay_world = a_lon * sin_h + a_lat * cos_h
    yaw_rate = float(getattr(dyn, "angular_velocity", 0.0))

    length = float(getattr(fp, "length", getattr(fp, "vehicle_parameters", None) or 4.7))
    if hasattr(fp, "vehicle_parameters"):
        vp = fp.vehicle_parameters
        length = float(vp.length)
        width = float(vp.width)
        rear_to_center = float(vp.rear_axle_to_center)
    else:
        length = float(getattr(fp, "length", 4.7))
        width = float(getattr(fp, "width", 1.85))
        rear_to_center = float(getattr(fp, "rear_axle_to_center", 1.46))

    return EgoSnapshot(
        timestamp_us=int(ego_state.time_point.time_us),
        pose=Pose2D(x=float(pose.x), y=float(pose.y), heading=float(pose.heading)),
        vx=float(vx_world),
        vy=float(vy_world),
        ax=float(ax_world),
        ay=float(ay_world),
        yaw_rate=yaw_rate,
        length=length,
        width=width,
        rear_axle_to_center=rear_to_center,
        pose_at_center=False,
    )


def _detections_to_agents(detections: Any) -> List[AgentSnapshot]:
    out: List[AgentSnapshot] = []
    objs = getattr(detections, "tracked_objects", None) or detections
    for obj in objs:
        box = getattr(obj, "box", None) or getattr(obj, "_box", None)
        velocity = getattr(obj, "velocity", None)
        center = getattr(obj, "center", None) or getattr(box, "center", None)
        if box is None or center is None:
            continue
        heading = float(getattr(center, "heading", 0.0))
        vx = vy = 0.0
        if velocity is not None:
            vx = float(getattr(velocity, "x", 0.0))
            vy = float(getattr(velocity, "y", 0.0))
        out.append(
            AgentSnapshot(
                track_id=str(getattr(obj, "track_token", getattr(obj, "metadata", obj).__hash__())),
                object_type=_agent_type(getattr(obj, "tracked_object_type", None)),
                pose=Pose2D(x=float(center.x), y=float(center.y), heading=heading),
                vx=vx,
                vy=vy,
                length=float(getattr(box, "length", 4.5)),
                width=float(getattr(box, "width", 1.8)),
            )
        )
    return out


def _tls_to_status(tls: Any) -> List[TrafficLightStatus]:
    out: List[TrafficLightStatus] = []
    for tl in tls or []:
        lc_id = str(getattr(tl, "lane_connector_id", ""))
        if not lc_id:
            continue
        out.append(
            TrafficLightStatus(
                lane_connector_id=lc_id,
                state=_tl_state(getattr(tl, "status", None)),
            )
        )
    return out


def _polygon_xy(poly: Any) -> List[Tuple[float, float]]:
    """Extract (x, y) coords from a shapely-like polygon or geometry."""
    geom = getattr(poly, "geometry", None) or poly
    coords = []
    exterior = getattr(geom, "exterior", None)
    if exterior is not None:
        coords = list(exterior.coords)
    elif hasattr(geom, "coords"):
        coords = list(geom.coords)
    return [(float(x), float(y)) for x, y in coords]


def _polyline_xy(line: Any) -> List[Tuple[float, float]]:
    geom = getattr(line, "linestring", None) or getattr(line, "geometry", None) or line
    coords = getattr(geom, "coords", [])
    return [(float(x), float(y)) for x, y in coords]


def _map_to_snapshot(
    map_api: Any,
    ego: EgoSnapshot,
    radius_m: float,
    SL: Any,
    include_lane_connectors: bool,
) -> MapSnapshot:
    """Query the nuplan map for elements near the ego and pack into MapSnapshot."""

    from nuplan.common.actor_state.state_representation import Point2D  # type: ignore

    point = Point2D(ego.pose.x, ego.pose.y)
    layers = [
        SL.LANE,
        SL.INTERSECTION,
        SL.STOP_LINE,
        SL.CROSSWALK,
        SL.WALKWAYS,
        SL.DRIVABLE_AREA,
    ]
    if include_lane_connectors:
        layers.append(SL.LANE_CONNECTOR)

    proximal = map_api.get_proximal_map_objects(point, radius_m, layers)

    lanes: List[LaneSnapshot] = []
    for lane in proximal.get(SL.LANE, []):
        lanes.append(_convert_lane(lane, is_lane_connector=False))

    lane_connectors: List[LaneSnapshot] = []
    if include_lane_connectors:
        for lc in proximal.get(SL.LANE_CONNECTOR, []):
            lane_connectors.append(_convert_lane(lc, is_lane_connector=True))

    crosswalks = [
        CrosswalkSnapshot(crosswalk_id=str(o.id), polygon=_polygon_xy(o.polygon), is_marked=True)
        for o in proximal.get(SL.CROSSWALK, [])
    ]

    stop_lines: List[StopLineSnapshot] = []
    for o in proximal.get(SL.STOP_LINE, []):
        try:
            polyline = _polyline_xy(o.polygon if hasattr(o, "polygon") else o)
        except Exception:
            polyline = []
        stop_type_name = getattr(o, "stop_line_type", None)
        stype = StopType.GENERIC
        if stop_type_name is not None:
            n = getattr(stop_type_name, "name", str(stop_type_name))
            if "STOP" in n.upper():
                stype = StopType.STOP_SIGN
            elif "TRAFFIC" in n.upper() or "LIGHT" in n.upper():
                stype = StopType.TRAFFIC_LIGHT
            elif "YIELD" in n.upper():
                stype = StopType.YIELD_SIGN
        stop_lines.append(
            StopLineSnapshot(
                stop_line_id=str(o.id),
                polyline=polyline,
                stop_type=stype,
                associated_lane_id=None,
            )
        )

    intersections = [
        IntersectionSnapshot(
            intersection_id=str(o.id),
            polygon=_polygon_xy(o.polygon),
            is_signalized=bool(getattr(o, "is_intersection", False)),
        )
        for o in proximal.get(SL.INTERSECTION, [])
    ]

    drivable = [
        DrivableAreaSnapshot(polygon=_polygon_xy(o.polygon))
        for o in proximal.get(SL.DRIVABLE_AREA, [])
    ]

    walkways = [
        WalkwaySnapshot(walkway_id=str(o.id), polygon=_polygon_xy(o.polygon))
        for o in proximal.get(SL.WALKWAYS, [])
    ]

    return MapSnapshot(
        lanes=tuple(lanes),
        lane_connectors=tuple(lane_connectors),
        crosswalks=tuple(crosswalks),
        stop_lines=tuple(stop_lines),
        intersections=tuple(intersections),
        drivable_area=tuple(drivable),
        walkways=tuple(walkways),
        bike_lanes=tuple(),  # NuPlan does not expose dedicated bike-lane semantics
    )


def _convert_lane(lane: Any, is_lane_connector: bool) -> LaneSnapshot:
    polygon = _polygon_xy(lane.polygon)
    baseline = getattr(lane, "baseline_path", None) or getattr(lane, "centerline", None)
    centerline = _polyline_xy(baseline) if baseline is not None else []
    speed_limit = getattr(lane, "speed_limit_mps", None)
    inc = tuple(str(getattr(l, "id", l)) for l in (getattr(lane, "incoming_edges", []) or []))
    out = tuple(str(getattr(l, "id", l)) for l in (getattr(lane, "outgoing_edges", []) or []))
    heading_start = None
    if len(centerline) >= 2:
        a, b = centerline[0], centerline[1]
        heading_start = math.atan2(b[1] - a[1], b[0] - a[0])
    return LaneSnapshot(
        lane_id=str(lane.id),
        centerline=centerline,
        polygon=polygon,
        speed_limit_mps=float(speed_limit) if speed_limit is not None else None,
        heading_at_start=heading_start,
        is_in_intersection=is_lane_connector,
        is_bike_lane=False,
        is_lane_connector=is_lane_connector,
        incoming_lane_ids=inc,
        outgoing_lane_ids=out,
    )
