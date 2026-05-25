"""Synthetic scene builders for the observer tests."""

from __future__ import annotations

import math
from typing import Iterable, List, Optional, Sequence, Tuple

from lexicone.observer.types import (
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


def make_ego(
    x: float = 0.0,
    y: float = 0.0,
    heading: float = 0.0,
    speed: float = 5.0,
    ax: float = 0.0,
    ay: float = 0.0,
    yaw_rate: float = 0.0,
    timestamp_us: int = 0,
    length: float = 4.7,
    width: float = 1.85,
) -> EgoSnapshot:
    vx = speed * math.cos(heading)
    vy = speed * math.sin(heading)
    return EgoSnapshot(
        timestamp_us=timestamp_us,
        pose=Pose2D(x=x, y=y, heading=heading),
        vx=vx,
        vy=vy,
        ax=ax,
        ay=ay,
        yaw_rate=yaw_rate,
        length=length,
        width=width,
        rear_axle_to_center=0.0,
        pose_at_center=True,
    )


def make_agent(
    track_id: str,
    object_type: AgentType,
    x: float,
    y: float,
    heading: float = 0.0,
    speed: float = 0.0,
    length: float = 4.5,
    width: float = 1.8,
) -> AgentSnapshot:
    return AgentSnapshot(
        track_id=track_id,
        object_type=object_type,
        pose=Pose2D(x=x, y=y, heading=heading),
        vx=speed * math.cos(heading),
        vy=speed * math.sin(heading),
        length=length,
        width=width,
    )


def straight_lane(
    lane_id: str = "lane0",
    y_center: float = 0.0,
    length_m: float = 200.0,
    width: float = 3.5,
    speed_limit_mps: Optional[float] = 13.4,  # ~30 mph
    heading: float = 0.0,
    is_lane_connector: bool = False,
    is_bike_lane: bool = False,
) -> LaneSnapshot:
    x0 = -length_m / 2.0
    x1 = length_m / 2.0
    cos_h = math.cos(heading)
    sin_h = math.sin(heading)

    def _rotate(px, py):
        return (px * cos_h - py * sin_h, px * sin_h + py * cos_h)

    centerline = []
    for s in [x0, x1]:
        cx, cy = _rotate(s, y_center)
        centerline.append((cx, cy))
    half_w = width / 2.0
    polygon = []
    for sx, sy in [(x0, y_center - half_w), (x1, y_center - half_w), (x1, y_center + half_w), (x0, y_center + half_w)]:
        rx, ry = _rotate(sx, sy)
        polygon.append((rx, ry))
    return LaneSnapshot(
        lane_id=lane_id,
        centerline=centerline,
        polygon=polygon,
        speed_limit_mps=speed_limit_mps,
        heading_at_start=heading,
        is_in_intersection=is_lane_connector,
        is_bike_lane=is_bike_lane,
        is_lane_connector=is_lane_connector,
    )


def rectangle_polygon(cx: float, cy: float, length: float, width: float, heading: float = 0.0) -> Sequence[Tuple[float, float]]:
    L2, W2 = length / 2.0, width / 2.0
    cos_h = math.cos(heading)
    sin_h = math.sin(heading)
    corners = [(-L2, -W2), (L2, -W2), (L2, W2), (-L2, W2)]
    return [(cx + cx_l * cos_h - cy_l * sin_h, cy + cx_l * sin_h + cy_l * cos_h) for cx_l, cy_l in corners]


def build_straight_road_scene(
    ego_x: float = 0.0,
    ego_y: float = 0.0,
    ego_speed: float = 10.0,
    ax: float = 0.0,
    ay: float = 0.0,
    yaw_rate: float = 0.0,
    speed_limit_mps: float = 13.4,
    timestamp_us: int = 0,
    agents: Optional[Iterable[AgentSnapshot]] = None,
    extra_lanes: Optional[Iterable[LaneSnapshot]] = None,
    crosswalks: Optional[Iterable[CrosswalkSnapshot]] = None,
    walkways: Optional[Iterable[WalkwaySnapshot]] = None,
    bike_lanes: Optional[Iterable[LaneSnapshot]] = None,
    intersections: Optional[Iterable[IntersectionSnapshot]] = None,
    stop_lines: Optional[Iterable[StopLineSnapshot]] = None,
    lane_connectors: Optional[Iterable[LaneSnapshot]] = None,
    traffic_lights: Optional[Iterable[TrafficLightStatus]] = None,
    route_lane_ids: Optional[Iterable[str]] = None,
) -> SceneSnapshot:
    lane = straight_lane(speed_limit_mps=speed_limit_mps)
    lanes = [lane]
    if extra_lanes is not None:
        lanes.extend(extra_lanes)
    drivable = [DrivableAreaSnapshot(polygon=rectangle_polygon(0, 0, 200.0, 12.0))]
    ego = make_ego(
        x=ego_x,
        y=ego_y,
        speed=ego_speed,
        ax=ax,
        ay=ay,
        yaw_rate=yaw_rate,
        timestamp_us=timestamp_us,
    )
    return SceneSnapshot(
        timestamp_us=timestamp_us,
        ego=ego,
        agents=tuple(agents or []),
        map=MapSnapshot(
            lanes=tuple(lanes),
            lane_connectors=tuple(lane_connectors or []),
            crosswalks=tuple(crosswalks or []),
            stop_lines=tuple(stop_lines or []),
            intersections=tuple(intersections or []),
            drivable_area=tuple(drivable),
            walkways=tuple(walkways or []),
            bike_lanes=tuple(bike_lanes or []),
        ),
        traffic_lights=tuple(traffic_lights or []),
        route_lane_ids=tuple(route_lane_ids) if route_lane_ids else None,
    )
