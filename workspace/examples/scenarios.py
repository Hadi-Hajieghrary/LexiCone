"""Richer scene builders for the example demos.

These helpers go beyond ``tests/synthetic.py``: they produce full episodes (a
sequence of :class:`SceneSnapshot` per tick) and populate every map layer the
rule engine reads — drivable area, lanes, lane connectors, intersections,
crosswalks, walkways, bike lanes, stop lines, and traffic-light statuses.

Each builder returns an :class:`Episode` dataclass (``snapshots`` and
``scenario_name``) so the visualiser can label the output. Snapshots carry
``route_lane_ids`` so the route-adherence rule has something to evaluate;
``planned_trajectory`` is left ``None`` on these scripted episodes and only
populated by demos that plug in a real planner (e.g. via ``simulate(...)``
in :mod:`examples.simulation`, which threads the planner's command-side
``planned_trajectory`` through each snapshot).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

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


TICK_DT_US = 100_000  # 10 Hz


# --------------------------------------------------------------------------- #
# Low-level geometry helpers
# --------------------------------------------------------------------------- #


def _rect(cx: float, cy: float, length: float, width: float, heading: float = 0.0):
    """Oriented rectangle as a closed polygon (4 vertices)."""
    L2, W2 = length / 2.0, width / 2.0
    cos_h, sin_h = math.cos(heading), math.sin(heading)
    return [
        (cx + dx * cos_h - dy * sin_h, cy + dx * sin_h + dy * cos_h)
        for dx, dy in [(-L2, -W2), (L2, -W2), (L2, W2), (-L2, W2)]
    ]


def _straight_lane(
    lane_id: str,
    *,
    x_center: float = 0.0,
    y_center: float = 0.0,
    length: float = 200.0,
    width: float = 3.5,
    heading: float = 0.0,
    speed_limit_mps: Optional[float] = 13.4,
    is_bike_lane: bool = False,
    is_lane_connector: bool = False,
    incoming: Tuple[str, ...] = (),
    outgoing: Tuple[str, ...] = (),
) -> LaneSnapshot:
    cos_h, sin_h = math.cos(heading), math.sin(heading)

    def offset(dx: float, dy: float):
        return (x_center + dx * cos_h - dy * sin_h, y_center + dx * sin_h + dy * cos_h)

    half = length / 2.0
    centerline = [offset(-half, 0.0), offset(half, 0.0)]
    polygon = _rect(x_center, y_center, length, width, heading)
    return LaneSnapshot(
        lane_id=lane_id,
        centerline=centerline,
        polygon=polygon,
        speed_limit_mps=speed_limit_mps,
        heading_at_start=heading,
        is_in_intersection=is_lane_connector,
        is_bike_lane=is_bike_lane,
        is_lane_connector=is_lane_connector,
        incoming_lane_ids=incoming,
        outgoing_lane_ids=outgoing,
    )


# --------------------------------------------------------------------------- #
# Ego / agent helpers
# --------------------------------------------------------------------------- #


def _ego_at(
    *,
    timestamp_us: int,
    x: float,
    y: float,
    heading: float,
    speed: float,
    ax: float = 0.0,
    ay: float = 0.0,
    yaw_rate: float = 0.0,
) -> EgoSnapshot:
    return EgoSnapshot(
        timestamp_us=timestamp_us,
        pose=Pose2D(x=x, y=y, heading=heading),
        vx=speed * math.cos(heading),
        vy=speed * math.sin(heading),
        ax=ax,
        ay=ay,
        yaw_rate=yaw_rate,
        length=4.7,
        width=1.85,
        rear_axle_to_center=0.0,
        pose_at_center=True,
    )


def _agent(
    track_id: str,
    object_type: AgentType,
    *,
    x: float,
    y: float,
    heading: float = 0.0,
    speed: float = 0.0,
    length: Optional[float] = None,
    width: Optional[float] = None,
) -> AgentSnapshot:
    defaults = {
        AgentType.VEHICLE: (4.5, 1.8),
        AgentType.MOTORCYCLE: (2.2, 0.9),
        AgentType.BICYCLE: (1.7, 0.6),
        AgentType.PEDESTRIAN: (0.6, 0.6),
    }.get(object_type, (1.0, 1.0))
    return AgentSnapshot(
        track_id=track_id,
        object_type=object_type,
        pose=Pose2D(x=x, y=y, heading=heading),
        vx=speed * math.cos(heading),
        vy=speed * math.sin(heading),
        length=length if length is not None else defaults[0],
        width=width if width is not None else defaults[1],
    )


# --------------------------------------------------------------------------- #
# Episode builders
# --------------------------------------------------------------------------- #


@dataclass
class Episode:
    """An ordered series of snapshots plus a friendly scenario name."""

    name: str
    snapshots: List[SceneSnapshot]

    def __len__(self) -> int:
        return len(self.snapshots)


def intersection_red_light_episode(n_ticks: int = 60) -> Episode:
    """Ego approaches a signalised intersection and rolls through the red.

    Map: one east-bound lane → lane connector through a square intersection
    → one east-bound exit lane, plus a north-bound cross lane. A marked
    crosswalk sits at the intersection entry. The connector is showing RED;
    a stop polyline is associated with the entry lane. Two cross-traffic
    vehicles wait. The ego decelerates only modestly and crosses the line.
    """
    # Map layout
    entry = _straight_lane(
        "entry", x_center=-15.0, y_center=0.0, length=30.0, width=3.5, outgoing=("connector",)
    )
    exit_ = _straight_lane(
        "exit", x_center=15.0, y_center=0.0, length=30.0, width=3.5, incoming=("connector",)
    )
    connector = _straight_lane(
        "connector",
        x_center=0.0,
        y_center=0.0,
        length=10.0,
        width=3.5,
        is_lane_connector=True,
        incoming=("entry",),
        outgoing=("exit",),
    )
    cross_lane = _straight_lane(
        "cross", x_center=0.0, y_center=0.0, length=30.0, width=3.5, heading=math.pi / 2.0
    )
    intersection = IntersectionSnapshot(
        intersection_id="i1", polygon=_rect(0.0, 0.0, 10.0, 10.0), is_signalized=True
    )
    crosswalk = CrosswalkSnapshot(crosswalk_id="cw1", polygon=_rect(-4.0, 0.0, 1.5, 8.0))
    stop_line = StopLineSnapshot(
        stop_line_id="sl1",
        polyline=[(-5.0, -2.0), (-5.0, 2.0)],
        stop_type=StopType.TRAFFIC_LIGHT,
        associated_lane_id="entry",
    )
    walkway_north = WalkwaySnapshot("ww_n", polygon=_rect(0.0, 9.0, 30.0, 4.0))
    walkway_south = WalkwaySnapshot("ww_s", polygon=_rect(0.0, -9.0, 30.0, 4.0))
    # Drivable area = main horizontal corridor ∪ cross-road corridor, so the
    # ego stays in-bounds along the full route and only the intersection rules
    # fire.
    drivable_main = DrivableAreaSnapshot(polygon=_rect(0.0, 0.0, 120.0, 14.0))
    drivable_cross = DrivableAreaSnapshot(polygon=_rect(0.0, 0.0, 14.0, 30.0))

    # Cross-traffic waiting on the perpendicular axis, far enough away that
    # they don't trip 1r0's priority-zone check on approach.
    cross_a = _agent("cross_a", AgentType.VEHICLE, x=0.0, y=-18.0, heading=math.pi / 2.0, speed=0.0)
    cross_b = _agent("cross_b", AgentType.VEHICLE, x=0.0, y=18.0, heading=-math.pi / 2.0, speed=0.0)
    static_agents = [cross_a, cross_b]

    snapshots: List[SceneSnapshot] = []
    # Ego starts 25 m before the stop line, decelerating mildly but not enough.
    for k in range(n_ticks):
        t_s = k * TICK_DT_US * 1e-6
        # Speed profile: 14 m/s, light decel after t=2 s, ax = -1.5 m/s^2.
        if t_s < 2.0:
            v = 14.0
            ax = 0.0
        else:
            v = max(4.0, 14.0 - 1.5 * (t_s - 2.0))
            ax = -1.5
        # Position: integrate v over time approximately.
        # Use trapezoidal integration for ~consistency with v(t).
        if k == 0:
            x = -25.0
        else:
            prev = snapshots[-1].ego
            dt = TICK_DT_US * 1e-6
            x = prev.pose.x + prev.speed * dt + 0.5 * prev.ax * dt * dt
        ego = _ego_at(timestamp_us=k * TICK_DT_US, x=x, y=0.0, heading=0.0, speed=v, ax=ax)
        snap = SceneSnapshot(
            timestamp_us=k * TICK_DT_US,
            ego=ego,
            agents=tuple(static_agents),
            map=MapSnapshot(
                lanes=(entry, exit_, cross_lane),
                lane_connectors=(connector,),
                crosswalks=(crosswalk,),
                stop_lines=(stop_line,),
                intersections=(intersection,),
                drivable_area=(drivable_main, drivable_cross),
                walkways=(walkway_north, walkway_south),
                bike_lanes=(),
            ),
            traffic_lights=(
                TrafficLightStatus(lane_connector_id="connector", state=TrafficLightState.RED),
            ),
            route_lane_ids=("entry", "connector", "exit"),
        )
        snapshots.append(snap)

    return Episode(name="intersection_red_light", snapshots=snapshots)


def cyclist_overtake_episode(n_ticks: int = 50) -> Episode:
    """Ego overtakes a slow cyclist with a marginal lateral gap.

    Map: two parallel travel lanes (vehicle + bike), 30 mph limit, sidewalks
    on both sides. A cyclist rides in the bike lane; the ego drifts toward
    the bike lane while passing, exceeding the speed limit slightly.
    """
    vehicle_lane = _straight_lane("v_lane", y_center=0.0, length=300.0, width=3.5, speed_limit_mps=11.0)
    bike_lane = _straight_lane(
        "b_lane", y_center=2.5, length=300.0, width=1.5, speed_limit_mps=None, is_bike_lane=True
    )
    walkway_south = WalkwaySnapshot("ww_s", polygon=_rect(0.0, -4.0, 300.0, 3.0))
    walkway_north = WalkwaySnapshot("ww_n", polygon=_rect(0.0, 5.0, 300.0, 3.0))
    drivable = DrivableAreaSnapshot(polygon=_rect(0.0, 1.0, 300.0, 7.0))

    snapshots: List[SceneSnapshot] = []
    # Cyclist at constant 3 m/s, starts at x=20.
    for k in range(n_ticks):
        dt = TICK_DT_US * 1e-6
        cyclist_x = 20.0 + 3.0 * (k * dt)
        cyclist = _agent("c1", AgentType.BICYCLE, x=cyclist_x, y=2.5, heading=0.0, speed=3.0)
        # Ego: start at x=0, speed 12.5 (above 11 m/s limit). Drift toward bike lane.
        ego_speed = 12.5
        if k == 0:
            ego_x = 0.0
            ego_y = 0.0
        else:
            prev = snapshots[-1].ego
            ego_x = prev.pose.x + ego_speed * dt
            # Drift 0.08 m/s toward y=2.5 until close, then hold.
            target_y = 1.5
            dy = target_y - prev.pose.y
            ego_y = prev.pose.y + max(-0.08 * dt * 10.0, min(0.08 * dt * 10.0, dy))
        ego = _ego_at(
            timestamp_us=k * TICK_DT_US,
            x=ego_x,
            y=ego_y,
            heading=0.0,
            speed=ego_speed,
            ay=0.4,
        )
        snap = SceneSnapshot(
            timestamp_us=k * TICK_DT_US,
            ego=ego,
            agents=(cyclist,),
            map=MapSnapshot(
                lanes=(vehicle_lane,),
                lane_connectors=(),
                crosswalks=(),
                stop_lines=(),
                intersections=(),
                drivable_area=(drivable,),
                walkways=(walkway_south, walkway_north),
                bike_lanes=(bike_lane,),
            ),
            traffic_lights=(),
            route_lane_ids=("v_lane",),
        )
        snapshots.append(snap)

    return Episode(name="cyclist_overtake", snapshots=snapshots)


def random_episode(seed: int, n_ticks: int = 60) -> Episode:
    """Randomly populated scene around a straight road.

    Deterministic for a given ``seed``. The scene always has a drivable
    rectangle and one travel lane (so 9r1, 7r0, 3r0 can apply), and the RNG
    optionally adds: crosswalks, walkways, a bike lane with a cyclist,
    nearby pedestrians, a leader vehicle, and a lateral neighbour. Ego
    behaviour (target speed, lateral drift, acceleration profile) is also
    randomised, so most runs end up tripping a different mix of rules.
    """
    rng = random.Random(seed)

    speed_limit = rng.choice([8.94, 11.18, 13.41, 17.88])  # 20/25/30/40 mph
    main_lane = _straight_lane("main", length=300.0, width=3.5, speed_limit_mps=speed_limit)
    drivable = DrivableAreaSnapshot(polygon=_rect(0.0, 0.0, 300.0, 12.0))

    extras_lanes: List[LaneSnapshot] = []
    extras_walkways: List[WalkwaySnapshot] = []
    extras_crosswalks: List[CrosswalkSnapshot] = []
    extras_bike: List[LaneSnapshot] = []

    if rng.random() < 0.7:
        extras_walkways.append(WalkwaySnapshot("ww_n", polygon=_rect(0.0, 7.0, 300.0, 3.0)))
        extras_walkways.append(WalkwaySnapshot("ww_s", polygon=_rect(0.0, -7.0, 300.0, 3.0)))

    has_bike_lane = rng.random() < 0.5
    if has_bike_lane:
        extras_bike.append(
            _straight_lane(
                "bike",
                y_center=2.6,
                length=300.0,
                width=1.5,
                speed_limit_mps=None,
                is_bike_lane=True,
            )
        )

    if rng.random() < 0.6:
        cw_x = rng.uniform(40.0, 80.0)
        extras_crosswalks.append(CrosswalkSnapshot(f"cw_{cw_x:.0f}", polygon=_rect(cw_x, 0.0, 2.0, 8.0)))

    agents_template: List[Tuple[str, AgentType, float, float, float, float]] = []
    if rng.random() < 0.7:
        agents_template.append(
            ("leader", AgentType.VEHICLE, rng.uniform(20.0, 50.0), 0.0, 0.0, rng.uniform(3.0, 8.0))
        )
    if rng.random() < 0.4:
        agents_template.append(
            ("neighbour", AgentType.VEHICLE, rng.uniform(-2.0, 4.0), rng.choice([-3.0, 3.0]), 0.0, rng.uniform(5.0, 9.0))
        )
    if has_bike_lane and rng.random() < 0.7:
        agents_template.append(
            ("bike", AgentType.BICYCLE, rng.uniform(15.0, 40.0), 2.6, 0.0, rng.uniform(2.0, 4.0))
        )
    if rng.random() < 0.5:
        agents_template.append(
            ("ped", AgentType.PEDESTRIAN, rng.uniform(50.0, 90.0), rng.uniform(-2.5, 2.5), 0.0, 0.0)
        )

    target_speed = rng.uniform(speed_limit * 0.8, speed_limit * 1.4)
    target_y = rng.uniform(-1.0, 2.0) if has_bike_lane else rng.uniform(-1.0, 1.0)
    ax_profile = rng.uniform(-2.5, 2.5)

    snapshots: List[SceneSnapshot] = []
    for k in range(n_ticks):
        dt = TICK_DT_US * 1e-6
        if k == 0:
            ego_x, ego_y, ego_v = 0.0, 0.0, target_speed
        else:
            prev = snapshots[-1].ego
            ego_v = max(0.0, prev.speed + ax_profile * dt)
            ego_x = prev.pose.x + prev.speed * dt
            dy = target_y - prev.pose.y
            step = max(-0.5 * dt, min(0.5 * dt, dy))
            ego_y = prev.pose.y + step
        ego = _ego_at(
            timestamp_us=k * TICK_DT_US,
            x=ego_x,
            y=ego_y,
            heading=0.0,
            speed=ego_v,
            ax=ax_profile,
        )

        # Advance scripted agents.
        agents: List[AgentSnapshot] = []
        for tid, otype, x0, y0, heading, speed in agents_template:
            agents.append(_agent(tid, otype, x=x0 + speed * (k * dt), y=y0, heading=heading, speed=speed))

        snap = SceneSnapshot(
            timestamp_us=k * TICK_DT_US,
            ego=ego,
            agents=tuple(agents),
            map=MapSnapshot(
                lanes=(main_lane,),
                lane_connectors=(),
                crosswalks=tuple(extras_crosswalks),
                stop_lines=(),
                intersections=(),
                drivable_area=(drivable,),
                walkways=tuple(extras_walkways),
                bike_lanes=tuple(extras_bike),
            ),
            traffic_lights=(),
            route_lane_ids=("main",),
        )
        snapshots.append(snap)

    return Episode(name=f"random_seed{seed}", snapshots=snapshots)


# --------------------------------------------------------------------------- #
# Convenience: registry of named builders for the demos
# --------------------------------------------------------------------------- #


SCENARIO_BUILDERS = {
    "intersection_red_light": intersection_red_light_episode,
    "cyclist_overtake": cyclist_overtake_episode,
    "random": random_episode,
}
