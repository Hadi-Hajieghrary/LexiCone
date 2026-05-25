"""Self-contained closed-loop simulation harness for the lexicone demos.

The :class:`World` holds the static map plus a list of scripted agents whose
state at each tick is a pure function of time. The :func:`simulate` driver
calls a :class:`Planner` once per tick, advances the ego with simple
kinematic integration of the returned command, and emits a stream of
:class:`SceneSnapshot`s — exactly what the rule engine and visualiser
expect.

A demo therefore looks like::

    world = build_my_world()
    planner = IDMPlanner(desired_speed=12.0)
    snapshots = simulate(world, planner, initial_ego=..., n_ticks=80)
    render_episode(engine=RuleEngine(), snapshots=snapshots, ...)

The harness intentionally avoids any nuPlan dependency. It mirrors the
*structure* of closed-loop planning (planner-in-the-loop, world updates,
per-tick context) without the simulator infrastructure, so demos run
anywhere.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Callable, List, Optional, Protocol, Sequence, Tuple, Union

from lexicone.observer import SceneContext
from lexicone.observer.geometry import agent_footprint, ego_footprint
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
    TrafficLightState,
    TrafficLightStatus,
    WalkwaySnapshot,
)


TICK_DT_US = 100_000  # 10 Hz; same cadence as the recorded demos


# --------------------------------------------------------------------------- #
# World + scripted agents
# --------------------------------------------------------------------------- #


# A ScriptedAgent is a function from elapsed seconds to an AgentSnapshot.
ScriptedAgent = Callable[[float], AgentSnapshot]


def constant_velocity_agent(initial: AgentSnapshot) -> ScriptedAgent:
    """A scripted agent that moves at the initial linear velocity forever."""

    def at(t_s: float) -> AgentSnapshot:
        return replace(
            initial,
            pose=Pose2D(
                x=initial.pose.x + initial.vx * t_s,
                y=initial.pose.y + initial.vy * t_s,
                heading=initial.pose.heading,
            ),
        )

    return at


def static_agent(initial: AgentSnapshot) -> ScriptedAgent:
    """A scripted agent that holds its initial pose."""
    return lambda _t: initial


def crossing_pedestrian(
    *,
    track_id: str,
    x: float,
    y_start: float,
    y_end: float,
    speed_mps: float = 1.4,
    t_start_s: float = 0.0,
    length: float = 0.6,
    width: float = 0.6,
    object_type: AgentType = AgentType.PEDESTRIAN,
) -> ScriptedAgent:
    """Pedestrian that waits on the curb until ``t_start_s``, walks at a
    constant lateral speed from ``y_start`` to ``y_end``, then stops on the
    far sidewalk.

    The pedestrian's heading is set to the direction of travel, so the
    visualiser shows the body box oriented correctly.
    """
    duration = abs(y_end - y_start) / max(speed_mps, 1e-3)
    direction = 1.0 if y_end > y_start else -1.0
    heading = math.pi / 2.0 if direction > 0 else -math.pi / 2.0

    def at(t_s: float) -> AgentSnapshot:
        if t_s <= t_start_s:
            y = y_start
            vy = 0.0
        elif t_s >= t_start_s + duration:
            y = y_end
            vy = 0.0
        else:
            y = y_start + direction * speed_mps * (t_s - t_start_s)
            vy = direction * speed_mps
        return AgentSnapshot(
            track_id=track_id,
            object_type=object_type,
            pose=Pose2D(x=x, y=y, heading=heading),
            vx=0.0,
            vy=vy,
            length=length,
            width=width,
        )

    return at


# Either a static traffic-light tuple, or a function of time returning one.
TrafficLightSchedule = Union[
    Tuple[TrafficLightStatus, ...], Callable[[float], Tuple[TrafficLightStatus, ...]]
]


@dataclass
class World:
    """Static map + scripted dynamic state for the simulation.

    The map (lanes / drivable / crosswalks / …) is fixed for the whole
    episode. ``scripted_agents`` are pure functions of elapsed seconds, so
    pedestrians and other vehicles can follow arbitrary trajectories.
    ``traffic_lights`` is either a static tuple or a function of time —
    pass a function to model a cycling signal head.
    """

    lanes: Tuple[LaneSnapshot, ...] = ()
    lane_connectors: Tuple[LaneSnapshot, ...] = ()
    crosswalks: Tuple[CrosswalkSnapshot, ...] = ()
    stop_lines: Tuple[StopLineSnapshot, ...] = ()
    intersections: Tuple[IntersectionSnapshot, ...] = ()
    drivable_area: Tuple[DrivableAreaSnapshot, ...] = ()
    walkways: Tuple[WalkwaySnapshot, ...] = ()
    bike_lanes: Tuple[LaneSnapshot, ...] = ()
    traffic_lights: TrafficLightSchedule = ()
    scripted_agents: Tuple[ScriptedAgent, ...] = ()
    route_lane_ids: Optional[Tuple[str, ...]] = None

    def map_snapshot(self) -> MapSnapshot:
        return MapSnapshot(
            lanes=self.lanes,
            lane_connectors=self.lane_connectors,
            crosswalks=self.crosswalks,
            stop_lines=self.stop_lines,
            intersections=self.intersections,
            drivable_area=self.drivable_area,
            walkways=self.walkways,
            bike_lanes=self.bike_lanes,
        )

    def agents_at(self, t_s: float) -> Tuple[AgentSnapshot, ...]:
        return tuple(fn(t_s) for fn in self.scripted_agents)

    def traffic_lights_at(self, t_s: float) -> Tuple[TrafficLightStatus, ...]:
        if callable(self.traffic_lights):
            return tuple(self.traffic_lights(t_s))
        return tuple(self.traffic_lights)


def traffic_light_cycle(
    connector_id: str,
    *,
    green_s: float = 4.0,
    yellow_s: float = 1.0,
    red_s: float = 4.0,
    phase_offset_s: float = 0.0,
) -> Callable[[float], Tuple[TrafficLightStatus, ...]]:
    """Return a scheduler producing a green→yellow→red cycle for one connector.

    Use ``phase_offset_s`` to stagger cross-direction signals. For a 4-way
    intersection, west/east and north/south usually run 180° out of phase
    (offset by ``green_s + yellow_s``).
    """
    period = green_s + yellow_s + red_s

    def at(t_s: float) -> Tuple[TrafficLightStatus, ...]:
        phase = (t_s + phase_offset_s) % period
        if phase < green_s:
            state = TrafficLightState.GREEN
        elif phase < green_s + yellow_s:
            state = TrafficLightState.YELLOW
        else:
            state = TrafficLightState.RED
        return (TrafficLightStatus(lane_connector_id=connector_id, state=state),)

    return at


def combine_traffic_light_schedules(
    *schedules: Callable[[float], Tuple[TrafficLightStatus, ...]],
) -> Callable[[float], Tuple[TrafficLightStatus, ...]]:
    """Combine several connector-level schedules into one world-level schedule."""

    def at(t_s: float) -> Tuple[TrafficLightStatus, ...]:
        out: List[TrafficLightStatus] = []
        for s in schedules:
            out.extend(s(t_s))
        return tuple(out)

    return at


# --------------------------------------------------------------------------- #
# Planner protocol + command type
# --------------------------------------------------------------------------- #


@dataclass
class PlannerCommand:
    """What a planner returns each tick.

    ``ax_mps2`` is longitudinal acceleration in the ego frame (positive =
    forward). ``yaw_rate_radps`` is rotation about the vertical axis (positive
    = left turn). ``planned_trajectory`` is the planner's intended future
    state sequence (used purely for visualisation; the simulator advances the
    ego from the command, not from this trajectory).
    """

    ax_mps2: float = 0.0
    yaw_rate_radps: float = 0.0
    planned_trajectory: Sequence[EgoSnapshot] = field(default_factory=tuple)


class Planner(Protocol):
    name: str

    def reset(self) -> None: ...
    def plan(self, ctx: SceneContext) -> PlannerCommand: ...


# --------------------------------------------------------------------------- #
# Kinematic integration
# --------------------------------------------------------------------------- #


def _advance_ego(ego: EgoSnapshot, cmd: PlannerCommand, dt_s: float) -> EgoSnapshot:
    """Bicycle-ish kinematic step: integrate yaw, then speed, then position."""
    new_heading = ego.pose.heading + cmd.yaw_rate_radps * dt_s
    speed = max(0.0, ego.speed + cmd.ax_mps2 * dt_s)
    cos_h = math.cos(new_heading)
    sin_h = math.sin(new_heading)
    # Use mid-point (heading) for less drift on turns.
    mid_cos = math.cos((ego.pose.heading + new_heading) / 2.0)
    mid_sin = math.sin((ego.pose.heading + new_heading) / 2.0)
    mid_speed = (ego.speed + speed) / 2.0
    new_x = ego.pose.x + mid_speed * mid_cos * dt_s
    new_y = ego.pose.y + mid_speed * mid_sin * dt_s
    # Lateral acceleration approx (centripetal).
    ay = mid_speed * cmd.yaw_rate_radps
    return EgoSnapshot(
        timestamp_us=ego.timestamp_us + int(dt_s * 1e6),
        pose=Pose2D(x=new_x, y=new_y, heading=new_heading),
        vx=speed * cos_h,
        vy=speed * sin_h,
        ax=cmd.ax_mps2,
        ay=ay,
        yaw_rate=cmd.yaw_rate_radps,
        length=ego.length,
        width=ego.width,
        rear_axle_to_center=ego.rear_axle_to_center,
        pose_at_center=ego.pose_at_center,
        turn_signal=ego.turn_signal,
    )


def project_forward(
    ego: EgoSnapshot,
    cmd: PlannerCommand,
    horizon_s: float = 4.0,
    n_steps: int = 16,
) -> List[EgoSnapshot]:
    """Cheap straight-line forward projection used as the planned trajectory.

    Real planners would produce something curvy and reactive; for the demo
    visualiser we just propagate the command kinematically. Useful as both
    a sanity-check and a reference for the visualiser's dashed line.
    """
    out: List[EgoSnapshot] = []
    state = ego
    dt = horizon_s / max(1, n_steps)
    for _ in range(n_steps):
        state = _advance_ego(state, cmd, dt)
        out.append(state)
    return out


# --------------------------------------------------------------------------- #
# Closed-loop driver
# --------------------------------------------------------------------------- #


def initial_ego(
    *,
    x: float = 0.0,
    y: float = 0.0,
    heading: float = 0.0,
    speed: float = 5.0,
    length: float = 4.7,
    width: float = 1.85,
) -> EgoSnapshot:
    return EgoSnapshot(
        timestamp_us=0,
        pose=Pose2D(x=x, y=y, heading=heading),
        vx=speed * math.cos(heading),
        vy=speed * math.sin(heading),
        ax=0.0,
        ay=0.0,
        yaw_rate=0.0,
        length=length,
        width=width,
        rear_axle_to_center=0.0,
        pose_at_center=True,
    )


def _ego_collides_with_any(
    ego: EgoSnapshot, agents: Sequence[AgentSnapshot]
) -> Optional[AgentSnapshot]:
    """Return the first agent whose footprint overlaps ego's, or ``None``."""
    fp = ego_footprint(ego)
    for a in agents:
        if fp.intersects(agent_footprint(a)):
            return a
    return None


def simulate(
    world: World,
    planner: Planner,
    initial: EgoSnapshot,
    n_ticks: int,
    dt_s: float = TICK_DT_US * 1e-6,
    *,
    halt_on_collision: bool = True,
) -> List[SceneSnapshot]:
    """Drive ``planner`` over ``world`` for ``n_ticks`` ticks at ``dt_s``.

    Returns one :class:`SceneSnapshot` per tick. Each snapshot carries the
    planner's intended forward trajectory in ``planned_trajectory``, which
    the visualiser draws as the dashed line.

    Collision handling: when ``halt_on_collision`` is True (the default), the
    simulator detects ego-agent footprint overlap and **freezes the ego** at
    that pose for the rest of the episode (zero command applied, timestamp
    still advances). The rule engine keeps evaluating each tick — collision
    and headway rules will continue to flag, which is the correct outcome.
    Pass ``halt_on_collision=False`` to let the ego phase through agents (the
    old behaviour, useful for stress-testing rules under impossible states).
    """
    planner.reset()

    snapshots: List[SceneSnapshot] = []
    ego = initial
    collided: bool = False
    # When collision happens, freeze BOTH vehicles at their contact poses.
    # The ego is frozen via the ``collided`` flag; the leader is frozen by
    # overriding ``agents_now`` with this dict for the rest of the episode.
    frozen_agents: dict[str, AgentSnapshot] = {}
    for k in range(n_ticks):
        t_s = k * dt_s
        agents_now = world.agents_at(t_s)
        if frozen_agents:
            # Replace any frozen agent's scripted state with its contact pose,
            # but keep agents that weren't part of the collision moving.
            agents_now = tuple(frozen_agents.get(a.track_id, a) for a in agents_now)
        snap_pre = SceneSnapshot(
            timestamp_us=ego.timestamp_us,
            ego=ego,
            agents=agents_now,
            map=world.map_snapshot(),
            traffic_lights=world.traffic_lights_at(t_s),
            planned_trajectory=None,  # filled in below
            route_lane_ids=world.route_lane_ids,
        )
        if collided:
            # Frozen ego: no planner call, no motion; emit snapshot as-is.
            snapshots.append(replace(snap_pre, planned_trajectory=()))
            ego = replace(
                ego,
                timestamp_us=ego.timestamp_us + int(dt_s * 1e6),
                ax=0.0,
                ay=0.0,
                yaw_rate=0.0,
                vx=0.0,
                vy=0.0,
            )
            continue

        ctx = SceneContext(snap_pre)
        cmd = planner.plan(ctx)
        planned = list(cmd.planned_trajectory) if cmd.planned_trajectory else project_forward(ego, cmd)
        snapshots.append(replace(snap_pre, planned_trajectory=planned))

        # Advance ego, then check collision against the *next* tick's agents.
        next_ego = _advance_ego(ego, cmd, dt_s)
        if halt_on_collision:
            next_agents = world.agents_at((k + 1) * dt_s)
            hit = _ego_collides_with_any(next_ego, next_agents)
            if hit is not None:
                # Freeze ego with a hard decel impulse (so 0r2 flags impact).
                ego = replace(
                    next_ego,
                    vx=0.0,
                    vy=0.0,
                    ax=-min(next_ego.speed / max(dt_s, 1e-3), 12.0),
                    ay=0.0,
                    yaw_rate=0.0,
                )
                # Freeze the colliding agent at its contact pose with zero
                # velocity so it does not drive away from the impact site.
                frozen_agents[hit.track_id] = replace(hit, vx=0.0, vy=0.0)
                collided = True
                continue
        ego = next_ego
    return snapshots
