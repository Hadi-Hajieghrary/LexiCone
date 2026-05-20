"""Data model for the trajectory observer.

The types here are intentionally NuPlan-agnostic so the observer can be unit-
tested without nuplan-devkit installed. ``NuPlanSceneSource`` converts NuPlan
scenarios into these snapshots.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple


class AgentType(str, Enum):
    VEHICLE = "VEHICLE"
    PEDESTRIAN = "PEDESTRIAN"
    BICYCLE = "BICYCLE"
    MOTORCYCLE = "MOTORCYCLE"
    TRAFFIC_CONE = "TRAFFIC_CONE"
    BARRIER = "BARRIER"
    CZONE_SIGN = "CZONE_SIGN"
    GENERIC_OBJECT = "GENERIC_OBJECT"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def is_vru(cls, t: "AgentType | str") -> bool:
        s = t.value if isinstance(t, AgentType) else str(t)
        return s in {cls.PEDESTRIAN.value, cls.BICYCLE.value}


class TrafficLightState(str, Enum):
    RED = "RED"
    YELLOW = "YELLOW"
    GREEN = "GREEN"
    UNKNOWN = "UNKNOWN"


class StopType(str, Enum):
    STOP_SIGN = "STOP_SIGN"
    TRAFFIC_LIGHT = "TRAFFIC_LIGHT"
    YIELD_SIGN = "YIELD_SIGN"
    GENERIC = "GENERIC"


@dataclass
class Pose2D:
    """A 2-D pose in world frame. Heading is in radians, CCW from +x."""

    x: float
    y: float
    heading: float


@dataclass
class EgoSnapshot:
    """Ego state at one instant.

    All quantities are in the world frame unless noted. ``length`` and ``width``
    are the bounding box. ``pose`` refers to the rear-axle center by default;
    set ``pose_at_center=True`` if the pose is at the geometric center.
    """

    timestamp_us: int
    pose: Pose2D
    vx: float
    vy: float
    ax: float = 0.0
    ay: float = 0.0
    yaw_rate: float = 0.0
    length: float = 4.7
    width: float = 1.85
    rear_axle_to_center: float = 1.46  # NuPlan ego default
    pose_at_center: bool = False
    # Optional, populated by adapters when available.
    turn_signal: Optional[str] = None  # 'LEFT', 'RIGHT', 'OFF'

    @property
    def speed(self) -> float:
        return (self.vx * self.vx + self.vy * self.vy) ** 0.5


@dataclass
class AgentSnapshot:
    """A tracked dynamic object at one instant."""

    track_id: str
    object_type: AgentType
    pose: Pose2D
    vx: float = 0.0
    vy: float = 0.0
    length: float = 4.5
    width: float = 1.8

    @property
    def speed(self) -> float:
        return (self.vx * self.vx + self.vy * self.vy) ** 0.5


# Polygon = closed sequence of (x, y). LineString = open sequence.
Polygon2D = Sequence[Tuple[float, float]]
Polyline2D = Sequence[Tuple[float, float]]


@dataclass
class LaneSnapshot:
    lane_id: str
    centerline: Polyline2D
    polygon: Polygon2D
    speed_limit_mps: Optional[float] = None
    heading_at_start: Optional[float] = None  # radians; computed from centerline if None
    is_in_intersection: bool = False
    is_bike_lane: bool = False
    is_lane_connector: bool = False
    incoming_lane_ids: Tuple[str, ...] = ()
    outgoing_lane_ids: Tuple[str, ...] = ()


@dataclass
class CrosswalkSnapshot:
    crosswalk_id: str
    polygon: Polygon2D
    is_marked: bool = True


@dataclass
class StopLineSnapshot:
    stop_line_id: str
    polyline: Polyline2D
    stop_type: StopType = StopType.GENERIC
    associated_lane_id: Optional[str] = None


@dataclass
class IntersectionSnapshot:
    intersection_id: str
    polygon: Polygon2D
    is_signalized: bool = False


@dataclass
class DrivableAreaSnapshot:
    polygon: Polygon2D


@dataclass
class WalkwaySnapshot:
    walkway_id: str
    polygon: Polygon2D  # sidewalk / pedestrian-only surface


@dataclass
class TrafficLightStatus:
    """Status of a traffic-light controlled lane connector."""

    lane_connector_id: str
    state: TrafficLightState = TrafficLightState.UNKNOWN


@dataclass
class MapSnapshot:
    """All map elements within the observer's region of interest at this tick."""

    lanes: Sequence[LaneSnapshot] = field(default_factory=tuple)
    lane_connectors: Sequence[LaneSnapshot] = field(default_factory=tuple)
    crosswalks: Sequence[CrosswalkSnapshot] = field(default_factory=tuple)
    stop_lines: Sequence[StopLineSnapshot] = field(default_factory=tuple)
    intersections: Sequence[IntersectionSnapshot] = field(default_factory=tuple)
    drivable_area: Sequence[DrivableAreaSnapshot] = field(default_factory=tuple)
    walkways: Sequence[WalkwaySnapshot] = field(default_factory=tuple)
    bike_lanes: Sequence[LaneSnapshot] = field(default_factory=tuple)


@dataclass
class SceneSnapshot:
    """A scene at a single instant: ego, agents, nearby map, signals.

    ``planned_trajectory`` is the planner's intended future (≥0 states starting
    at or after ``ego``); ``None`` when not available (e.g. open-loop replay).
    ``route_lane_ids`` are the planned route lanes used by route-adherence
    rules.
    """

    timestamp_us: int
    ego: EgoSnapshot
    agents: Sequence[AgentSnapshot] = field(default_factory=tuple)
    map: MapSnapshot = field(default_factory=MapSnapshot)
    traffic_lights: Sequence[TrafficLightStatus] = field(default_factory=tuple)
    planned_trajectory: Optional[Sequence[EgoSnapshot]] = None
    route_lane_ids: Optional[Sequence[str]] = None
    # Optional extra context (weather, sensor health, etc.) for rule overrides.
    extras: Mapping[str, Any] = field(default_factory=dict)

    @property
    def timestamp_s(self) -> float:
        return self.timestamp_us * 1e-6


@dataclass
class RuleEvaluation:
    """Per-tick result for one rule."""

    rule_id: str
    rule_level: int
    rule_name: str
    timestamp_us: int
    applies: bool
    violation_rate: float  # >= 0; 0 when not applicable or fully compliant
    is_violated: bool
    details: Mapping[str, Any] = field(default_factory=dict)

    @property
    def timestamp_s(self) -> float:
        return self.timestamp_us * 1e-6


@dataclass
class RuleSummary:
    """Summary of one rule's behavior over a window of ticks."""

    rule_id: str
    rule_level: int
    rule_name: str
    n_steps_total: int
    n_steps_applicable: int
    n_steps_violated: int
    duration_applicable_s: float
    integrated_violation: float  # ∫ violation_rate · dt
    max_violation_rate: float
    first_violation_t_s: Optional[float]
    last_violation_t_s: Optional[float]

    @property
    def fraction_applicable(self) -> float:
        return self.n_steps_applicable / self.n_steps_total if self.n_steps_total else 0.0

    @property
    def fraction_violated_when_applicable(self) -> float:
        return (
            self.n_steps_violated / self.n_steps_applicable
            if self.n_steps_applicable
            else 0.0
        )


@dataclass
class EpisodeSummary:
    """Aggregate over all rules for a window of ticks."""

    rule_summaries: Mapping[str, RuleSummary]
    duration_s: float
    n_steps: int
    window_start_us: int
    window_end_us: int

    def violated_rules(self) -> Iterable[RuleSummary]:
        return (s for s in self.rule_summaries.values() if s.n_steps_violated > 0)

    def by_level(self) -> Mapping[int, Sequence[RuleSummary]]:
        out: dict[int, list[RuleSummary]] = {}
        for s in self.rule_summaries.values():
            out.setdefault(s.rule_level, []).append(s)
        return out
