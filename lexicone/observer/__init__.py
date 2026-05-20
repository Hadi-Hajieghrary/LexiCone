"""Online trajectory rule observer.

Evaluates the ego's trajectory against a NuPlan-applicable subset of the rule
book during simulation or recorded data replay. Each rule reports per-tick
applicability and a violation rate; the observer accumulates these into a
windowed summary.

Public entry points
-------------------
- :class:`TrajectoryObserver` — main per-tick / per-window observer.
- :class:`SceneSnapshot` — input data model (NuPlan-agnostic).
- :class:`RuleEvaluation`, :class:`RuleSummary`, :class:`EpisodeSummary` — outputs.
- :func:`build_default_rules` — returns the 24-rule NuPlan-applicable subset.
- :class:`NuPlanSceneSource` — adapter that converts a NuPlan scenario into
  :class:`SceneSnapshot` objects (nuplan-devkit is imported lazily).
"""

from .types import (
    AgentSnapshot,
    CrosswalkSnapshot,
    DrivableAreaSnapshot,
    EgoSnapshot,
    EpisodeSummary,
    IntersectionSnapshot,
    LaneSnapshot,
    MapSnapshot,
    Pose2D,
    RuleEvaluation,
    RuleSummary,
    SceneSnapshot,
    StopLineSnapshot,
    TrafficLightStatus,
    WalkwaySnapshot,
)
from .rule import ObserverRule
from .observer import TrajectoryObserver
from .registry import build_default_rules

__all__ = [
    "AgentSnapshot",
    "CrosswalkSnapshot",
    "DrivableAreaSnapshot",
    "EgoSnapshot",
    "EpisodeSummary",
    "IntersectionSnapshot",
    "LaneSnapshot",
    "MapSnapshot",
    "ObserverRule",
    "Pose2D",
    "RuleEvaluation",
    "RuleSummary",
    "SceneSnapshot",
    "StopLineSnapshot",
    "TrafficLightStatus",
    "TrajectoryObserver",
    "WalkwaySnapshot",
    "build_default_rules",
]
