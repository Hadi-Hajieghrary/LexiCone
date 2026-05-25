"""Online trajectory rule engine.

Evaluates the ego's trajectory against a NuPlan-applicable subset of the rule
book during simulation or recorded data replay. The :class:`RuleEngine` builds
a per-tick :class:`SceneContext` (vicinity map, ego derivations, semantic
relationships), decides which rules apply at that moment, and summons each
applicable rule's per-tick violation rate. The engine accumulates these into
a windowed :class:`EpisodeSummary`.

Public entry points
-------------------
- :class:`RuleEngine` — main per-tick / per-window engine.
- :class:`SceneSnapshot` — raw input data model (NuPlan-agnostic).
- :class:`SceneContext` — per-tick situational context fed to every rule.
- :class:`ObserverRule` — abstract base class for rules (each in its own
  module under :mod:`lexicone.observer.rules`).
- :class:`RuleEvaluation`, :class:`RuleSummary`, :class:`EpisodeSummary` —
  outputs.
- :func:`build_default_rules` — returns the 25-rule NuPlan-applicable subset.
- :class:`NuPlanSceneSource` — adapter that converts a NuPlan scenario into
  :class:`SceneSnapshot` objects (nuplan-devkit is imported lazily).
"""

from .context import SceneContext
from .engine import RuleEngine
from .registry import build_default_rules
from .rule import ObserverRule
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
    "RuleEngine",
    "RuleEvaluation",
    "RuleSummary",
    "SceneContext",
    "SceneSnapshot",
    "StopLineSnapshot",
    "TrafficLightStatus",
    "WalkwaySnapshot",
    "build_default_rules",
]
