"""Default rule set for the trajectory observer.

24 rules, drawn from the README's NuPlan-applicable subset.
"""

from __future__ import annotations

from typing import List

from .rule import ObserverRule
from .rules.comfort import (
    LateralAccelerationRule,
    LateralComfortRule,
    LongitudinalComfortRule,
)
from .rules.crosswalk_rules import (
    CrosswalkPedestrianYieldRule,
    StopInCrosswalkRule,
    UnmarkedCrosswalkYieldRule,
)
from .rules.cyclist_and_bike_lane import BikeLaneEncroachmentRule, CyclistPassingRule
from .rules.drivable_and_surfaces import (
    DrivableBoundaryRule,
    NonTraversableSurfaceRule,
    SidewalkDriveRule,
)
from .rules.headway_and_lateral import (
    LaneIntrusionRule,
    LateralClearanceRule,
    SafeHeadwayRule,
)
from .rules.lane_direction import OneWayDirectionRule, OpposingLaneRule
from .rules.route_and_intersection import (
    BlockTheBoxRule,
    RouteAdherenceRule,
    UncontrolledIntersectionRule,
    YieldPriorityRule,
)
from .rules.speed_limit import SpeedLimitRule
from .rules.stop_and_traffic_light import MandatoryStopRule, TrafficLightComplianceRule
from .rules.vehicle_collision import VehicleCollisionRule
from .rules.vru_collision import VRUCollisionRule


DEFAULT_RULE_IDS = (
    # Level 10
    "10r0",
    "10r3",
    "10r4",
    "10r5",
    # Level 9
    "9r0",
    "9r1",
    # Level 8
    "8r0",
    "8r1",
    # Level 7
    "7r0",
    "7r1",
    "7r2",
    "7r3",
    "7r4",
    "7r5",
    # Level 3
    "3r0",
    "3r3",
    "3r5",
    "3r6",
    # Level 2
    "2r2",
    # Level 1
    "1r0",
    "1r2",
    "1r5",
    "1r11",
    # Level 0
    "0r2",
    "0r3",
)


def build_default_rules() -> List[ObserverRule]:
    """Instantiate the 24-rule NuPlan-applicable subset.

    Each call returns a fresh list so per-rule state (e.g. comfort rules'
    last-acceleration cache, mandatory-stop's approach state) is isolated to
    one observer instance.
    """
    return [
        VRUCollisionRule(),
        UnmarkedCrosswalkYieldRule(),
        CyclistPassingRule(),
        BikeLaneEncroachmentRule(),
        VehicleCollisionRule(),
        NonTraversableSurfaceRule(),
        MandatoryStopRule(),
        CrosswalkPedestrianYieldRule(),
        DrivableBoundaryRule(),
        TrafficLightComplianceRule(),
        OpposingLaneRule(),
        OneWayDirectionRule(),
        StopInCrosswalkRule(),
        SidewalkDriveRule(),
        SpeedLimitRule(),
        SafeHeadwayRule(),
        LateralClearanceRule(),
        LaneIntrusionRule(),
        RouteAdherenceRule(),
        YieldPriorityRule(),
        BlockTheBoxRule(),
        UncontrolledIntersectionRule(),
        LateralAccelerationRule(),
        LongitudinalComfortRule(),
        LateralComfortRule(),
    ]


# Sanity guard: keep DEFAULT_RULE_IDS in sync with build_default_rules().
def _assert_consistency() -> None:
    ids = [r.id for r in build_default_rules()]
    if sorted(ids) != sorted(DEFAULT_RULE_IDS):
        missing = set(DEFAULT_RULE_IDS) - set(ids)
        extra = set(ids) - set(DEFAULT_RULE_IDS)
        raise RuntimeError(
            f"Registry / DEFAULT_RULE_IDS mismatch. missing={sorted(missing)} extra={sorted(extra)}"
        )


_assert_consistency()
