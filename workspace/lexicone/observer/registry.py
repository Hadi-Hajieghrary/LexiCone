"""Default rule set for the rule engine.

25 rules, drawn from the README's NuPlan-applicable subset. Each rule lives
in its own module under :mod:`lexicone.observer.rules`.
"""

from __future__ import annotations

from typing import List

from .rule import ObserverRule
from .rules.bike_lane_encroachment import BikeLaneEncroachmentRule
from .rules.block_the_box import BlockTheBoxRule
from .rules.crosswalk_pedestrian_yield import CrosswalkPedestrianYieldRule
from .rules.cyclist_passing import CyclistPassingRule
from .rules.drivable_boundary import DrivableBoundaryRule
from .rules.lane_intrusion import LaneIntrusionRule
from .rules.lateral_acceleration import LateralAccelerationRule
from .rules.lateral_clearance import LateralClearanceRule
from .rules.lateral_comfort import LateralComfortRule
from .rules.longitudinal_comfort import LongitudinalComfortRule
from .rules.mandatory_stop import MandatoryStopRule
from .rules.non_traversable_surface import NonTraversableSurfaceRule
from .rules.one_way_direction import OneWayDirectionRule
from .rules.opposing_lane import OpposingLaneRule
from .rules.route_adherence import RouteAdherenceRule
from .rules.safe_headway import SafeHeadwayRule
from .rules.sidewalk_drive import SidewalkDriveRule
from .rules.speed_limit import SpeedLimitRule
from .rules.stop_in_crosswalk import StopInCrosswalkRule
from .rules.traffic_light_compliance import TrafficLightComplianceRule
from .rules.uncontrolled_intersection import UncontrolledIntersectionRule
from .rules.unmarked_crosswalk_yield import UnmarkedCrosswalkYieldRule
from .rules.vehicle_collision import VehicleCollisionRule
from .rules.vru_collision import VRUCollisionRule
from .rules.yield_priority import YieldPriorityRule


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
    """Instantiate the 25-rule NuPlan-applicable subset.

    Each call returns a fresh list so per-rule state (e.g. comfort rules'
    last-acceleration cache, mandatory-stop's approach state) is isolated
    to one engine instance.
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
