# lexicone.observer.rules — the 25-rule rulebook

This directory contains one Python module per rule in the default rule set. Each rule inherits from [`ObserverRule`](../rule.py) and provides two methods:

- `applies(ctx) -> (bool, dict)` — gating: does this rule's precondition hold right now?
- `violation(ctx) -> (float, dict)` — only called when `applies` is true: the non-negative violation rate at this tick.

The 25 rules cover a NuPlan-applicable subset of a real-world driving rulebook, drawn from a longer internal taxonomy and grouped by **lexicographic level** (priority). Higher level = higher priority — e.g. a collision (level 9–10) is conceptually superior to a comfort issue (level 0).

## Rule-ID convention

`<level>r<n>` where `level ∈ {0, 1, 2, 3, 7, 8, 9, 10}` and `n` is a within-level rule number from the master taxonomy (the gaps in `n` reflect rules in the taxonomy that *aren't* included in this NuPlan-applicable subset).

## The catalogue

The table below is the authoritative list as wired up in [`registry.py`](../registry.py:39-73). It is kept consistent with `build_default_rules()` by a runtime assertion (`_assert_consistency()`) that runs at import time.

| ID | Level | Class (file) | Purpose | Trigger when … |
|---|---|---|---|---|
| **10r0** | 10 | `VRUCollisionRule` ([`vru_collision.py`](vru_collision.py)) | Avoid collision with VRUs | Ego footprint overlaps an inflated pedestrian/cyclist footprint. |
| **10r3** | 10 | `UnmarkedCrosswalkYieldRule` ([`unmarked_crosswalk_yield.py`](unmarked_crosswalk_yield.py)) | Yield at unmarked crosswalks | At an intersection without marked crosswalk, ego moves while a pedestrian is in the implicit crossing zone. |
| **10r4** | 10 | `CyclistPassingRule` ([`cyclist_passing.py`](cyclist_passing.py)) | Safe passing distance for cyclists | Lateral clearance to a cyclist being passed is below a speed-dependent minimum. |
| **10r5** | 10 | `BikeLaneEncroachmentRule` ([`bike_lane_encroachment.py`](bike_lane_encroachment.py)) | Do not encroach into bicycle lanes | Ego footprint overlaps a bike-lane polygon. |
| **9r0** | 9 | `VehicleCollisionRule` ([`vehicle_collision.py`](vehicle_collision.py)) | Avoid collision with non-VRU vehicles or obstacles | Ego footprint overlaps any non-VRU obstacle (vehicle, motorcycle, barrier, cone, generic). |
| **9r1** | 9 | `NonTraversableSurfaceRule` ([`non_traversable_surface.py`](non_traversable_surface.py)) | Avoid non-traversable surface | Any portion of ego footprint is outside the mapped drivable polygon. |
| **8r0** | 8 | `MandatoryStopRule` ([`mandatory_stop.py`](mandatory_stop.py)) | Comply with mandatory stops | At a stop sign or red-light controlled stop line, ego either crosses the line moving or fails to stop. |
| **8r1** | 8 | `CrosswalkPedestrianYieldRule` ([`crosswalk_pedestrian_yield.py`](crosswalk_pedestrian_yield.py)) | Yield right-of-way to pedestrians at crosswalks | Ego speed is above yield threshold while a pedestrian is on a marked crosswalk in the ego's path. |
| **7r0** | 7 | `DrivableBoundaryRule` ([`drivable_boundary.py`](drivable_boundary.py)) | Stay within drivable surface boundaries | Ego footprint is beyond the lane boundary, excluding intentional intersection traversal. |
| **7r1** | 7 | `TrafficLightComplianceRule` ([`traffic_light_compliance.py`](traffic_light_compliance.py)) | Obey traffic-light states | Time inside or entering an intersection while its lane-connector traffic-light is RED (or YELLOW when stopping is feasible). |
| **7r2** | 7 | `OpposingLaneRule` ([`opposing_lane.py`](opposing_lane.py)) | Avoid opposing lane with oncoming traffic | Ego footprint overlaps a lane whose heading is opposite to ego's, with oncoming traffic present. |
| **7r3** | 7 | `OneWayDirectionRule` ([`one_way_direction.py`](one_way_direction.py)) | Obey one-way street directionality | Ego occupies a lane whose heading is opposite to ego's (wrong-way driving). |
| **7r4** | 7 | `StopInCrosswalkRule` ([`stop_in_crosswalk.py`](stop_in_crosswalk.py)) | Do not stop inside crosswalks | Ego footprint overlaps a marked crosswalk while essentially stopped. |
| **7r5** | 7 | `SidewalkDriveRule` ([`sidewalk_drive.py`](sidewalk_drive.py)) | Do not drive on sidewalks/pedestrian areas | Ego footprint overlaps a walkway polygon (weighted by speed). |
| **3r0** | 3 | `SpeedLimitRule` ([`speed_limit.py`](speed_limit.py)) | Obey posted speed limits | Ego speed exceeds lane's posted limit plus a small tolerance (1 m/s default). |
| **3r3** | 3 | `SafeHeadwayRule` ([`safe_headway.py`](safe_headway.py)) | Maintain safe following headway | Time headway (THW) or time-to-collision (TTC) to in-lane lead falls below safe minimums. |
| **3r5** | 3 | `LateralClearanceRule` ([`lateral_clearance.py`](lateral_clearance.py)) | Maintain lateral clearance | Lateral distance to adjacent agents is below a dynamic minimum that grows with relative lateral velocity. |
| **3r6** | 3 | `LaneIntrusionRule` ([`lane_intrusion.py`](lane_intrusion.py)) | Manage lane intrusions from adjacent vehicles | Lateral time-to-collision with an adjacent vehicle is low enough to require gap creation. |
| **2r2** | 2 | `RouteAdherenceRule` ([`route_adherence.py`](route_adherence.py)) | Adhere to the planned global route | Ego drifts laterally outside the planned route's lane corridor or topologically deviates. |
| **1r0** | 1 | `YieldPriorityRule` ([`yield_priority.py`](yield_priority.py)) | Yield to higher-priority road users | Ego encroaches into the path of a prioritised agent and forces it to react. |
| **1r2** | 1 | `BlockTheBoxRule` ([`block_the_box.py`](block_the_box.py)) | Don't block the box | Ego is stopped inside an intersection with no clear downstream gap to exit. |
| **1r5** | 1 | `UncontrolledIntersectionRule` ([`uncontrolled_intersection.py`](uncontrolled_intersection.py)) | Negotiate uncontrolled intersections safely | At an intersection without traffic-light/stop-sign for ego's lane, ego violates speed or proximity to cross-traffic. |
| **1r11** | 1 | `LateralAccelerationRule` ([`lateral_acceleration.py`](lateral_acceleration.py)) | Limit lateral acceleration for comfort | \|ay\| (measured or approximated as `v · yaw_rate`) exceeds comfort threshold. |
| **0r2** | 0 | `LongitudinalComfortRule` ([`longitudinal_comfort.py`](longitudinal_comfort.py)) | Limit uncomfortable longitudinal manoeuvres | \|ax\| or longitudinal jerk (finite-differenced) exceeds comfort threshold. |
| **0r3** | 0 | `LateralComfortRule` ([`lateral_comfort.py`](lateral_comfort.py)) | Limit uncomfortable lateral manoeuvres | \|ay\| or lateral jerk exceeds comfort threshold (smoother sister of 1r11). |

## Writing a new rule

To add a 26th rule:

1. Create a file in this directory implementing a subclass of `ObserverRule`. Set the class attributes:
   ```python
   class MyRule(ObserverRule):
       id = "<level>r<n>"
       level = <int>
       name = "Human-readable name"
       description = "What this rule encodes."

       def applies(self, ctx):
           ...
           return applies, {"why_applies": ...}

       def violation(self, ctx):
           ...
           return float(rate), {"why_violated": ...}
   ```
2. Import the class into [`registry.py`](../registry.py) and add an entry both to `DEFAULT_RULE_IDS` (so the consistency check sees it) and to the `build_default_rules()` list.
3. Add a positive coverage test in [`../tests/test_rule_coverage.py`](../tests/test_rule_coverage.py).

The registry's import-time assertion (`_assert_consistency()`) will refuse to import if the ID list and the `build_default_rules()` output go out of sync.

## Design notes per category

### Collision rules (10r0, 9r0)
Both return *area* (m²) of footprint overlap as the violation rate. Pedestrians and cyclists have their footprints inflated by a small radius before the overlap test (VRU rule) so brushing past counts as "near collision" with a smooth ramp.

### Surface rules (9r1, 7r0, 7r5, 10r5)
Each tests footprint overlap with a specific surface class. `9r1` is the union complement of drivable area; `7r0` is the lane boundary specifically; `7r5` is the walkway polygon; `10r5` is the bike-lane polygon. They use the same Shapely `intersection().area` primitive from [`geometry.py`](../geometry.py).

### Lane-orientation rules (7r2, 7r3)
Both rely on `heading_difference(lane_heading, ego_heading)`. The difference between them is the precondition: `7r3` fires on any wrong-way lane occupancy; `7r2` only fires when oncoming traffic is actually present in that lane.

### Speed / headway rules (3r0, 3r3, 3r5, 3r6)
Continuous quadratic-ish penalties driven by physical thresholds (speed limit, TTC, THW, lateral gap vs lateral velocity).

### Comfort rules (0r2, 0r3, 1r11)
Each caches the previous acceleration (or yaw rate) per instance to estimate jerk via finite difference. This is the only state held across ticks by any rule, which is why `build_default_rules()` returns a *fresh* list every call (so a new engine instance gets virgin rule state).

### Stop/yield rules (8r0, 8r1, 10r3)
The mandatory-stop rule (`8r0`) keeps approach state across ticks: it watches the ego decelerate toward the stop line and registers whether a full stop actually occurred.
