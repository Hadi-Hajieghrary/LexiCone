# lexicone.observer — per-tick trajectory rule engine

A pure-Python engine that watches a stream of driving scenes (synthetic or recorded) and reports, *every tick*, which of 25 driving rules are applicable and to what degree they are violated. It is designed for two complementary purposes:

1. **Evaluation** — given a saved nuPlan `SimulationLog` (or a synthetic episode), score the ego against a NuPlan-applicable subset of a real-world rulebook to find safety/comfort regressions.
2. **Reward shaping** — the same per-tick rate signals can drive reward functions for RL or cost terms for MPC.

## Conceptual model

A **rule** is a piece of code that, given the current scene context, answers two questions:

1. *Does this rule apply right now?* (e.g. "is the ego on a lane with a posted speed limit?")
2. *If it applies, how much is the ego in violation?* (a non-negative `violation_rate`)

The engine then integrates the per-tick rate over time:

```
integrated_violation = ∫ violation_rate(t) dt        (only over applicable ticks)
```

Each rule lives in its own file under [`rules/`](rules/) (see [`rules/README.md`](rules/README.md) for the catalogue). The engine [`engine.py`](engine.py) is dumb: it just dispatches each tick's `SceneContext` to every rule and records the results.

## The data model

All data lives in [`types.py`](types.py). The shapes below are the public contract every adapter (live nuPlan, saved log, synthetic) must produce.

### `SceneSnapshot` — the raw input per tick

```python
@dataclass
class SceneSnapshot:
    timestamp_us: int
    ego: EgoSnapshot
    agents: Sequence[AgentSnapshot] = ()
    map: MapSnapshot = MapSnapshot()
    traffic_lights: Sequence[TrafficLightStatus] = ()
    planned_trajectory: Optional[Sequence[EgoSnapshot]] = None
    route_lane_ids: Optional[Sequence[str]] = None
    extras: Mapping[str, Any] = {}
```

`MapSnapshot` itself groups: `lanes`, `lane_connectors`, `crosswalks`, `stop_lines`, `intersections`, `drivable_area`, `walkways`, `bike_lanes`. Every map feature carries an id + polygon + (for lanes) centerline + speed limit + incoming/outgoing IDs.

### `SceneContext` — per-tick derived state

[`context.py`](context.py) wraps a snapshot and adds *cached* derivations used by multiple rules:

```python
class SceneContext:
    snapshot: SceneSnapshot
    ego: EgoSnapshot
    timestamp_us: int
    ego_footprint: Polygon                     # Shapely OBB
    ego_center: Tuple[float, float]
    @cached_property ego_lane: Optional[LaneSnapshot]
    @cached_property ego_speed_limit_mps: Optional[float]
    @cached_property drivable_polygons: Sequence[Polygon]
    @cached_property ego_in_intersection: bool
    @cached_property walkway_overlap_m2: float
    @cached_property bike_lane_overlap_m2: float
    def agents_by_type(types, radius_m=None) -> Sequence[AgentSnapshot]
    def lead_agent(max_distance_m=80.0, lateral_tol_m=1.6) -> Optional[LeadAgent]
    def lateral_neighbors(max_long_m=8.0, lateral_band_m=4.0) -> List[LateralNeighbor]
    def red_or_yellow_for_lane(lane_id) -> Tuple[bool, Optional[str]]
    def stop_polygon_for_ego(ego_lane=None) -> Optional[Tuple[Polygon, float]]
```

`functools.cached_property` (see [`context.py:18`](context.py)) means each derivation is computed at most once per tick *if* a rule asks for it — rules that don't touch `walkway_overlap_m2` pay nothing for it.

### `RuleEvaluation` — per-tick output for one rule

```python
@dataclass
class RuleEvaluation:
    rule_id: str                # e.g. "3r0"
    rule_level: int             # lexicographic level 0..10
    rule_name: str              # human-readable
    timestamp_us: int
    applies: bool               # gating condition
    violation_rate: float       # ≥ 0
    is_violated: bool           # applies and violation_rate > 0
    details: Mapping[str, Any]  # {"applicability": {...}, "violation": {...}}
```

`is_violated` is strictly `applies and violation_rate > 0.0`. A rule that is *not applicable* contributes nothing to integrated violation, no matter what the rate would have been.

### `RuleSummary` / `EpisodeSummary` — windowed aggregates

```python
@dataclass
class RuleSummary:
    rule_id: str
    rule_level: int
    rule_name: str
    n_steps_total: int
    n_steps_applicable: int
    n_steps_violated: int
    duration_applicable_s: float          # ∫ dt over applicable steps
    integrated_violation: float           # ∫ rate · dt over applicable steps
    max_violation_rate: float
    first_violation_t_s: Optional[float]
    last_violation_t_s: Optional[float]
```

`EpisodeSummary` is the dict of per-rule summaries plus the window bounds. It exposes convenience accessors `violated_rules()` and `by_level()` for grouping output by lexicographic level.

## The `ObserverRule` interface

Every rule subclasses [`ObserverRule`](rule.py):

```python
class ObserverRule:
    id: str = ""
    level: int = -1
    name: str = ""
    description: str = ""

    def applies(self, ctx: SceneContext) -> Tuple[bool, Mapping[str, Any]]:
        raise NotImplementedError

    def violation(self, ctx: SceneContext) -> Tuple[float, Mapping[str, Any]]:
        raise NotImplementedError

    def evaluate(self, ctx: SceneContext) -> RuleEvaluation:
        applies, app_details = self.applies(ctx)
        if not applies:
            return RuleEvaluation(..., applies=False, violation_rate=0.0,
                                  is_violated=False, details={"applicability": app_details})
        rate, viol_details = self.violation(ctx)
        rate = max(0.0, float(rate))
        return RuleEvaluation(..., applies=True, violation_rate=rate,
                              is_violated=rate > 0.0,
                              details={"applicability": app_details, "violation": viol_details})
```

The contract is intentionally narrow: rules are pure functions of `SceneContext`, with one caveat — they may keep their own internal state across ticks (e.g. the longitudinal-comfort rule caches the previous `ax` to estimate jerk; the mandatory-stop rule tracks approach state). The engine does **not** call any reset hook between episodes, so each `RuleEngine` instance gets a *fresh* set of rule instances (via [`build_default_rules()`](registry.py) returning a new list every call).

## The `RuleEngine`

The engine ([`engine.py`](engine.py)) has a streaming and a batch API:

```python
class RuleEngine:
    def __init__(self, rules: Iterable[ObserverRule] = None):
        # defaults to build_default_rules() — 25 instances

    def step(self, snap: SceneSnapshot) -> List[RuleEvaluation]:
        # construct SceneContext(snap), evaluate every rule, append to history.

    def run_replay(self, scenes: Iterable[SceneSnapshot]) -> List[List[RuleEvaluation]]:
        # convenience loop around step().

    @property
    def history -> Sequence[Sequence[RuleEvaluation]]
    @property
    def snapshots -> Sequence[SceneSnapshot]
    def current_applicable_rules() -> List[RuleEvaluation]
    def current_violations() -> List[RuleEvaluation]

    def summary(self, window_s: Optional[Tuple[float, float]] = None) -> EpisodeSummary:
        # integrate per-rule rates over a window (default: whole episode).
```

The integration in `summary()` uses per-tick `dt` computed as the time delta to the next snapshot, with the first tick mirroring the second tick's `dt` to avoid spurious zero-duration. The integral is `Σ rate[k] · dt[k]` over the indices where `applies and not is_violated == False` (i.e., where the rule was applicable, regardless of whether the rate was zero).

## The 25-rule default set

[`registry.py`](registry.py) exposes `build_default_rules()` which instantiates the 25 rules and `DEFAULT_RULE_IDS` which lists their IDs. A runtime check (`_assert_consistency()` at module load) refuses to import if the two go out of sync.

The IDs follow the convention **`<level>r<n>`** where `level ∈ {0, 1, 2, 3, 7, 8, 9, 10}` and `n` is a within-level rule number. Higher level = higher priority. See [`rules/README.md`](rules/README.md) for the catalogue and per-rule semantics.

## Lexicographic priority — current usage

Each rule sets a `level: int` class attribute. This value is:

- **Reported** on every `RuleEvaluation` and `RuleSummary` (`rule_level` field).
- **Groupable** via `EpisodeSummary.by_level()` for post-hoc analysis.
- **Not used** by `RuleEngine` itself for sorting, weighting, or scheduling. The engine evaluates every rule independently.

Future cost-aggregation or constraint-priority logic could read the level field; today, it is metadata for the consumer.

## Adapters

### `NuPlanSceneSource` — live scenario stream

[`nuplan_adapter.py`](nuplan_adapter.py) converts an `AbstractScenario` (the live nuPlan object you'd get from a `ScenarioBuilder`) into a `SceneSnapshot` stream:

```python
source = NuPlanSceneSource(
    scenario,
    radius_m=80.0,
    planner_predictions=optional_iterable,   # one trajectory per iteration
    route_lane_ids=None,                     # falls back to scenario.get_route_roadblock_ids()
    include_lane_connectors=True,
)
for snap in source:    # one SceneSnapshot per iteration
    ...
```

Internally each `snapshot_at(iteration)` queries the scenario's ego state, detections, traffic lights, and proximal map objects, and converts each into the lexicone dataclasses. The nuPlan-side `SemanticMapLayer` enum is imported lazily so the rest of the observer package works without `nuplan-devkit` installed.

A small portability fix in `_map_to_snapshot()` filters the requested map-object layers through `map_api.get_available_map_objects()` so layers exposed only as raster (e.g. `DRIVABLE_AREA` on NuPlanMap) are silently skipped — see [`nuplan_adapter.py:272-289`](nuplan_adapter.py).

### `NuPlanSimulationLogSource` — saved replay

[`simulation_log_adapter.py`](simulation_log_adapter.py) is the same idea but for a *pre-recorded* `SimulationLog` (produced by nuPlan's `SimulationLogCallback`, e.g. by the `run_simulation.py` Hydra script in [`demos/13_idm_simulation_and_record.py`](../../../DevContainers/demos/13_idm_simulation_and_record.py) or by our own [`examples/11_simulated_two_level_mpc_planner.py`](../../examples/11_simulated_two_level_mpc_planner.py)):

```python
source = NuPlanSimulationLogSource.from_path(
    log_path,
    radius_m=80.0,
    route_lane_ids=None,            # falls back to scenario.get_route_roadblock_ids()
    include_lane_connectors=True,
)
for snap in source:                 # one SceneSnapshot per SimulationHistorySample
    ...
```

Both adapters share helpers under the hood (see `_ego_to_snapshot`, `_detections_to_agents`, `_map_to_snapshot` in [`nuplan_adapter.py`](nuplan_adapter.py)). The difference is the iteration source: `NuPlanSceneSource` walks `scenario.get_number_of_iterations()`; `NuPlanSimulationLogSource` walks `simulation_log.simulation_history.data`.

## Geometry primitives

[`geometry.py`](geometry.py) is the small numerical core that the rules rely on. Everything in this module is Shapely- and NumPy-based:

| Helper | Purpose |
|---|---|
| `ego_footprint(ego)` | Oriented bounding box of the ego, accounting for rear-axle offset. |
| `ego_center(ego)` | `(x, y)` tuple of the geometric centre. |
| `agent_footprint(agent)` | Same as `ego_footprint` for a tracked agent. |
| `polygon_from_points(points)` | Build a valid Shapely polygon, repairing if necessary. |
| `polyline_from_points(points)` | Same for LineString. |
| `planar_distance(a, b)` | Euclidean distance between two `(x, y)` points. |
| `heading_difference(a, b)` | Signed minimal heading diff wrapped to `(-π, π]`. |
| `is_same_direction(h1, h2, tol_rad)` | `abs(diff) ≤ tol`. |
| `is_opposite_direction(h1, h2, tol_rad)` | `abs(diff - π) ≤ tol`. |
| `project_onto_polyline(point, polyline)` | Returns `(arc_length, signed_lateral_offset, segment_heading)`. |
| `find_ego_lane(ego, lanes)` | Lane polygon containing ego centre; tiebreak by heading alignment. |
| `footprint_overlaps_polygon(footprint, polygon)` | Overlap area in m². |
| `footprint_outside_drivable(footprint, drivable_polygons)` | Area outside the union of drivable polygons. |
| `closest_agent(ego, agents)` | `(agent, distance)` of the nearest tracked object. |

## Per-tick example

```python
from lexicone.observer import RuleEngine, SceneSnapshot, build_default_rules

engine = RuleEngine()                          # 25 default rules
for snap in snapshots:                         # snapshots: Iterable[SceneSnapshot]
    evaluations = engine.step(snap)            # 25 RuleEvaluation objects
    violated_now = [e for e in evaluations if e.is_violated]

summary = engine.summary()                     # EpisodeSummary across all ticks
for s in summary.violated_rules():
    print(f"{s.rule_id:>5}  L{s.rule_level}  "
          f"{s.n_steps_violated}/{s.n_steps_applicable} ticks  "
          f"max={s.max_violation_rate:.3f}  integ={s.integrated_violation:.3f}")
```

## Tests

The `tests/` directory contains:

- `test_observer.py` — engine smoke tests (default 25-rule construction, integration math, applicability gating for a few representative rules).
- `test_rule_coverage.py` — 21 per-rule positive/negative tests covering **16 of the 25 rule IDs**: `0r2`, `0r3`, `1r0`, `1r2`, `1r5`, `2r2`, `3r5`, `3r6`, `7r0`, `7r2`, `7r3`, `7r4`, `8r1`, `9r1`, `10r3`, `10r4`. Rules **without** dedicated coverage in this file: `10r0`, `10r5`, `9r0`, `8r0`, `7r1`, `7r5`, `3r0`, `3r3`, `1r11` — some of these are exercised in `test_observer.py` instead (e.g. `3r0` speed-limit, `10r0` VRU collision, `1r11` lateral accel), the rest are uncovered by the test suite today.

Both files use synthetic `SceneSnapshot` objects (no nuPlan-devkit required) constructed via `tests/synthetic.py`.

Run:

```bash
PYTHONPATH=/workspace/nuplan-project/workspace \
    pytest workspace/lexicone/observer/tests/
```

## Files at a glance

| File | What it contains |
|---|---|
| [`__init__.py`](__init__.py) | Public re-exports — `RuleEngine`, `SceneContext`, `ObserverRule`, `build_default_rules`, plus every dataclass from `types.py`. **The adapters are deliberately not re-exported** (see `__all__`) because they import nuplan-devkit at module scope; import them from `lexicone.observer.nuplan_adapter` / `lexicone.observer.simulation_log_adapter` directly. |
| [`rule.py`](rule.py) | `ObserverRule` ABC and its `evaluate()` template-method. (Note: `RuleEvaluation`, `RuleSummary`, `EpisodeSummary` and every other dataclass live in [`types.py`](types.py).) |
| [`engine.py`](engine.py) | `RuleEngine` — per-tick dispatch + windowed `summary()`. |
| [`types.py`](types.py) | All dataclasses: `EgoSnapshot`, `AgentSnapshot`, `MapSnapshot`, `LaneSnapshot`, `SceneSnapshot`, `RuleEvaluation`, `RuleSummary`, `EpisodeSummary`. |
| [`context.py`](context.py) | `SceneContext` — `cached_property` derivations + neighbour queries. |
| [`registry.py`](registry.py) | `build_default_rules()` returning the 25 rules; `DEFAULT_RULE_IDS` sanity-checked at import time. |
| [`geometry.py`](geometry.py) | Shapely-based footprint, projection, lane-finding helpers. |
| [`nuplan_adapter.py`](nuplan_adapter.py) | `NuPlanSceneSource` for live scenarios. |
| [`simulation_log_adapter.py`](simulation_log_adapter.py) | `NuPlanSimulationLogSource` for saved logs. |
| [`rules/`](rules/) | One file per rule, 25 total. |
| [`tests/`](tests/) | Engine + per-rule unit tests. |
