# examples — runnable demos, batches, and the analysis pipeline

Thirteen numbered demo scripts (`01_*.py` through `13_*.py`) plus three analysis / post-processing scripts (`analyze_protocol.py`, `metrics_smoothness.py`, plus the four shared-infrastructure modules `simulation.py`, `planners.py`, `scenarios.py`, `visualizer.py`).

The numbered scripts fall into four groups:

| Range | Group | Harness |
|---|---|---|
| 01–05, 07–10 | Self-contained synthetic-world demos | [`simulation.py`](simulation.py) + [`planners.py`](planners.py) + [`scenarios.py`](scenarios.py) |
| 06 | Render a saved nuPlan SimulationLog | [`lexicone.observer.simulation_log_adapter`](../lexicone/observer/simulation_log_adapter.py) |
| 11–12 | Run the two-level MPC planner inside the real nuPlan simulator (single scenario or 16-scenario batch) | [`lexicone.planning`](../lexicone/planning/) + nuPlan's `run_simulation.py` |
| 13 | Multi-seed multi-condition driver for the comparative-effectiveness protocol — wraps demo 12 across 5 planner conditions × 5 seeds × 16 scenarios | calls demo 12 as a subprocess per cell |

All artefacts land under [`outputs/`](outputs/); see [`outputs/README.md`](outputs/README.md) for the full output-directory tree.

## Shared infrastructure

### `simulation.py` — synthetic closed-loop harness

The harness loops at **10 Hz** (`TICK_DT_US = 100_000` microseconds, [`simulation.py:49`](simulation.py)). One tick:

1. Build a `SceneContext` from the current snapshot.
2. Call `planner.plan(ctx) -> PlannerCommand`.
3. Apply the command via kinematic integration (`_advance_ego`).
4. Update scripted agents at `t = k · dt`.
5. Update traffic lights at `t = k · dt`.
6. Build the next `SceneSnapshot`.
7. Collision check; freeze ego and the colliding agent if `halt_on_collision=True`.
8. Append the snapshot to the returned list.

The simulator does **not** consume the planner's `planned_trajectory` — the ego is advanced from `ax_mps2` + `yaw_rate_radps` only. The trajectory is recorded for visualisation.

Key APIs:

```python
TICK_DT_US = 100_000   # 10 Hz cadence

@dataclass
class PlannerCommand:
    ax_mps2: float = 0.0
    yaw_rate_radps: float = 0.0
    planned_trajectory: Sequence[EgoSnapshot] = ()

class Planner(Protocol):
    name: str
    def reset(self) -> None: ...
    def plan(self, ctx: SceneContext) -> PlannerCommand: ...

@dataclass
class World:
    lanes: Tuple[LaneSnapshot, ...] = ()
    lane_connectors: Tuple[LaneSnapshot, ...] = ()
    crosswalks: Tuple[CrosswalkSnapshot, ...] = ()
    stop_lines: Tuple[StopLineSnapshot, ...] = ()
    intersections: Tuple[IntersectionSnapshot, ...] = ()
    drivable_area: Tuple[DrivableAreaSnapshot, ...] = ()
    walkways: Tuple[WalkwaySnapshot, ...] = ()
    bike_lanes: Tuple[LaneSnapshot, ...] = ()
    traffic_lights: TrafficLightSchedule = ()    # tuple OR function-of-time
    scripted_agents: Tuple[ScriptedAgent, ...] = ()
    route_lane_ids: Optional[Tuple[str, ...]] = None

def simulate(
    world: World,
    planner: Planner,
    initial: EgoSnapshot,         # note: parameter is named ``initial``, not ``initial_ego``
    n_ticks: int,
    dt_s: float = TICK_DT_US * 1e-6,
    *,
    halt_on_collision: bool = True,
) -> List[SceneSnapshot]
```

`initial_ego(x=0.0, y=0.0, heading=0.0, speed=5.0, length=4.7, width=1.85)` is a constructor *helper* — its return value is what you pass as the `initial=` keyword to `simulate(...)`.

Scripted-agent helpers:

- `constant_velocity_agent(initial)` — moves at the initial linear velocity forever.
- `static_agent(initial)` — holds the initial pose.
- `crossing_pedestrian(track_id, x, y_start, y_end, speed_mps, t_start_s, …)` — waits on the curb, then walks laterally at the given speed, then stops on the far side.

### `planners.py` — the synthetic-harness planners

Six classes today, all conforming to the `Planner` protocol above.

| Class | Description |
|---|---|
| `ConstantSpeedPlanner` | Holds current speed, never steers. Baseline. |
| `IDMPlanner(desired_speed_mps, time_headway_s, min_gap_m, max_accel_mps2, comfort_decel_mps2, respect_speed_limit)` | Treiber's IDM: free-flow term + interaction term based on lead vehicle. |
| `AggressivePlanner(target_speed_mps, max_accel_mps2, lateral_drift_radps, ignore_lead_below_m)` | Pushes hard toward a high target speed, ignores lead unless within `ignore_lead_below_m`. Rear-ends slow leads. |
| `OvertakePlanner(cruise_speed_mps, overtake_speed_mps, lateral_offset_m, trigger_gap_m, clear_gap_m, …)` | Five-state machine: `approach → merge_left → pass → merge_right → cruise`. P-on-position/heading cascade for steering. |
| `UrbanDrivingPlanner(…)` | Composite IDM + traffic-light + stop-sign + pedestrian yield. Picks the most restrictive of four longitudinal hazards each tick. |
| `LaneChangePlanner(…)` | Six-state machine: `follow → prepare_lane_change → merge → pass → return_to → follow`. Verifies target-lane clearance within `merge_corridor_m` before merging. |

These are pedagogical — they're tuned for the small synthetic worlds in `scenarios.py`, not for full nuPlan logs. (When citing kwargs in your own code, use the exact suffixed names — `desired_speed_mps`, `target_speed_mps`, etc.; the README in earlier revisions used the unsuffixed forms in passing prose, which the actual `__init__` signatures do not accept.)

### `scenarios.py` — canned and seeded synthetic worlds

Geometry primitives:

- `_rect(cx, cy, length, width)` — oriented rectangle as a 4-vertex polygon.
- `_straight_lane(lane_id, length, width, speed_limit_mps, …)` — a `LaneSnapshot` with centerline, polygon, optional speed limit.
- `_ego_at(...)`, `_agent(...)` — helpers for building snapshots.

Three episode builders:

- `intersection_red_light_episode()` — signalised intersection with entry/connector/exit lanes, cross-traffic, RED at start.
- `cyclist_overtake_episode()` — parallel vehicle + bike lanes, cyclist at 3 m/s, ego at 12.5 m/s.
- `random_episode(seed)` — deterministic seeded generator producing a straight-road episode with random map features (walkways, crosswalks, bike lane), random agents, and random ego behaviour.

Each builder returns a small `Episode` object carrying the snapshots and a scenario name.

### `visualizer.py` — `render_episode` — MP4 + summary PNG + CSV

```python
def render_episode(
    *,
    engine: RuleEngine,
    snapshots: Sequence[SceneSnapshot],
    scenario_name: str,
    output_dir: Path,
    fps: int = 10,
    map_margin_m: float = 60.0,
) -> Dict[str, Path]
```

Drives the engine over the snapshots and produces three artefacts in `output_dir`. The MP4 was redesigned to IEEE Transactions typography (Times serif, 10 pt body, 300 dpi) via [`../scripts/ieee_style.py`](../scripts/ieee_style.py)`apply()`:

- **`<scenario_name>.mp4`** — animation laid out in **four rows**:
  - **Header band** (top) — scenario name + tick / time / ego state ($v$, $a_x$, $a_y$) + status badge (`✓ COMPLIANT` or `✗ N VIOL / M APPL`), colour-coded by compliance.
  - **Map + sidebar** — left two-thirds: clean top-down map (lane geometry, ego, agents, planned trajectory dashed, traffic-light markers, scale bar). **Nothing overlays the map** — lane labels are restricted to lanes within `LANE_LABEL_RADIUS_M = 14 m` of the ego and show only the speed limit. Right third: context card (current lane id, $v_{\lim}$, TL ahead, lead distance, applicable rule count) + active-rules panel sorted by violation rate descending, colour-coded by priority level.
  - **Violation strip** — heatmap of per-rule violation rate over the episode with a moving tick cursor.
  - **Legend row** (bottom) — map-feature legend (drivable / lane / connector / crosswalk / walkway / intersection / ego / vehicles / pedestrians / cyclists / planned path / ego trail / TL colours / stop line).
- **`<scenario_name>_summary.png`** — static episode summary: sorted integrated-violation bar chart + per-rule violation-rate heatmap + applicability-vs-violation count.
- **`<scenario_name>_log.csv`** — one row per `(tick, rule)`: `t_s`, `rule_id`, `rule_name`, `applies`, `is_violated`, `violation_rate`. The canonical input to every downstream analysis script.

Colour conventions live in three module-level dicts (`AGENT_COLORS`, `LAYER_STYLE`, `TL_COLORS`) at the top of [`visualizer.py`](visualizer.py).

## The 12 numbered demos

### Synthetic-world demos (01–05, 07–10)

| Demo | Title | Planner | Scenario |
|---|---|---|---|
| **01** | [`01_intersection_red_light.py`](01_intersection_red_light.py) | scripted ego (no planner) | Signalised intersection. Ego rolls through a RED light at ~12 m/s. Expected violations: 7r1 (traffic-light), 0r2 (deceleration ramp). |
| **02** | [`02_cyclist_overtake.py`](02_cyclist_overtake.py) | scripted ego drifting | Parallel vehicle + bike lane. Ego at 12.5 m/s (over 25 mph limit) drifts toward cyclist. Violations observed in the checked-in log: `3r0`, `3r5`, `10r0`, `10r4`, `10r5`. |
| **03** | [`03_random_scenario.py`](03_random_scenario.py) | scripted ego, seeded random | Three seeded random straight-road episodes with random map features, agents, ego behaviour. Stress-test for the rule engine. |
| **04** | [`04_simulated_idm_following.py`](04_simulated_idm_following.py) | `IDMPlanner(desired_speed=11)` | 300 m east-bound lane, slow leader at 4 m/s 25 m ahead. Expected: clean (some 0r2 from initial decel ramp). |
| **05** | [`05_simulated_aggressive_planner.py`](05_simulated_aggressive_planner.py) | `AggressivePlanner(target_speed_mps=18.0)` | Same world as 04. Ego rear-ends the slow leader. Violations observed in the checked-in log: `0r2`, `3r0`, `3r3`, `3r5`, `9r0`. Contrasts with 04 to show rule-profile reversal. |
| **07** | [`07_simulated_overtake.py`](07_simulated_overtake.py) | `OvertakePlanner` | Two parallel lanes, slow leader. The planner's state machine executes a clean overtake. Violations observed in the checked-in log: `0r2`, `0r3`, `2r2`, `3r3` (`2r2` because the merge-left lane is not in the planned route corridor; no `3r5` because the merge gap is verified before lateral motion begins). |
| **08** | [`08_simulated_signalized_intersection.py`](08_simulated_signalized_intersection.py) | `UrbanDrivingPlanner` | 4-way signalised intersection with cross-traffic. Ego waits for green, launches smoothly. Expected: clean 7r1; some 0r2 from brake/launch. |
| **09** | [`09_simulated_urban_crossing.py`](09_simulated_urban_crossing.py) | `UrbanDrivingPlanner` | Marked crosswalk + dynamic pedestrian. Ego stops before the crosswalk while the pedestrian crosses. Violations observed in the checked-in log: `0r2`, `1r0`, `8r1` (the planner's brake ramp triggers comfort + a brief priority-yield window even though no collision occurs; `10r0` does not fire). |
| **10** | [`10_simulated_highway_multitraffic.py`](10_simulated_highway_multitraffic.py) | `LaneChangePlanner` | 3-lane highway with a slow truck and a fast left-lane vehicle. Lane change is deferred until the fast traffic clears. Violations observed in the checked-in log: `0r2`, `0r3`, `2r2`, `3r3`, `3r5` (comfort during the merge/return ramps; `2r2` because the passing lane is not part of the planned route). |

### Bridge demo (06)

| Demo | Description |
|---|---|
| **06** | [`06_render_nuplan_simulation_log.py`](06_render_nuplan_simulation_log.py) — load a real nuPlan `SimulationLog` (msgpack.xz or pickle) via `NuPlanSimulationLogSource`, convert each `SimulationHistorySample` to a `SceneSnapshot`, and render the same MP4/PNG/CSV triple. Useful for inspecting any nuPlan run (your own, IDM, MLPlanner, …) through the rule engine. |

### nuPlan-simulator demos (11–12)

These run the actual nuPlan closed-loop simulator (`nuplan/planning/script/run_simulation.py`) with our two-level MPC planner, then load the produced `SimulationLog` and render it.

| Demo | Description |
|---|---|
| **11** | [`11_simulated_two_level_mpc_planner.py`](11_simulated_two_level_mpc_planner.py) — single-scenario invocation. Picks a dynamic mini-split scenario (`following_lane_with_lead`, `high_magnitude_speed`, …) and runs the MPC. Uses `closed_loop_reactive_agents` so traffic responds to the ego. |
| **12** | [`12_batch_two_level_mpc_planner.py`](12_batch_two_level_mpc_planner.py) — batch runner over 16 default scenario types (overtake / lane-change / turn / dynamic-speed). Writes per-scenario MP4 + per-tick CSV + summary PNG plus a top-level `batch_summary[_<suffix>].csv`. CLI exposes the full LCP-mode flag set: `--penalty-form {l1, l2}` and `--runtime-mode {ws, cascade}` (output-dir suffix `__l1`, `__l2`, `__l1_cascade` keeps each mode separate); `--slp-max-iterations` and `--slp-residual-tol-m` (SLP outer-loop tuning); `--seed`, `--types <csv>` + `--label-offset` for custom batches, `--limit` for partial runs, `--output-dir`, plus `--mpc-horizon-s`, `--mpc-dt-s`, `--replan-period-s`, `--desired-speed-mps`, `--margin`, `--radius`, `--fps`, `--max-logs` for planner tuning. |

### Comparative-protocol driver (13) + analysis scripts

| Script | Description |
|---|---|
| **13** | [`13_run_protocol.py`](13_run_protocol.py) — multi-seed, multi-condition driver for the comparative-effectiveness protocol. Wraps demo 12 across up to 5 planner conditions (`C0_legacy`, `C1_ws_l1`, `C2_ws_l1_slp3`, `C3_ws_l2`, `C4_cascade_l1`) × N seeds × 16 scenarios. Runs cells in parallel via `multiprocessing.Pool` with a cell-unique `NUPLAN_EXP_ROOT` per `(condition, seed)` (race-free by construction) and post-cell disk cleanup to bound peak disk. CLI: `--conditions C0,C1,C2,C3,C4`, `--seeds 7,17,27,37,47`, `--parallel 4`, `--resume` (skips cells whose `batch_summary*.csv` already has ≥ 16 rows), `--limit N`, `--dry-run`. Output: `outputs/13_protocol/<cond>/seed_<n>/<label>/`. See [`outputs/13_protocol/README.md`](outputs/13_protocol/README.md) for the full layout. |
| **`analyze_protocol.py`** | [`analyze_protocol.py`](analyze_protocol.py) — post-batch analysis. Reads every per-tick CSV under `outputs/13_protocol/`, computes per-priority-level integrated violation $V_{\mathrm{MPC},\ell}$ restricted to the 16 MPC-controlled rules (partition declared at [`../lexicone/planning/tests/test_rule_level_mapping.py`](../lexicone/planning/tests/test_rule_level_mapping.py)), aggregates seeds by per-scenario median, runs pairwise lex-Pareto-dominance comparisons against a baseline (default `C0_legacy`), and emits F1 (per-level per-scenario stacked bars), F2 (per-level percent-reduction violins with BCa bootstrap CIs), F3 (lex-dominance heatmap) plus long-form `per_cell_metrics.csv` and `lex_dominance.csv`. Headline lex-win-rate with exact binomial 95% CI is printed to stdout. |
| **`metrics_smoothness.py`** | [`metrics_smoothness.py`](metrics_smoothness.py) — post-batch smoothness extractor. For each cell's `.msgpack.xz` simulation log, replays via `NuPlanSimulationLogSource` and computes `(lat_jerk_rms, lon_jerk_rms, peak_a_x, peak_a_y, mean_speed, distance_m)` from the ego-pose stream. Output is a long-form CSV consumed by the smoothness Pareto figure (F7 in the paper's discussion). ~5 s replay per scenario; no simulator re-run. |

Both demos read their planner config from [`../lexicone/planning/config/planner/two_level_mpc_planner.yaml`](../lexicone/planning/config/planner/two_level_mpc_planner.yaml) (extended via Hydra's `hydra.searchpath` override).

## How a synthetic demo is wired

```python
from examples.simulation import World, initial_ego, simulate, constant_velocity_agent
from examples.scenarios import _straight_lane, _rect
from examples.planners import IDMPlanner
from examples.visualizer import render_episode
from lexicone.observer import RuleEngine
from lexicone.observer.types import AgentSnapshot, AgentType, Pose2D, DrivableAreaSnapshot

world = World(
    lanes=(_straight_lane("main", length=300.0, width=3.5, speed_limit_mps=11.0),),
    drivable_area=(DrivableAreaSnapshot(polygon=_rect(0.0, 0.0, 300.0, 8.0)),),
    scripted_agents=(
        constant_velocity_agent(AgentSnapshot(
            track_id="leader",
            object_type=AgentType.VEHICLE,
            pose=Pose2D(x=25.0, y=0.0, heading=0.0),
            vx=4.0, vy=0.0, length=4.5, width=1.8,
        )),
    ),
    route_lane_ids=("main",),
)

planner = IDMPlanner(desired_speed_mps=11.0)
snapshots = simulate(
    world=world,
    planner=planner,
    initial=initial_ego(x=0.0, y=0.0, heading=0.0, speed=8.0),
    n_ticks=80,
)

engine = RuleEngine()
artefacts = render_episode(engine=engine, snapshots=snapshots,
                          scenario_name="idm_following", output_dir=Path("outputs/04_idm/"))
```

## How a nuPlan-simulator demo is wired

```text
demo (11 or 12)
   │
   ├── pick scenario type(s)
   ├── subprocess.run(
   │       python nuplan/planning/script/run_simulation.py
   │         +simulation=closed_loop_reactive_agents
   │         planner=two_level_mpc_planner          ← our YAML
   │         scenario_builder=nuplan_mini
   │         scenario_filter=all_scenarios
   │         scenario_filter.scenario_types=[…]
   │         scenario_filter.log_names=[…readable mini DBs…]
   │         scenario_filter.limit_total_scenarios=1
   │         scenario_filter.shuffle=true
   │         hydra.searchpath=[file://…/lexicone/planning/config, …]
   │   )
   ├── find the produced .msgpack.xz SimulationLog
   ├── NuPlanSimulationLogSource.from_path(log_path)
   ├── list(source) → List[SceneSnapshot]
   ├── RuleEngine()                                        ← scores the run
   └── render_episode(...) → MP4 + summary PNG + CSV       ← same renderer as 01–10
```

## Outputs

Each demo writes to `outputs/<demo_label>/`:

```text
outputs/
├── 04_simulated_idm_following/
│   ├── idm_following.mp4
│   ├── idm_following_summary.png
│   └── idm_following_log.csv
├── 11_two_level_mpc_planner/
│   ├── c8e6df66017b5ea4.msgpack.mp4
│   ├── c8e6df66017b5ea4.msgpack_summary.png
│   └── c8e6df66017b5ea4.msgpack_log.csv
├── 12_batch_two_level_mpc_planner/
│   ├── 01_following_slow_lead/
│   │   ├── 01_following_slow_lead.mp4
│   │   ├── 01_following_slow_lead_summary.png
│   │   └── 01_following_slow_lead_log.csv
│   ├── …  (one folder per scenario)
│   └── batch_summary.csv
└── …
```

## Why a "synthetic harness" *and* a "real nuPlan" harness?

The synthetic harness is for understanding the **rule engine** in isolation — every world is small, hand-built, and deterministic; every demo isolates one or two rules. The nuPlan harness is for evaluating the **planner** at realistic scale — it pulls the actual recorded scenes from `nuplan-v1.1/splits/mini`, runs the simulator's full machinery (TwoStageController, IDM-reactive agents, metric engine, log callback), and produces evaluation artefacts the team can compare across planner variants.

Both end up emitting the same `list[SceneSnapshot]` to the visualiser, so the rendering pipeline doesn't need to know the difference.
