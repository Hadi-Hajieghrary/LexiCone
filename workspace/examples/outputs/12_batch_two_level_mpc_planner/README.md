# `12_batch_two_level_mpc_planner/` — first iteration of the 16-scenario LCP batch

This is the **per-scenario LCP-WS-L1 sweep** that produced the first empirical evidence of the planner's behaviour across nuPlan-mini's dynamic-scenario classes. It predates the rigorous comparative protocol under [`../13_protocol/`](../13_protocol/) — the data here is point estimates with one seed per scenario, useful as a sanity reference and as the source for the artefact pipeline ([`../artifacts/`](../artifacts/)).

## Producer

[examples/12_batch_two_level_mpc_planner.py](../../12_batch_two_level_mpc_planner.py). Each run invokes nuPlan's `run_simulation.py` once per scenario type via Hydra, loads the resulting `SimulationLog`, replays it through the [`RuleEngine`](../../../lexicone/observer/), and writes MP4 + per-tick CSV + episode-summary PNG via [`examples/visualizer.py`](../../visualizer.py)`render_episode`.

Re-generate the whole directory:

```bash
cd workspace
python examples/12_batch_two_level_mpc_planner.py \
    --penalty-form l1 --runtime-mode ws \
    --output-dir examples/outputs/12_batch_two_level_mpc_planner
```

(~50 min wall on a single CPU; one scenario takes ~3 min.)

## Files at the top level

| File | Source | Contents |
|---|---|---|
| `batch_summary_l1.csv` | Written at end of `12_batch_two_level_mpc_planner.py:425` when `--penalty-form l1` is set | One row per scenario: `label, description, scenario_type, status, duration_s, n_ticks, scenario_token, n_violating_rules, top_rule_id, top_integrated_violation, mp4_path` |
| `batch_summary.csv` | Same script, no `--penalty-form` flag (legacy MPC) | Same schema, from an earlier session's legacy-MPC sweep. Kept for historical comparison; superseded by the rigorous `13_protocol/C0_legacy/` data. |
| `batch_summary_extra_17.csv`, `…_extra_18.csv` | Same script with `--types <csv> --label-offset 17/18` | Custom one-off scenario sweeps; same schema. |

## Per-scenario subdirectories `<label>__l1/`

Sixteen subdirs, one per `DEFAULT_SCENARIOS` entry. The `__l1` suffix is the LCP-mode marker that `12_batch_two_level_mpc_planner.py:166` appends to keep LCP and legacy outputs separate.

Each subdir contains exactly four files plus a per-scenario `README.md`:

| File | Source | Contents |
|---|---|---|
| `<label>.mp4` | [`examples/visualizer.py`](../../visualizer.py) `_render_mp4` | Per-tick top-down animation: ego + agents + planned trajectory + lane geometry + sidebar (active rules, current tick metadata) + bottom violation strip. 10 fps. IEEE-grade typography. |
| `<label>.gif` | [`../../../mp4_to_gif_recursive.sh`](../../../mp4_to_gif_recursive.sh) (post-process) | GitHub-embedded GIF version of the MP4 (two-pass `ffmpeg` palette / paletteuse filter graph). 20 fps, 960 px wide. The per-scenario README below embeds this. |
| `<label>_log.csv` | `visualizer.py` `_write_csv_log` | One row per `(tick, rule)`: `t_s, rule_id, rule_name, applies, is_violated, violation_rate`. ~150 ticks × 25 rules = ~3750 rows. The raw data feeding every aggregate plot. |
| `<label>_summary.png` | `visualizer.py` `_render_summary` | Static episode summary: sorted integrated-violation bar chart + per-rule violation-rate heatmap + applicability-vs-violation count. 1-PNG-per-scenario overview. |
| `README.md` | hand-written | Scenario narrative: what's happening, what the LCP planner does, which rules fire and why, with the embedded GIF. |

## How to read the visualisation

Every MP4/GIF in this directory shares the same four-row layout (described in detail at [`../../visualizer.py`](../../visualizer.py)):

- **Header band (top).** Scenario label, current tick / wall time, ego state ($v$, $a_x$, $a_y$), and a status badge — green `✓ COMPLIANT` if no rule is firing at this tick, red `✗ N VIOL / M APPL` otherwise.
- **Map (left two-thirds).** Clean top-down view in the world frame. **Ego** is the red oriented rectangle with a short red heading line. **Agents** are coloured rectangles (vehicles green, pedestrians orange, cyclists purple, motorcycles blue, barriers brown, cones pink). **Planned trajectory** is the dashed dark-blue polyline emanating from the ego — the 3-second MPC prediction. **Ego trail** is the solid faded-red track behind the ego. **Lane geometry** is drawn in beige (drivable / travel lane / connector); crosswalks in cream-yellow; walkways in pale green; intersections in tan. **Traffic-light markers** are filled circles at the centre of the controlling lane connector, coloured red / yellow / green / grey by state. **Stop lines** are short red segments. A small `10 m` scale bar sits in the bottom-left of the map.
- **Sidebar (right third).** A **context card** (current lane id, posted $v_\text{lim}$, distance to next traffic light + state, distance to in-lane lead vehicle, applicable rule count) plus the **active-rules panel** — every rule whose `applies()` returned `True` at this tick, sorted by `violation_rate` descending, colour-coded by priority level (L10 dark red → L0 grey).
- **Violation strip (bottom).** A heatmap with rules on the y-axis and ticks on the x-axis; cell colour = `violation_rate` (white → red). A black vertical cursor tracks the current tick. Useful for spotting *when* during the episode each rule fires.

The colour code for priority levels is consistent across this dir, the [`../artifacts/violations/`](../artifacts/violations/) snapshots, and every figure under [`../artifacts/`](../artifacts/):

| L10 | L9 | L8 | L7 | L3 | L2 | L1 | L0 |
|---|---|---|---|---|---|---|---|
| dark red | red | orange | amber | teal | blue | purple | grey |
| Safety (VRU) | Safety (vehicle/surface) | Mandatory stop / yield | Legal (lane/light) | Comfort / headway | Route adherence | Priority / lateral | Comfort (long/lat) |

## 16-Scenario index table

Each row links to a per-scenario README with embedded GIF and a written description. The `Top rule` column is the rule with the largest integrated violation across the episode; the `∫ violation` column is its magnitude.

| # | Scenario | nuPlan type | Class | Top rule | ∫ violation | README |
|---|---|---|---|---|---|---|
| 01 | Following slow lead | `following_lane_with_slow_lead` | Overtake-style | `3r3` safe headway | 68.29 | [→](01_following_slow_lead__l1/README.md) |
| 02 | Near long vehicle | `near_long_vehicle` | Overtake-style | `7r2` opposing lane | 131.20 | [→](02_near_long_vehicle__l1/README.md) |
| 03 | Near multiple vehicles | `near_multiple_vehicles` | Overtake-style | `7r2` opposing lane | 316.19 | [→](03_near_multiple_vehicles__l1/README.md) |
| 04 | Changing lane (any) | `changing_lane` | Lane change | `2r2` route adherence | 44.70 | [→](04_changing_lane__l1/README.md) |
| 05 | Changing lane (left) | `changing_lane_to_left` | Lane change | `2r2` route adherence | 44.70 | [→](05_changing_lane_left__l1/README.md) |
| 06 | Changing lane (right) | `changing_lane_to_right` | Lane change | `2r2` route adherence | 44.90 | [→](06_changing_lane_right__l1/README.md) |
| 07 | Starting left turn | `starting_left_turn` | Turn (sharp) | `7r1` traffic-light | 47.99 | [→](07_starting_left_turn__l1/README.md) |
| 08 | Starting right turn | `starting_right_turn` | Turn | `2r2` route adherence | 44.99 | [→](08_starting_right_turn__l1/README.md) |
| 09 | High-speed turn | `starting_high_speed_turn` | Turn (high-speed) | `7r2` opposing lane | 90.79 | [→](09_high_speed_turn__l1/README.md) |
| 10 | Low-speed turn | `starting_low_speed_turn` | Turn (low-speed) | `7r2` opposing lane | 369.72 | [→](10_low_speed_turn__l1/README.md) |
| 11 | Protected cross | `starting_protected_cross_turn` | Turn (protected) | `7r2` opposing lane | 174.24 | [→](11_protected_cross__l1/README.md) |
| 12 | Unprotected cross | `starting_unprotected_cross_turn` | Turn (unprotected) | `7r2` opposing lane | 366.05 | [→](12_unprotected_cross__l1/README.md) |
| 13 | High-magnitude speed | `high_magnitude_speed` | Dynamic | `1r0` yield priority | 45.16 | [→](13_high_magnitude_speed__l1/README.md) |
| 14 | Medium-magnitude speed | `medium_magnitude_speed` | Dynamic | `3r0` speed limit | 48.80 | [→](14_medium_magnitude_speed__l1/README.md) |
| 15 | Near high-speed vehicle | `near_high_speed_vehicle` | Dynamic | `7r2` opposing lane | 104.94 | [→](15_near_high_speed_vehicle__l1/README.md) |
| 16 | Traversing intersection | `traversing_intersection` | Dynamic | `7r2` opposing lane | 186.39 | [→](16_traversing_intersection__l1/README.md) |

---

## Visual gallery — all 16 scenarios embedded

The GIFs below render inline on GitHub, on VS Code's markdown preview, and in any standard markdown viewer. Click each scenario heading to drill into the full per-scenario README with detailed narrative + rule analysis.

### [01 — Following Slow Lead](01_following_slow_lead__l1/README.md)
Overtake-style. Ego approaches a slower lead in the same lane; headway encoder (`3r3`) opens the L3 slack as the gap closes. Top violation: `3r3` safe headway (68.29).

![01 Following slow lead](01_following_slow_lead__l1/01_following_slow_lead.gif)

### [02 — Near Long Vehicle](02_near_long_vehicle__l1/README.md)
Overtake-style. Ego skirts a long vehicle; lane-corridor + opposing-lane constraints compete for slack. Top violation: `7r2` opposing lane (131.20).

![02 Near long vehicle](02_near_long_vehicle__l1/02_near_long_vehicle.gif)

### [03 — Near Multiple Vehicles](03_near_multiple_vehicles__l1/README.md)
Overtake-style, dense traffic. Multi-agent collision slack dominates; LCP priority structure pays off vs flat-weight. Top violation: `7r2` opposing lane (316.19).

![03 Near multiple vehicles](03_near_multiple_vehicles__l1/03_near_multiple_vehicles.gif)

### [04 — Changing Lane (any)](04_changing_lane__l1/README.md)
Lane change. Observer-only `2r2` dominates — route corridor and recorded path diverge during the merge. Top violation: `2r2` route adherence (44.70).

![04 Changing lane](04_changing_lane__l1/04_changing_lane.gif)

### [05 — Changing Lane to Left](05_changing_lane_left__l1/README.md)
Lane change to left. Same log as 04, directional variant; global planner reprojects onto target lane. Top violation: `2r2` route adherence (44.70).

![05 Changing lane left](05_changing_lane_left__l1/05_changing_lane_left.gif)

### [06 — Changing Lane to Right](06_changing_lane_right__l1/README.md)
Lane change to right. **Cleanest in the batch** (5 violating rules). Useful baseline for contrast. Top violation: `2r2` route adherence (44.90).

![06 Changing lane right](06_changing_lane_right__l1/06_changing_lane_right.gif)

### [07 — Starting Left Turn](07_starting_left_turn__l1/README.md)
Sharp left turn at a signalised intersection. **Highest violating-rule count** (14). Demonstrates the new `TrafficLightRule` + `OpposingLaneRule` encoders firing together. Top violation: `7r1` traffic light (47.99).

![07 Starting left turn](07_starting_left_turn__l1/07_starting_left_turn.gif)

### [08 — Starting Right Turn](08_starting_right_turn__l1/README.md)
Right turn. Simpler than 07 — no opposing-lane yield obligation. Top violation: `2r2` route adherence (44.99).

![08 Starting right turn](08_starting_right_turn__l1/08_starting_right_turn.gif)

### [09 — High-Speed Turn](09_high_speed_turn__l1/README.md)
High-speed cornering — proxy for ramp exit. Lateral-acceleration encoder (`1r11`) becomes binding at cruise. Top violation: `7r2` opposing lane (90.79).

![09 High-speed turn](09_high_speed_turn__l1/09_high_speed_turn.gif)

### [10 — Low-Speed Turn](10_low_speed_turn__l1/README.md)
Tight low-speed turn. **Largest single-rule violation in the batch** (`7r2` at 369.72) — sustained low-rate lane-edge clipping over many ticks. Top violation: `7r2` opposing lane (369.72).

![10 Low-speed turn](10_low_speed_turn__l1/10_low_speed_turn.gif)

### [11 — Protected Cross Turn](11_protected_cross__l1/README.md)
Turn at a protected intersection. Wait for green, then commit. Cleaner than the unprotected variant. Top violation: `7r2` opposing lane (174.24).

![11 Protected cross](11_protected_cross__l1/11_protected_cross.gif)

### [12 — Unprotected Cross Turn](12_unprotected_cross__l1/README.md)
Turn at an unprotected intersection. Creep-and-commit pattern; the archetypal yield-to-oncoming scenario. Top violation: `7r2` opposing lane (366.05).

![12 Unprotected cross](12_unprotected_cross__l1/12_unprotected_cross.gif)

### [13 — High-Magnitude Speed](13_high_magnitude_speed__l1/README.md)
Sustained high-speed open-road cruising. **Tied for cleanest** (5 violating rules). Shows the planner's behaviour when no rule encoders are stressed. Top violation: `1r0` yield priority (45.16, observer-only).

![13 High-magnitude speed](13_high_magnitude_speed__l1/13_high_magnitude_speed.gif)

### [14 — Medium-Magnitude Speed](14_medium_magnitude_speed__l1/README.md)
Mid-speed cruising. **Only scenario** where `3r0` SpeedLimit is the top violation — transient over-shoot of the posted limit. Top violation: `3r0` speed limit (48.80).

![14 Medium-magnitude speed](14_medium_magnitude_speed__l1/14_medium_magnitude_speed.gif)

### [15 — Near High-Speed Vehicle](15_near_high_speed_vehicle__l1/README.md)
High relative-speed encounter. Lateral keep-out tightens fast; safety > legal trade-off engages. Top violation: `7r2` opposing lane (104.94).

![15 Near high-speed vehicle](15_near_high_speed_vehicle__l1/15_near_high_speed_vehicle.gif)

### [16 — Traversing Intersection](16_traversing_intersection__l1/README.md)
Through-pass of a full intersection. **Showcase scenario** — multiple rule encoders active simultaneously. Cross-references the visualiser redesign and the comparative-protocol expectation. Top violation: `7r2` opposing lane (186.39).

![16 Traversing intersection](16_traversing_intersection__l1/16_traversing_intersection.gif)

## Status

These outputs are **point estimates from one seed per scenario** (`--seed 7` advancing by 1 per scenario position). They are not the basis for the paper's empirical claims — those use the multi-seed comparative protocol under [`../13_protocol/`](../13_protocol/). The data here is preserved because:

1. The artefact pipeline ([../artifacts/](../artifacts/)) was built and validated against this batch.
2. The 16 per-scenario MP4s are useful for qualitative inspection of LCP behaviour.
3. The 16 `*_summary.png` plots are the per-scenario reference cards.
