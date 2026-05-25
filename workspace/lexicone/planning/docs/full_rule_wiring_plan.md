# Wiring all 25 observer rules into the LCP MPC — full plan

> **Update (offline simulation, no real-time constraint).** Per-tick compute
> budget is no longer the binding constraint. This changes the design centrally:
> we **stop approximating** the lex cascade with a weighted-sum solve and
> **just run the cascade** at every MPC tick. That removes Algorithm 1A/1B and
> the calibration cache from the critical path (they become validation /
> deployment-time tools only), eliminates the "WS weights don't match lex
> trade-offs" failure mode the user observed, and lets us spend the freed
> budget on quality (more SLP iterations, longer horizon, denser constraint
> sets, proper OR-of-half-planes). See §13 "Offline-budget revision" for the
> reordered phases and what changes per phase.

## 0. Problem framing

The observer evaluates 25 driving rules per tick. The MPC currently encodes 6 (`9r0`+`10r0` collision, `7r0` lane corridor, `3r0` speed limit, `3r3` headway, `0r2` longitudinal comfort, `1r11` lateral acceleration). The other 19 rules are **judged but not constrained**. The aggregate-violation 17% improvement we measured after the lane-corridor fix is the cap of what's reachable without expanding the rule wiring further.

This plan turns that 6 into 25 (where 25 is structurally possible — some rules are inherently behavioural and outside MPC scope; we'll be honest about that). The plan also designs the **offline calibration runner** that populates the `CalibrationCache` with paper-grade Algorithm 1A / 1B weights — without it, the WS we run is *not* lex-equivalent and trade-offs across levels aren't formally guaranteed.

Three orthogonal axes of work, sequenced together:

1. **Per-rule encoding** — add the missing 19 rule encoders, each with the right per-step applicability mask and linearisation.
2. **Data plumbing** — extend `MapLifter` / `EncoderContext` to carry the per-step semantic state every rule needs (oncoming-lane labels, per-step traffic-light projection, crosswalk occupancy, etc.).
3. **Cross-tick state** — wrap rules whose semantics span multiple ticks (mandatory stop, crosswalk yield approach phase) in a `RuleStateManager` that survives across `compute_planner_trajectory` calls.

Plus two pieces that are not rules but unblock the framework's formal guarantees:

4. **Behavioural layer** — for `2r2`, `1r2`, `1r5`, `10r3`, `1r0`, none of which are reachable from a single-shooting MPC.
5. **Offline calibration runner** — runs the lex cascade + Algorithm 1A/1B per scenario class and writes the cache.

---

## 1. Rule-by-rule classification

Every rule falls into one of six encoding patterns. The classification drives both the implementation order and which infrastructure changes each requires.

### Pattern A — Already encoded (6 rules, no work)
| ID | Class | Notes |
|---|---|---|
| 9r0 + 10r0 | `CollisionRule` | Quadratic, linearised at warm-start. VRUs use inflated radius. |
| 7r0 | `LaneCorridorRule` | Two linear half-planes/step around route centerline. |
| 3r0 | `SpeedLimitRule` | Linear `v − v_lim ≤ 0`. |
| 3r3 | `SafeHeadwayRule` | Linearised gap to in-lane lead. |
| 0r2 | `LongitudinalComfortRule` | Linear box on `u_a`. |
| 1r11 | `LateralAccelerationRule` | Linearised `v² tan δ / L`. |

### Pattern B — Linear half-plane vs polygon (5 rules)
Stay outside (or inside) a polygon. Encoded by half-planes; the SLP loop re-picks which half-plane is binding around the warm-start each iteration.

| ID | Encoding | Data needed |
|---|---|---|
| **7r5 Sidewalk drive** | Stay outside walkway polygons. For each step, find the **closest walkway polygon** to warm-start `(x̄_k, ȳ_k)`. From its half-planes, pick the one with the largest *positive* margin (i.e., the half-plane the ego is currently most-outside) and encode `−n·p + d + r_ego ≤ t`. Multiple SLP iterations + the lazy 1-half-plane-per-polygon trick are equivalent to OR-of-half-planes near the current iterate. | `MapHorizonView.walkways` (already populated). |
| **10r5 Bike lane** | Same as 7r5 but with bike-lane polygons. **No-op on nuPlan-mini** (bike-lane layer not exposed). Reserve the encoder + slots so it works on datasets that do. | Future: bike-lane layer from a richer map provider. |
| **7r2 Opposing lane** | Stay outside any lane whose centerline heading is **opposite** to the ego's heading at the same arc-length region. Per step: enumerate `MapHorizonView.nonroute_lanes ∪ MapHorizonView.lane_connectors`, classify each as "oncoming" if `|principal_value(lane_heading − ego_heading)| > π − tol`, and encode stay-outside half-planes for the closest K. | `MapLifter` already returns lane headings; add `is_oncoming_at(point, ego_heading)` helper. |
| **7r3 One-way directionality** | Same data as 7r2, looser predicate: stay outside any lane whose centerline heading differs from ego's by more than 90°. Effectively a superset of 7r2. Encoded with the same slots; can share. | Same as 7r2. |
| **3r5 Lateral clearance** | `d_min_lat(v_rel_y) − |y_k − y_jk| ≤ 0` for each adjacent agent j. Absolute-value handled by *sign-of-y-at-warm-start*: at warm-start the ego is on one side of the agent, encode that single inequality; the SLP loop catches the wrap. Per side, one constraint per adjacent agent within `lateral_band_m`. | Already in `EncoderContext.agents_local`. |

### Pattern C — Linear constraint with per-step applicability mask (3 rules)
The constraint itself is linear, but it applies only over a *subrange* of horizon steps determined by predicted scene state.

| ID | Encoding | Mask source |
|---|---|---|
| **7r1 Traffic light** | If the connector ahead is RED, ego must not cross its stop line. Per step k, if the predicted TL state at `t + k·dt` is RED *and* the warm-start position `x̄_k` is before the stop line projection s_stop along the reference, set `mask=1` and encode `s_k − s_stop ≤ t`, otherwise `mask=0`. | Predicted TL state per step (see §3 below). Need stop-line→connector association. |
| **7r4 Stop in crosswalk** | If at step k the warm-start footprint overlaps a crosswalk polygon, the ego must NOT be near-stopped: `v_thresh − v_k ≤ t` with `mask=1`. Otherwise `mask=0`. | Per-step crosswalk overlap predicate from warm-start trajectory + crosswalk polygons. |
| **3r6 Lane intrusion** | For each adjacent vehicle, lateral TTC ≤ threshold triggers a "give-room" constraint. Per step, evaluate predicted lateral approach rate at warm-start; if `TTC < T_min`, mask=1 and encode lateral-distance lower bound. | Already in `EncoderContext.agents_local`; needs per-step interpolation of agent trajectory. |

### Pattern D — Multi-tick state machine (4 rules)
The rule's semantics span multiple MPC ticks. We need a `RuleStateManager` that owns a small FSM per active rule instance and persists across `compute_planner_trajectory` calls.

| ID | FSM | Encoding |
|---|---|---|
| **8r0 Mandatory stop** | States: `APPROACHING` → `STOPPED` (must dwell ≥ `t_stop_min`) → `DEPARTED`. Entry: ego approaches a stop sign / red-light stop line within `lookahead_m`. Exit: ego has completed full-stop dwell and is past the line. | `APPROACHING`: enforce per-step `s_k ≤ s_stop` (mask=1 until step k_cross). `STOPPED`: enforce `v_k ≤ v_stop` until dwell satisfied. `DEPARTED`: rule inactive. |
| **8r1 Crosswalk ped yield** | States: `NONE` → `PED_PRESENT` (mask active across crosswalk approach) → `PED_CLEAR`. | Same speed-bound encoding as 7r4 but the mask is gated on observer's pedestrian-on-crosswalk detection (not the ego's overlap). |
| **10r3 Unmarked crosswalk yield** | Same FSM shape as 8r1; the "crosswalk" is the implicit zone at intersections without painted crosswalks. | Same as 8r1; the implicit zone is generated by the MapLifter from intersection polygons + walkway endpoints. |
| **10r4 Cyclist passing** | States: `IDLE` → `APPROACHING_CYCLIST` (lateral clearance grows toward `d_min_pass`) → `PASSING` (lateral clearance enforced) → `RETURNED`. | Pass-phase encoding is a stronger 3r5 with the cyclist agent specifically tagged. |

The `RuleStateManager` is a per-`TwoLevelMPCPlanner`-instance object; it gets called in `compute_planner_trajectory` *before* `RuleSet.encode_all` and writes the per-step masks into the `EncoderContext`.

### Pattern E — Inherently behavioural / NOT MPC-encodable (3 rules)
The decision is *which lane to occupy* or *which agent to defer to* — a discrete choice that an MPC inside a single shooting interval cannot make. These need a **behavioural layer above the MPC**.

| ID | Why not MPC | Where it goes |
|---|---|---|
| **2r2 Route adherence** | The global planner picks one lane through the route corridor; the observer fires `2r2` whenever the ego is not on the *recorded expert's* lane. The MPC has no way to know which lane the expert chose. | Behavioural layer needs to (a) ingest the scenario's `route_roadblock_ids` *and* a tighter "expert lane sequence" hint, or (b) accept that `2r2` will fire on every off-route lane (benign). |
| **1r2 Block the box** | Multi-tick reasoning: "don't stop in intersection without downstream gap clear". Requires predicting downstream traffic flow over ≥10 s. | Behavioural: a "go / no-go" gate that triggers an MPC speed cap before the intersection if downstream is congested. |
| **1r5 Uncontrolled intersection** | Game-theoretic priority negotiation with cross-traffic. MPC can't decide "I yield". | Behavioural: pre-MPC priority arbiter that sets a speed cap until cross-traffic clears. |
| **1r0 Yield priority** | Same flavour as 1r5 — multi-agent priority. | Behavioural arbiter (shared with 1r5). |

The behavioural layer is a new module — see §4 for sketch.

### Pattern F — Comfort (1 rule)
| ID | Encoding |
|---|---|
| **0r3 Lateral comfort** | Linear box on `Δδ` (steering rate). Already partially enforced via the hard steering-rate-box; we'd add a soft version with the comfort threshold below the hard limit. |

### Pattern G — Subsumed by others (2 rules)
| ID | Why no separate encoder |
|---|---|
| **9r1 Non-traversable surface** | Subsumed by `7r0 LaneCorridorRule` (corridor IS the drivable surface) and `7r5 SidewalkDriveRule`. The observer counts them separately but no additional planner-side constraint is needed. |
| **10r0 VRU collision** | Subsumed into `CollisionRule` with VRU radius inflation. |

### Tally
- **Phase 1 (Pattern B + F)**: 6 new encoders (7r2, 7r3, 7r5, 10r5, 3r5, 0r3) — ~2 weeks.
- **Phase 2 (Pattern C)**: 3 new encoders + per-step mask plumbing (7r1, 7r4, 3r6) — ~1.5 weeks.
- **Phase 3 (Pattern D)**: 4 new encoders + `RuleStateManager` (8r0, 8r1, 10r3, 10r4) — ~2 weeks.
- **Phase 4 (Pattern E)**: behavioural layer for 4 rules (2r2, 1r0, 1r2, 1r5) — ~3 weeks; out of scope for the trajectory-level MPC but in scope for the planner-as-a-whole.
- **Phase 5 (Calibration runner)**: offline lex cascade + Algorithm 1A/1B per scenario class, cache population — ~1 week.
- **Phase 6 (Multi-active Algorithm 1B)**: extend `compute_l2_sensitivity_constants` to handle multiple boundary-binding actives per level — ~3 days.

**Total: 9–10 weeks of focused engineering.**

---

## 2. Required `MapLifter` extensions

```text
LocalLane                                    (already present)
  + heading_at_arc_length(s) → float        (new helper)
  + width_along_lane(s) → float             (new — for adaptive corridor width)
  + parent_intersection_id                  (new — links lane connectors to their intersection)

MapHorizonView
  + crosswalk_zones: Tuple[LocalCrosswalkZone]  (new — extends crosswalks with stop-line projection)
  + traffic_light_schedule: Dict[connector_id, Tuple[(state, valid_until_s), ...]]
                                                (new — projects TL state over the horizon)
  + intersection_ego_overlap: Tuple[(t_enter, t_exit)]
                                                (new — when does the warm-start trajectory enter/exit each intersection)

LocalCrosswalkZone                                (new)
  polygon_local: LocalPolygon
  associated_stop_line_local: Optional[LocalStopLine]
  current_pedestrian_count: int                  (from observation, not map)

LocalStopLine                                    (already present, extend)
  + projected_arc_length_along_reference: float  (new — needed by 7r1)
```

**Traffic-light schedule projection** is the hardest map extension. nuPlan exposes the *current* TL state per connector at each iteration; for the MPC horizon (3 s) we need to project forward. Options:

- (a) Assume current state holds across the horizon (conservative — RED stays RED). Easy, biases the planner to over-stop.
- (b) Use the simulator's known TL phase schedule (when reading from a saved log, we can peek ahead). Best fidelity but requires plumbing the schedule through the `PlannerInput`.
- (c) Learn a transition model from data. Out of scope.

**Recommendation**: (a) for v0.5, (b) when wiring 8r0 / 8r1 since they need accurate phase predictions.

---

## 3. `EncoderContext` and per-step-mask plumbing

Today `EncoderContext` is rule-agnostic. To support Pattern C and D rules, add per-step semantic state that encoders can consult when deciding `mask`.

```python
@dataclass
class EncoderContext:
    # ... existing fields ...
    # Per-step predicates:
    horizon_steps_in_intersection: Tuple[bool, ...]         # length N
    horizon_steps_on_crosswalk: Tuple[bool, ...]            # length N
    horizon_steps_red_light_ahead: Tuple[bool, ...]         # length N — RED TL projected
    horizon_steps_pedestrian_in_crosswalk: Tuple[bool, ...] # length N
    # Per-step semantic state:
    horizon_arc_lengths: np.ndarray                         # (N+1,) — warm-start s along reference
    nearest_lane_per_step: Tuple[LocalLane, ...]            # length N
```

The orchestrator (`TwoLevelMPCPlanner._solve_lcp_tick`) computes these once per tick from the warm-start trajectory and the map view, and passes them into the `EncoderContext`. Encoders read them as masks.

**Cost of populating these per tick**: O(N × (|map_polygons| + |agents|)) ≈ 30 × 50 ≈ 1500 polygon-vs-point checks. Cheap; well under 5 ms.

---

## 4. `RuleStateManager` — cross-tick state for Pattern D rules

```python
class RuleStateManager:
    """Owns one FSM per active multi-tick rule instance.

    Methods:
        on_tick(ctx) -> None
            Advances every FSM. Updates ctx.horizon_*_per_step masks for
            FSM-dependent rules (e.g. mandatory-stop's APPROACHING phase
            enforces s_k ≤ s_stop; STOPPED phase enforces v_k ≤ v_stop).
        get_active_fsms() -> List[(rule_id, state_name, params)]
            For diagnostics / logging.
    """
    _mandatory_stop_fsms: Dict[stop_line_id, MandatoryStopFSM]
    _crosswalk_yield_fsms: Dict[crosswalk_id, CrosswalkYieldFSM]
    _cyclist_pass_fsms: Dict[cyclist_track_id, CyclistPassFSM]
```

The FSMs are persisted on the `TwoLevelMPCPlanner` instance (it lives for the whole scenario). Each FSM has its own transition logic with hysteresis (stop signs need a *dwell time* in STOPPED; cyclist passes need a *post-pass clearance* before transitioning to RETURNED). The FSM state is updated *before* `RuleSet.encode_all(ctx)` is called, so every encoder sees a consistent mask set.

Pickling: the FSM dataclasses are plain; they pickle without help. The `RuleStateManager` itself is plain Python.

---

## 5. Behavioural layer for Pattern E rules

The behavioural layer sits *above* the MPC and *below* the global route planner. It does two things:

1. **Manoeuvre selection**: given the current scene, decide *which lane* the ego should be in (lane keep / change left / change right / merge / turn). The output is a "preferred lane id" that the global planner uses to bias its BFS, replacing the current "first reachable" heuristic.
2. **Yield arbitration**: given other agents at the intersection / merge, decide *whether the ego yields* (sets a pre-intersection speed cap) or *takes the right of way* (lets MPC proceed).

This is a new module — `lexicone/planning/behaviour_layer.py` — that runs once per global-route replan tick (~5 s cadence). It produces:

```python
@dataclass
class BehaviourCommand:
    preferred_lane_id: Optional[str]      # lane the global planner should bias toward
    pre_intersection_speed_cap: Optional[float]   # m/s — bound active until ego clears intersection
    yield_to_track_ids: Tuple[str, ...]   # agents whose path the ego must not cross
```

The MPC consumes the speed cap as an additional `v_cap` term and `yield_to_track_ids` as a per-agent priority hint that the CollisionRule's slot ranking respects.

This layer is **research-grade**, not a clean engineering deliverable — designing a good behavioural arbiter is itself a major project. A reasonable v0 uses simple heuristics:
- Manoeuvre: stay in the lane the ego is currently in unless the global route forces a change.
- Yield arbiter: if any agent within 30 m has a heading orthogonal to ego's at an intersection, set a 5 m/s speed cap until the agent clears.

This v0 won't pass detailed `1r0` / `1r5` evaluations but is enough to fix `2r2` in the "expert chose this lane" case.

---

## 6. Offline calibration runner

The cache stays empty unless we explicitly populate it. The runner:

```text
For each scenario class C in DYNAMIC_SCENARIO_TYPES:
    1. Pick a representative scenario: first match for class C in the mini split.
    2. Run the closed-loop simulator with the LEGACY planner up to a "median" tick
       (e.g., tick 75 of 150). Record the (ego_state, agents, map_view, reference)
       snapshot.
    3. Build LCP rule_pack at that tick (using the fully-wired RuleSet).
    4. Run lex_cascade.run_cascade(...) → CascadeResult with p*, active_set, T_lex.
    5. Extract structured gradients from CasADi autodiff at z_lex* by re-evaluating
       each active rule encoder's symbolic Jacobian. Pack into WeightCalibrationInputs.
    6. Call weight_calibration.algorithm_1a(inputs) → WeightCalibrationResult.
    7. Solve the WS MPC at w† and verify b_eps_lex matches.
    8. Store (scenario_class=C, penalty_form="l1", weights=w†, b_eps_lex,
              cascade_p_star=p*, computed_at=now) in lcp_cache.json.
```

This is a separate executable: `workspace/scripts/offline_lcp_calibration.py`. Estimated runtime: ~5 min per scenario class × ~30 classes ≈ 2.5 hours one-time. Subsequent demo runs use the cached weights.

**Re-calibration trigger**: when `ComplianceChecker` reports an active-set mismatch on > 20% of ticks for a scenario class, the cache entry is invalidated and a background re-calibration runs on the next idle window.

**Multi-active extension to Algorithm 1B**: the current `compute_l2_sensitivity_constants` falls back to single-active when multiple actives are present, dropping the extras with a warning. For real nuPlan scenarios with 5–10 simultaneously-active collision constraints at level 1, we need the full reduced KKT linear system from Section 9.4 Step B2. Implementation: stack the boundary-binding gradients into a matrix `G`, compute `(G^T G)^{-1} G^T ∇J` and `(G^T G)^{-1} G^T ∇V_{i'}` for each violated level i'. This is ~50 lines.

---

## 7. Per-tick computational budget

At 10 Hz we have 100 ms per tick. The legacy MPC sustains ~30 ms per tick. With all rules wired:

| Stage | Estimated cost |
|---|---|
| `MapLifter.view()` | 15 ms (polygon transforms + half-plane reduction) |
| `RuleStateManager.on_tick()` | 5 ms (a few FSM advances) |
| Per-step semantic predicates | 5 ms |
| `RuleSet.encode_all()` (25 rules × 30 steps) | 30 ms |
| `LCPTrajectoryPlanner.push_parameters()` | 10 ms |
| IPOPT solve (1 SLP iter) | 40 ms |
| Optional 2nd SLP iter | 40 ms |
| Compliance check | 5 ms |
| **Total (1 SLP iter)** | **110 ms** |
| **Total (2 SLP iter)** | **150 ms** |

We're at-or-over budget. Mitigations:

- **Cache the `MapLifter.view()` across ticks** when the ego hasn't moved > 5 m. Drops the per-tick map cost to 2 ms.
- **Skip the second SLP iteration** when the first iteration's `max_step_size` is below `0.5 m / 0.05 rad`.
- **Sparse parameter updates**: only push parameters for rules whose `applies_to_horizon` flipped vs the previous tick.
- **Pre-build the OCP per scenario class** so the per-tick `push_parameters` is a memcpy.

With these, we should hold 70–90 ms/tick consistently — within the 10 Hz budget.

---

## 8. Tooling and diagnostics

To debug the wired rules, three new pieces:

1. **Per-tick rule contribution log** — for each rule, log the slack value summed over the horizon. Lets us spot which rule is driving the MPC's choices.
2. **Per-scenario rule heatmap** — extend the visualiser to draw a row-per-rule heatmap of slack-over-time alongside the existing observer-violation strip. Discrepancies between "what the planner thought it was violating" and "what the observer detected" are the highest-signal debug.
3. **Cascade vs WS A/B harness** — a per-tick toggle that solves the LCP MPC twice (once via cascade, once via WS at `w†`) and reports the trajectory delta. Confirms the cache-stored `w†` actually achieves lex equivalence.

These ship as part of the visualiser / a new `lcp_diagnostics.py`.

---

## 9. Phased delivery

| Phase | Scope | Estimated time | Demo-time evidence |
|---|---|---|---|
| **1a** | Add 7r2 + 7r3 + 7r5 + 10r5 (Pattern B half-plane rules) | 1 week | Top violation on intersection-turn scenarios stops being `7r2` |
| **1b** | Add 3r5 + 0r3 (Pattern B + F) | 3 days | Smoother trajectories on dense-traffic scenarios |
| **2** | Per-step mask plumbing + 7r1 + 7r4 + 3r6 (Pattern C) | 1.5 weeks | TL-compliant behaviour; no crosswalk-dwell |
| **3** | `RuleStateManager` + 8r0 + 8r1 + 10r3 + 10r4 (Pattern D) | 2 weeks | Stop signs are stopped at; ped yields clean |
| **4** | Offline-calibration runner + multi-active Alg 1B | 1.5 weeks | Cache populated; `b_eps_lex` mismatch rate < 5% |
| **5** | Behavioural-layer v0 (Pattern E, heuristics-only) | 2 weeks | `2r2` drops dramatically when route hint is correct |
| **6** | Diagnostics + per-tick rule heatmap | 3 days | Each remaining violation is attributable to a specific rule and constraint |
| **7** | Phase-1 to Phase-6 integration regression on all 16 scenarios | 1 week | LCP integrated violations cut > 70% vs current legacy |

**Total: ~10 weeks single-engineer focused work.**

A useful shorter milestone: Phase 1a + 1b + 2 (3 weeks) alone wires 9 more rules and should bring intersection-turn `7r2` violations to near-zero — that's the visible failure mode in the current batch.

---

## 10. Verification gates

Each phase must pass:

1. **Unit tests for every new encoder**: positive case (rule applies → constraint produced with correct (a, b, e, mask)), negative case (`applies_to_horizon` returns False under expected conditions), boundary case (the rule's threshold is exactly at the encoded inequality).
2. **Synthetic-OCP test**: run the lex cascade on a hand-built scenario where the new rule should bind (e.g., for 7r1, build a synthetic stop line at x=10 with TL=RED; assert `V_legal* > 0` and the trajectory's terminal x ≤ 10).
3. **Pipeline integration**: run `examples/12_batch_two_level_mpc_planner.py --penalty-form l1 --limit 2` on the first two scenarios and confirm:
   - No fall-back to legacy in `_solve_lcp_tick`.
   - The newly-wired rule shows non-trivial slack in the new diagnostics log.
4. **Full regression**: re-run all 16 scenarios, diff vs `batch_summary.csv`, expect monotonic improvement on the rules just wired.
5. **Performance regression**: per-tick wall-clock stays < 100 ms median, < 150 ms 99th percentile.

---

## 11. Risks and known structural limitations

1. **`2r2` is not solvable inside the planning module.** Even a perfect behavioural layer can only guess at "which lane did the expert choose" — the recorded route doesn't tell us. The best we can do is heuristic ("the lane closest to mission_goal at each roadblock"). Some `2r2` violations are irreducible.

2. **Map-data fidelity is the bottleneck.** nuPlan-mini does not expose bike lanes, gives only the *current* TL state (no schedule), and doesn't explicitly mark "oncoming" lanes — we infer from heading. For deployment on a richer map provider these would be data-side improvements rather than algorithmic work.

3. **The SLP linearisation has a finite trust region.** For very sharp turns (low-speed U-turns), the warm-start may be far from the actual route geometry; the linearised dynamics may produce unsafe trajectories that satisfy the linearised constraints. Mitigation: shrink the trust region adaptively when the cascade reports `max_dynamics_residual > 0.5 m`, and iterate up to 5 SLP outer iterations.

4. **Algorithm 1B's single-active reduction may misestimate `κ_i` on intersection scenarios** where multiple boundary-binding actives are typical (collision + lane corridor + headway often bind together). The multi-active extension is in Phase 4, but until it ships, the L₂ tolerance-compliance regime gives weaker guarantees than the paper formally promises.

5. **Behavioural layer scope creep.** The "yield priority arbiter" (1r0, 1r5) genuinely needs game-theoretic reasoning to do well — a heuristic v0 will fix some cases and worsen others. A clean implementation may require POMDP-style belief tracking; that's beyond the LCP framework entirely.

---

## 12. Where the lex cascade stops being meaningful

The LCP framework gives formal lex guarantees *within* the rules the planner can constrain on. Rules in Patterns E (behavioural) are outside that scope: the planner cannot enforce them and Algorithm 1A's KKT does not include their gradients. Their violations will continue to appear in the observer report, and that's the correct epistemic state — the planner is honest about what it can and cannot do.

This is why a "near-zero violation" demo run is *not* the right success criterion. The right criterion is:

- For every rule the planner does encode: the runtime compliance check reports < 5% mismatch rate against the cached `b_eps_lex`.
- For rules outside the planner's scope: the violation count is *bounded* by the behavioural-layer policy, and the policy's choices are *auditable* (the user can read the behavioural log and ask "why did the planner yield here?").

In other words, the goal is **a planner whose trajectory is provably lex-optimal within the rule set it encodes**, with the rules-it-cannot-encode explicitly documented as out-of-scope. Anything weaker is the failure mode the user just observed.

---

## 13. Offline-budget revision

The original plan assumed 10 Hz real-time deployment; the per-tick budget drove
the choice of "WS scalarisation with Algorithm 1A/1B-calibrated weights as a
substitute for the L+1-stage cascade". The user's clarification removes that
constraint: **the demos are offline**, so per-tick wall-clock is bounded only
by total batch runtime, not by tick cadence.

This is a substantial design shift. The cleanest revision is to **drop the WS
approximation as the runtime path and run the lex cascade directly at every
tick.**

### 13.1 What the cascade-per-tick mode buys

- **Formal lex-optimality without calibration**. The cascade *is* the lex
  optimum by construction (eq. 2 of the paper). No Algorithm 1A/1B, no
  calibration cache, no `b_eps_lex` mismatch failure mode. Whatever trade-offs
  the planner makes are demonstrably the lex-correct ones, not WS
  approximations of them.
- **No "heuristic-weights" excuse for over-permissive trade-offs**. Today the
  planner pays `w · t` for slack at *every* level simultaneously and the joint
  minimum can violate a high-priority constraint to relieve a low-priority
  one if the weights are mis-set. Under cascade, level i's slack is *bounded*
  by `V_i* + δ_lex` from the cascade's i-th stage — a hard constraint, not a
  weighted cost.
- **Algorithm 1A/1B become optional validation tools.** They stay in the
  codebase as paper-grade reference implementations (and for the eventual
  online-deployment story), but they are no longer on the critical path.
  The work to wire the offline calibration runner (originally Phase 4) is
  deferred or dropped entirely.
- **Quality knobs go up**: more SLP iterations, longer horizon, denser
  obstacle/half-plane budgets — see §13.3.

### 13.2 What changes in the existing architecture

The architecture from Phase 0–C is mostly preserved; the per-tick code path
switches mode:

```python
# Old: WS solve at calibrated w†
self._lcp_planner.push_parameters(...)
X_local, U_sol, T_sol = self._lcp_planner.solve_once()

# New: lex cascade per tick
cascade_result = lex_cascade.run_cascade(
    base_params=self._lcp_planner._params,
    vehicle_parameters=self._vehicle_parameters,
    affine_steps=affine_steps,
    x0_local=x0_local,
    u_prev=u_prev,
    v_cap=v_cap,
    Xref_local=Xref_local,
    rule_pack=rule_pack,
    delta_lex=self._cascade_delta_lex,
)
X_local = cascade_result.z_lex_X
U_sol   = cascade_result.z_lex_U
T_sol   = cascade_result.T_lex
```

This is one orchestrator change in `TwoLevelMPCPlanner._solve_lcp_tick`. The
config gets a new key:

```yaml
penalty_form: l1              # still controls the V_i penalty form
runtime_mode: cascade         # cascade | ws — defaults to cascade for offline
```

When `runtime_mode: ws` is chosen, the existing path runs (with the
calibration cache); when `cascade`, the new path runs. Both share the rule
encoders, the SLP linearisation, the map lifter — only the per-tick OCP
solving strategy differs.

The reactive-trigger "if cascade reports `V_i* > 0` for an unexpected level,
log and continue" replaces the old compliance-mismatch path.

### 13.3 Quality knobs we can now afford to crank

The original §7 performance budget projected ~110 ms / tick at 1 SLP
iteration and called out optimisations to stay under 100 ms. With per-tick
budgets relaxed to ~5 s, we can:

- **Run 3–5 SLP outer iterations per tick** until `sqp_convergence_metric`
  reports `max_dynamics_residual < 0.05 m` (was: 1 iteration, no
  convergence check). Eliminates the linearisation-drift failure mode that
  contributes to the current observer mismatch.
- **Extend the horizon to 5 s** (was 3 s) — 50 steps. The MPC anticipates
  TLs / stop signs / sharp turns much earlier; the global planner's
  reference and the LCP MPC's solution overlap more before drift.
- **Increase obstacle slot count to 20** (was 8) — sufficient for the
  densest nuPlan scenarios; collision constraints never get dropped.
- **Use proper OR-of-half-planes for "stay outside" rules** (7r5, 10r5).
  Encode as: for each polygon, a binary decision variable per half-plane
  with big-M relaxation; at least one half-plane's stay-outside inequality
  must hold. Increases problem size but stays tractable for offline solves.
  Removes the current single-half-plane shortcut that caused the original
  "sidewalk drift" bug.
- **Dense per-rule diagnostics** (originally Phase 6 with budget concerns)
  — log every rule's slack at every tick, plus per-cascade-stage `V_i*`,
  active-set fingerprint, and SLP convergence statistics. Storage cost is
  negligible.

### 13.4 Reordered phasing under offline budget

| Phase | Scope | Estimated time | What's different from original |
|---|---|---|---|
| **1a** | 7r2 + 7r3 + 7r5 + 10r5 with proper OR-of-half-planes | 1.5 wk | Was 1 wk; OR-of-half-planes adds 3 days |
| **1b** | 3r5 + 0r3 | 3 d | Unchanged |
| **2** | 7r1 + 7r4 + 3r6 + per-step mask plumbing | 1.5 wk | Unchanged |
| **3** | RuleStateManager + 8r0 + 8r1 + 10r3 + 10r4 | 2 wk | Unchanged |
| **4'** | **Switch runtime path to cascade-per-tick + SLP iterate-to-convergence + horizon-extension** | 3 d | NEW. Replaces the old "offline calibration runner + multi-active Algorithm 1B" phase. |
| **5** | Behavioural-layer v0 (2r2, 1r0, 1r2, 1r5) | 2 wk | Unchanged |
| **6** | Diagnostics + per-tick rule heatmap + per-cascade-stage logging | 5 d | Was 3 d; expanded to log cascade stage data |
| **7** | Full regression on all 16 scenarios | 1 wk | Unchanged |

**New total: ~9 weeks** (was ~10). The original Phase 4 (1.5 weeks of calibration-runner work) is replaced by a 3-day cascade switchover.

The 1-week milestone (Phase 4' alone, after Phase 0–C) is worth highlighting:
**switching the existing code to cascade-per-tick + extending the horizon +
running iterate-to-convergence SLP, on the rules already wired, should
produce visible improvement on the intersection-turn failures the user
observed** without any new rule encoders. It's the cheapest experiment that
isolates "did the WS approximation cause the disaster vs the missing
encoders?".

### 13.5 What the WS / Algorithm 1A/1B work stays around for

- **Paper validation** — the Example 1 / Example 2 replication tests still pass
  and demonstrate the implementation matches the paper. Don't delete.
- **Eventual online deployment** — if a future deployment needs 10 Hz, the
  cascade is too slow; switch back to WS at calibrated weights. The Phase 4
  calibration runner becomes the relevant work item at that point.
- **Diagnostic comparison** — `CascadeVsWS` A/B harness (§8) becomes a tool
  for diagnosing whether the WS approximation would have agreed with the
  cascade on a given scenario class. If WS and cascade disagree significantly,
  the active set is unstable across the cascade stages and the lex active
  set is poorly behaved — useful research signal.

### 13.6 New risk: cascade infeasibility under SLP drift

The cascade's inner stages constrain `V_{i'} ≤ V_{i'}* + δ_lex` for
`i' < i`. Under SLP linearisation, the linearised `V_i*` from stage 1 may be
infeasible at the *true* (nonlinear) trajectory generated in stage 2. With
`δ_lex = 1e-6` (the paper's recommendation for convex programs), the inner
stages can become numerically infeasible on real nuPlan scenarios where the
linearisation drift exceeds the slack.

**Mitigation**:
1. Adaptive `δ_lex`: start with `1e-6`; if stage i reports infeasibility,
   bump to `1e-3` and re-solve; report a warning if `1e-2` is needed.
2. SLP iterate-to-convergence at each stage, not just the final stage —
   the inner stages also benefit from outer-iteration refinement.
3. On final-stage infeasibility, fall back to WS-with-heuristic-weights for
   that tick and log the fall-back rate. Acceptable if < 1% of ticks.

These are routine engineering for SQP-on-nonlinear-dynamics; nothing
research-grade. They go into Phase 4'.

---

## 14. Recommended path forward given offline budget

1. **Immediately**: Phase 4' alone — flip the orchestrator to cascade-per-tick,
   extend horizon to 5 s, run 3 SLP iterations per tick. Re-run the 16-scenario
   batch. This isolates "did cascade help by itself?" from "did adding rules
   help?". Expected: significant reduction in the two worsened scenarios (10
   low-speed turn, 12 unprotected cross) since the cascade respects priority
   strictly; modest improvement elsewhere.
2. **Then**: Phase 1a + 1b + 2 — wire the 9 more rule encoders. Cascade-per-tick
   makes their effect visible immediately.
3. **Then**: Phase 3 — state-machine rules.
4. **Then**: Phase 5 — behavioural layer.
5. **Skip Phase 4** (offline calibration runner) unless future online deployment
   becomes a real requirement.
6. **Throughout**: Phase 6 diagnostics in parallel — every new rule encoder
   gets its slack contribution traced from day one.

The first item is **3 days** and gives the most signal per hour spent. Do it
first; the result will inform how aggressively to pursue the rest.
