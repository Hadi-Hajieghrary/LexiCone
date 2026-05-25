# `13_protocol/` — comparative-effectiveness protocol output tree

This is the **rigorous evidence base** for the paper's "LCP MPC is better than the legacy MPC" claim. The protocol turns the formal LCP framework's lex-priority promise into a falsifiable, multi-seed, multi-condition statistical comparison on the 16-scenario nuPlan-mini benchmark. The complete protocol specification — motivation, definition of "better", experimental conditions, instrumentation, metric suite, statistical analysis, visualisation deliverables, execution sequence, verification checklist — is inlined below.

### Why this protocol exists

The LCP framework promises **lexicographic priority** — higher-priority rules (safety) satisfied before lower-priority ones (comfort) — implemented via per-level epigraph slacks in a convex MPC at [`workspace/lexicone/planning/lcp_mpc.py`](../../../lexicone/planning/lcp_mpc.py). Three operational modes exist: legacy single-tier MPC (baseline), LCP-WS (operational, single solve at calibrated weights), and LCP-Cascade (formally lex-optimal at the linearised problem, ~6× slower). The framework's original empirical claim was anecdotal — one run each of legacy + LCP-WS-$L_1$ showing $-9.8\,\%$ aggregate top-rule violation, plus a single-scenario cascade smoke test — with no statistical testing, no per-priority-level decomposition, no cascade-vs-WS faithfulness measurement, no negative-result reporting. This protocol elevates that anecdote to a defensible "the method is better" demonstration suitable for an IEEE Transactions paper.

Critically, "better" in the LCP context is a *structured* claim, not a scalar one: the method must show lexicographic Pareto dominance — improvement at some priority level without regression at any higher level — across the benchmark.

The framework's own §14.9 reports the prior aggregate-violation observation in its own words: *"With the four-level epigraph-lifted LCP MPC ($L_1$ penalty form, Algorithm 1A-calibrated WS, single SLP iteration per tick) replacing a legacy single-tier MPC with a flat weighted cost, the aggregate integrated violation of the most-violated rule across the benchmark decreased by approximately $10\,\%$. Per-scenario reductions concentrated where the newly-encoded rules apply: starting left turn ($-55\,\%$), traversing intersection ($-55\,\%$), multi-vehicle near-overtake ($-11\,\%$), low-speed turn ($-4\,\%$). Scenarios dominated by observer-only rules (route adherence, yield priority, lane-marker speed limit) showed essentially no change, consistent with the inability of the per-step convex template to encode their semantics. These numbers are reported for orientation rather than as a formal evaluation. The intended takeaway is that the framework's structural contribution — organising constraints into priority levels with per-level epigraph slacks — produces measurable improvements on the scenarios its constraint template can express, and is necessarily silent on the rest."* This protocol exists to convert that observation into a statistically supported comparative claim.

## What it contains

Five planner variants — call them **conditions** — each run over the same 16 nuPlan-mini scenarios with **5 distinct seeds** (`7, 17, 27, 37, 47`), for a total of **5 × 16 × 5 = 400 simulation runs**. Within a run, the per-tick rule evaluations are captured as a CSV; cross-condition analysis is then computed by [`examples/analyze_protocol.py`](../../analyze_protocol.py).

## Conditions

| Code | Method | Hydra flags through `12_batch_two_level_mpc_planner.py` | Per-cell wall (16 scenarios) | Role |
|---|---|---|---|---|
| **C0_legacy** | Legacy flat-weight MPC (no LCP) | *(none)* | ~38 min | Baseline |
| **C1_ws_l1** | LCP with L₁ penalty, single weighted-sum solve | `--penalty-form l1 --runtime-mode ws --slp-max-iterations 1` | ~57 min | Operational LCP |
| **C2_ws_l1_slp3** | C1 with 3 SLP outer iterations | `--penalty-form l1 --runtime-mode ws --slp-max-iterations 3` | ~150 min | WS convergence sensitivity |
| **C3_ws_l2** | LCP with L₂ penalty | `--penalty-form l2 --runtime-mode ws --slp-max-iterations 1` | ~57 min | Penalty-form sensitivity |
| **C4_cascade_l1** | Full L+1-stage lex cascade per tick | `--penalty-form l1 --runtime-mode cascade --slp-max-iterations 1` | ~4 hr | Formally lex-optimal upper bound |

The **decisive comparisons** are:

- **C0 vs C4** — does priority structure help at all? (the headline)
- **C1 vs C4** — is the WS shortcut faithful to the cascade? (paper §14.7 claim)
- **C1 vs C2** — does additional SLP iteration matter? (paper §14.2 claim)

## Directory layout

```text
13_protocol/
├── README.md                                       ← you are here
├── <condition>/                                    ← one subdir per condition (5 total)
│   └── seed_<n>/                                   ← one subdir per seed (5 per condition)
│       ├── run.log                                 ← stdout/stderr of 12_batch_… for this cell
│       ├── batch_summary[<suffix>].csv             ← per-scenario summary for the cell
│       └── <label>[__l1|__l2|__l1_cascade]/        ← per-scenario artefacts
│           ├── <label>.mp4
│           ├── <label>_log.csv
│           └── <label>_summary.png
└── figures/                                        ← outputs of analyze_protocol.py
    ├── per_cell_metrics.csv                        ← long-form (cond, scen, seed, level, V) table
    ├── lex_dominance.csv                           ← pairwise (cond_M, baseline, n_wins, n_ties, n_losses)
    ├── F1_per_level_per_scenario.png               ← stacked bars: per-level violation, one panel per condition
    ├── F2_delta_violin.png                         ← per-level percent reduction box-plots, with strip
    └── F3_lex_dominance_heatmap.png                ← scenario × condition lex-Pareto winner matrix
```

The `<suffix>` on `batch_summary*.csv` encodes the LCP mode:

- C0: `batch_summary.csv`
- C1, C2, C4 (l1): `batch_summary_l1.csv` (cascade variant adds `_cascade`)
- C3 (l2): `batch_summary_l2.csv`

## Producer pipeline

```
                                  ┌─────────────────────┐
                                  │ 13_run_protocol.py  │
                                  │ (multi-seed driver, │
                                  │  4-way parallel)    │
                                  └──────────┬──────────┘
                                             │ for each (cond, seed):
                                             ▼
                          ┌─────────────────────────────────┐
                          │ 12_batch_two_level_mpc_planner  │ ← unique NUPLAN_EXP_ROOT
                          │ (Hydra → run_simulation.py →    │   per (cond, seed) so
                          │  render_episode → CSV)          │   parallel cells never
                          └──────────┬──────────────────────┘   collide
                                     │                          ↓
                                     │                        post-cell cleanup
                                     ▼                        deletes the EXP_ROOT
                       per-tick CSVs, MP4s, summary PNGs
                       under <cond>/<seed>/<label>/
                                     │
                                     ▼
                          ┌─────────────────────┐
                          │ analyze_protocol.py │
                          │ (per-level metrics, │
                          │  lex-dominance,     │
                          │  bootstrap CI,      │
                          │  IEEE figures)      │
                          └──────────┬──────────┘
                                     │
                                     ▼
                                figures/
```

## Re-generating the whole tree

End-to-end (~14 wall-hours on a 16-CPU machine with `--parallel 4`):

```bash
cd workspace
# Phase 2: comparative batch
python examples/13_run_protocol.py \
    --conditions C0,C1,C2,C3,C4 \
    --seeds 7,17,27,37,47 \
    --parallel 4 --resume

# Phase 4: analysis + IEEE figures
python examples/analyze_protocol.py \
    --protocol-root examples/outputs/13_protocol \
    --baseline C0_legacy
```

The driver is **resumable**: `--resume` skips any cell whose `batch_summary*.csv` already has ≥ 16 data rows. Useful for restarting after machine reboots, OOM kills, or hand-aborted runs.

## Disk-space management

The driver writes intermediate Hydra/nuPlan output to a **cell-unique `NUPLAN_EXP_ROOT`** (`/workspace/exp__<cond>__seed_<n>/`) and deletes it after the cell completes. Final artefacts (MP4 + per-tick CSV + summary PNG) are preserved under this directory; nothing under the cell EXP_ROOT is needed downstream. Peak disk per cell ≈ 0.5–2 GB; with 4-way parallel ≈ 8 GB worst-case peak.

## Definition of "better"

Let $V_{\mathrm{MPC},\ell}(M, S)$ be the integrated rule violation at priority level $\ell$ for method $M$ on scenario $S$, restricted to the **16 LCP-encoded rules** (the framework-controlled subset; the 9 observer-only state-machine rules are reported separately as invariant controls since no MPC method can affect them). The 16 controlled rules span $\ell \in \{10, 9, 7, 3, 1, 0\}$.

Three nested definitions, evaluated only on the controlled vector:

**(a) Lex-Pareto dominance (primary, structural)** — $M \succ_{\mathrm{lex}} B$ on $S$ iff

$$\exists\,\ell^\star \;\text{with}\; V_{\ell^\star}(M, S) < V_{\ell^\star}(B, S) \;\text{AND}\; V_\ell(M, S) \le V_\ell(B, S) + \tau_\ell \;\forall\,\ell > \ell^\star,$$

where $\tau_\ell = \epsilon_\ell \cdot T_S$ is a per-level tolerance derived from the LCP $\epsilon$ vector and the scenario duration. Headline metric: **lex-win rate** across the 16 scenarios.

**(b) Priority-weighted scalar (secondary, ranking)** — $J(M,S) = \sum_\ell 10^\ell \cdot V_\ell(M,S)$. Decade-separated weights ensure any higher-level violation outranks any finite lower-level total. Single-number ranking summary; never used for tuning.

**(c) Per-level percent reduction (secondary, diagnostic)** — $\Delta_\ell(M, B, S) = (V_\ell(B,S) - V_\ell(M,S)) / \max(V_\ell(B,S), \tau_\ell)$, aggregated as level-wise median across scenarios with bootstrap CI.

Headline claim format: *"LCP-Cascade lex-dominates Legacy on $k/16$ scenarios; on the remainder it ties at the top differing level and is no worse on any higher level. Median per-level reductions are $\Delta_{10}=\ldots, \Delta_9=\ldots$ with $95\,\%$ bootstrap CIs."*

Seeds: `{7, 17, 27, 37, 47}`. The existing `--seed` flag advances by 1 per scenario inside one batch (seed=7 → seeds 8..23 across the 16 scenarios). Five distinct base seeds → 5 distinct scenario-instance draws per `(scenario_type, condition)`.

## Adjustments to the optimisation

1. **Calibration pre-seed**. Before any LCP run, populate `${NUPLAN_EXP_ROOT}/lcp_cache.json` by running Algorithm 1A for each of the 16 scenario types. Cache key: `(scenario_class, penalty_form, epsilon_per_level)`. The existing resolver in [`workspace/lexicone/planning/calibration_cache.py`](../../../lexicone/planning/calibration_cache.py) reads this transparently; no batch-script changes needed once populated. ≈ 30 min total (≈ 2 min × 16 classes).

2. **Multi-seed driver**. Outer script [`examples/13_run_protocol.py`](../../13_run_protocol.py) loops `(condition, seed)`, invokes [`examples/12_batch_two_level_mpc_planner.py`](../../12_batch_two_level_mpc_planner.py) per cell, writes to `examples/outputs/13_protocol/<condition>/seed_<n>/`. Uses `multiprocessing.Pool(4)` to run 4 seeds in parallel (each subprocess gets a distinct `NUPLAN_EXP_ROOT` to avoid cache collisions).

3. **Per-tick solve-time logging** (new instrumentation). Optional `--solve-log <path>` flag through `12_batch_two_level_mpc_planner.py` → `two_level_planner.py` → `lcp_mpc.py` + `lex_cascade.py`. Each tick appends `(t_s, stage_index, solve_time_s, ipopt_status)` to `<label>_solve.csv`.

4. **Per-tick compliance-vector logging** (new instrumentation). [`compliance_checker.py`](../../../lexicone/planning/compliance_checker.py) `attach_csv_sink` wired into [`two_level_planner.py`](../../../lexicone/planning/two_level_planner.py). For C1/C2/C3 emit `<label>_compliance.csv` with the per-tick binary compliance vector. C4 emits the cascade's $b_\epsilon(z_\mathrm{lex}^\star)$ as reference. Post-process computes match rate per level.

5. **Smoothness extraction** (post-processing). A separate script [`examples/metrics_smoothness.py`](../../metrics_smoothness.py) replays each simulation log via existing `NuPlanSimulationLogSource` and extracts `(lateral_jerk_rms, peak_a_y, longitudinal_jerk_rms)`. No simulator re-run; just log replay (≈ 5 s per scenario).

## Metric suite

**Primary (lex structure) — comes for free from existing per-tick CSVs:**

- $V_{\mathrm{MPC},\ell}(M, S)$ for $\ell \in \{10, 9, 7, 3, 1, 0\}$, computed by grouping per-tick `violation_rate * dt` by the leading priority digit of `rule_id`
- Lex-dominance flag $\mathbb{1}[M \succ_{\mathrm{lex}} B]$ at tolerance $\tau$
- Priority-weighted aggregate $J(M, S)$

**Secondary (operational) — new instrumentation or post-processing:**

- Per-tick wall time (new CSV)
- Lateral-jerk RMS, peak $|a_y|$, longitudinal-jerk RMS (post-process from simulation log)
- Goal progression: longitudinal distance / scenario duration (post-process)

**Diagnostic (framework-specific) — new instrumentation:**

- WS-vs-cascade compliance match rate per level
- Active-set evolution per level (fraction of ticks with binding slack) — nice-to-have

## Statistical analysis

- Within each `(condition, scenario_type)`, aggregate the 5 seeds by **median** (robust to outlier scenario draws).
- **Headline lex-win rate** — paired sign test on the 16 per-scenario lex-dominance flags (majority-of-5 indicator per scenario), exact binomial $95\,\%$ CI via `scipy.stats.binomtest`.
- **Per-level $\Delta_\ell$** — stratified BCa bootstrap over the 16 scenarios, $10\,000$ resamples, BCa $95\,\%$ CI.
- **Wall-time / smoothness** — Wilcoxon signed-rank on per-scenario medians; Holm–Bonferroni across the 4 pairwise $(C_0, C_x)$ comparisons.
- **Negative-result reporting** — tabulate every `(scenario, level)` where LCP is worse, with effect size and visual inspection; do not suppress.

## Visualisation deliverables (IEEE Transactions format)

All figures reuse [`workspace/scripts/ieee_style.py`](../../../scripts/ieee_style.py) (Times serif, $10\,\mathrm{pt}$ body, $300\,\mathrm{dpi}$, $3.50''$ single-column / $7.16''$ double-column widths).

| Fig | Layout | Source | Paper placement |
|---|---|---|---|
| F1 | 4-panel stacked-bar across 16 scenarios, one panel per condition; bars stacked by level on log-y | Per-tick CSV | Results §A — headline |
| F2 | Per-level violin/box of $\Delta_\ell(\mathrm{C4, C0})$ across 16 scenarios with BCa CIs | Bootstrap | Results §A — backs F1 |
| F3 | Lex-dominance heatmap, rows = scenarios, cols = pairwise comparisons; cell = winner | Lex flags | Results §B — structural win |
| F4 | Wall-time vs lex-win-rate Pareto scatter, one dot per condition; seed-spread error bars | Wall-time + lex-win | Results §B — speed/quality |
| F5 | WS-vs-cascade compliance match rate per level (C1 vs C4 ref), 16-scenario small-multiples | New `*_compliance.csv` | Discussion §A — faithfulness |
| F6 | SLP-iteration sensitivity (C1 vs C2): per-level violation reduction vs iters | Per-tick CSV | Discussion §B — convergence |
| F7 | Smoothness Pareto: lateral-jerk RMS vs lex-aggregate, one dot per (scenario, condition) | Smoothness post-process | Discussion §C — no comfort regression |
| F8 | Single-scenario qualitative panel (e.g. `03_near_multiple_vehicles`): trajectories overlaid + per-level violation timeline + speed profile, all conditions colour-coded | Per-tick CSV + simulation log | Discussion §D — qualitative narrative |

## Files added / modified

**New scripts:**

- [`workspace/examples/13_run_protocol.py`](../../13_run_protocol.py) — outer driver: loops `(condition, seed)`, parallel via `multiprocessing.Pool(4)`, writes to `examples/outputs/13_protocol/<condition>/seed_<n>/`
- [`workspace/examples/calibrate_lcp_offline.py`](../../calibrate_lcp_offline.py) — runs Algorithm 1A per scenario type via existing [`workspace/lexicone/planning/weight_calibration.py`](../../../lexicone/planning/weight_calibration.py), writes to `${NUPLAN_EXP_ROOT}/lcp_cache.json`
- [`workspace/examples/metrics_smoothness.py`](../../metrics_smoothness.py) — log-replay smoothness extraction
- [`workspace/examples/analyze_protocol.py`](../../analyze_protocol.py) — reads all per-tick CSVs, applies rule→level map from [`workspace/lexicone/observer/registry.py`](../../../lexicone/observer/registry.py), computes lex-dominance + bootstrap, emits F1–F8 to `workspace/examples/outputs/13_protocol/figures/`
- [`workspace/lexicone/planning/tests/test_rule_level_mapping.py`](../../../lexicone/planning/tests/test_rule_level_mapping.py) — pytest asserting the 16 controlled + 9 observer-only IDs partition `DEFAULT_RULE_IDS`

**Modifications (instrumentation-only — no framework changes):**

- [`workspace/lexicone/planning/two_level_planner.py`](../../../lexicone/planning/two_level_planner.py) — accept `solve_log_path` and `compliance_log_path` ctor args; wire `ComplianceChecker.attach_csv_sink`
- [`workspace/lexicone/planning/lcp_mpc.py`](../../../lexicone/planning/lcp_mpc.py) — emit per-stage solve-time rows when `solve_log_path` is set
- [`workspace/lexicone/planning/lex_cascade.py`](../../../lexicone/planning/lex_cascade.py) — same
- [`workspace/examples/12_batch_two_level_mpc_planner.py`](../../12_batch_two_level_mpc_planner.py) — pass-through `--solve-log` and `--compliance-log` CLI flags; suffix summary-CSV filename with seed

## Execution sequence

```
Phase 0 — instrumentation + tests (1 hr coding)
   1. Add pytest at tests/test_rule_level_mapping.py
   2. Add solve-time + compliance logging to lcp_mpc.py, lex_cascade.py, two_level_planner.py
   3. Add --solve-log, --compliance-log pass-through to 12_batch_two_level_mpc_planner.py

Phase 1 — calibration pre-seed (~30 min wall)
   4. Implement examples/calibrate_lcp_offline.py
   5. Run it: produces ${NUPLAN_EXP_ROOT}/lcp_cache.json with 16 entries
   6. Verify entries via grep

Phase 2 — comparative batch (~14 hr wall on 4-CPU)
   7. Implement examples/13_run_protocol.py (parallel driver)
   8. Run it: 5 conditions × 16 scenarios × 5 seeds = 400 runs
   9. Periodic checkpoint: after C0 + C1 + C4 seed-1 (~6 hr), inspect headline numbers

Phase 3 — smoothness post-process (~10 min wall)
  10. Implement + run examples/metrics_smoothness.py over all 400 simulation logs

Phase 4 — analysis + figures (~30 min wall)
  11. Implement examples/analyze_protocol.py
  12. Run it: per-level metrics → bootstrap → F1–F8 PNGs (IEEE 7.16"×N, 300 dpi)
  13. Cross-check headline: lex-win-rate(C4 vs C0) ≥ 50 % with binomial p < 0.05

Phase 5 — paper integration
  14. Update the LCP paper §14.9 with the rigorous numbers
  15. Reference F1–F8 in §14.9 / §14.10
```

## Verification

Before publishing any number from the protocol:

1. **Rule-mapping smoke test** — `pytest workspace/lexicone/planning/tests/test_rule_level_mapping.py`: assert the 16 controlled IDs from `make_default_ruleset()` plus the 9 observer-only IDs partition `DEFAULT_RULE_IDS` from `registry.py`. A drift here invalidates every $V_\ell$.

2. **Single-scenario sanity replay** — re-run `01_following_slow_lead` under C0 + C4 with seed=7:
   - C4 wall time ≈ 16 min (matches earlier smoke result)
   - C4 $V_{10}$ and $V_9$ = 0 (collision rules must never fire in a benign overtake)
   - C1 vs C4 per-tick compliance match ≥ $95\,\%$ at $\ell = 10$

3. **Calibration cache check** — after `calibrate_lcp_offline.py`, grep `lcp_cache.json` for 16 non-default entries (no `1.0, 1.0, 1.0` heuristic fallbacks).

4. **Statistical sanity** — the lex-win rate must be monotone in theoretical strength: $\text{C4} \ge \text{C2} \ge \text{C1} \ge \text{C0}$. Violation = pipeline bug, not a finding.

5. **Negative-control plot** — per-condition histogram of *observer-only* rule violations must be statistically indistinguishable across C0…C4 (since MPC cannot affect them). If they differ, scenario-instance draws are drifting and the comparison is contaminated.

6. **Reproducibility check** — re-run one `(condition, scenario, seed)` cell from scratch; per-tick CSV must be bit-identical (the planner is deterministic given seed).

## Innovation angle (optional follow-up, not in this protocol)

If the headline numbers warrant a stronger claim, the next step is **adversarial scenario construction**: build 4–6 synthetic micro-scenarios where ANY non-priority-ordered planner must fail (e.g., a forced choice between hitting a pedestrian and being rear-ended; a red-light + lead-vehicle stop conflict). These would maximally stress the priority ordering and produce dramatic contrast in trajectories — but they require new scene-builder code in [`workspace/examples/scenarios.py`](../../scenarios.py) and are out of scope here.

## Status of this directory

While the comprehensive batch is running, this directory grows incrementally. Each cell's subdir materialises only after its `12_batch_…py` subprocess finishes. Check progress with:

```bash
ls examples/outputs/13_protocol/*/seed_*/batch_summary*.csv | wc -l   # cells done
```

(should be 25 when the full sweep finishes; analysis pipeline can run partially with whatever's done so far).
