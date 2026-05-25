# workspace — lexicographic rule observer + two-level motion planner for nuPlan

This directory is the project's own code (everything outside [`workspace/`](.) is either upstream tooling or data). It contains two complementary halves:

1. **[`lexicone/observer/`](lexicone/observer/)** — a per-tick rule engine that watches a driving scene and reports, rule by rule, whether the ego is compliant. Twenty-five rules, organised lexicographically (priority levels 0 … 10), implement a NuPlan-applicable subset of a real-world driving rulebook (safety-critical at the top, comfort at the bottom).
2. **[`lexicone/planning/`](lexicone/planning/)** — a two-level motion planner that drives the ego inside the nuPlan closed-loop simulator. A periodic *global route planner* walks the lane graph from the ego's current pose toward the scenario goal; a per-tick *trajectory planner* solves a nonlinear MPC (CasADi/IPOPT) over a kinematic-bicycle model to track that route subject to control, rate and obstacle constraints.

A third half — **[`examples/`](examples/)** — ties the two together. Demos 01–10 are self-contained synthetic-world rule-observer demos. Demo 06 renders saved nuPlan logs through the observer. Demos 11 and 12 run the planner inside the official nuPlan simulator and then visualise the result with the observer's renderer.

## Directory layout

```text
workspace/
├── README.md                                ← you are here
├── mp4_to_gif_recursive.sh                  (utility — see §Utility scripts below)
├── lexicone/                                ← public package
│   ├── README.md
│   ├── __init__.py
│   ├── observer/                            ← rule engine + adapters (25-rule rulebook)
│   │   ├── README.md
│   │   ├── __init__.py                      (public re-exports)
│   │   ├── rule.py                          (ObserverRule ABC)
│   │   ├── engine.py                        (RuleEngine: step / run_replay / summary)
│   │   ├── types.py                         (SceneSnapshot, RuleEvaluation, RuleSummary, EpisodeSummary, …)
│   │   ├── context.py                       (SceneContext + cached_property derivations)
│   │   ├── geometry.py                      (Shapely-based footprint / projection / lane-finding helpers)
│   │   ├── registry.py                      (build_default_rules → 25 rule instances)
│   │   ├── nuplan_adapter.py                (NuPlanSceneSource: live scenario → SceneSnapshot)
│   │   ├── simulation_log_adapter.py        (NuPlanSimulationLogSource: saved log → SceneSnapshot)
│   │   ├── rules/                           ← one module per rule (README + 25 *.py)
│   │   └── tests/                           ← engine + per-rule unit tests
│   └── planning/                            ← two-level MPC planner (legacy + LCP modes)
│       ├── README.md                        (see for full module map; 14 modules summarised below)
│       ├── __init__.py
│       ├── bicycle_model.py                 (CasADi symbolic kinematic bicycle, RK4)
│       ├── reference_path.py                (ReferencePath: arc-length-parameterised polyline)
│       ├── global_planner.py                (GlobalRoutePlanner: periodic BFS over lane graph)
│       ├── trajectory_planner.py            (MPCTrajectoryPlanner: legacy single-tier MPC)
│       ├── two_level_planner.py             (TwoLevelMPCPlanner: AbstractPlanner subclass; orchestrator)
│       ├── lcp_mpc.py                       (LCPTrajectoryPlanner: convex linearised LCP MPC)
│       ├── lex_cascade.py                   (run_cascade: L+1-stage lex cascade per tick)
│       ├── weight_calibration.py            (Algorithm 1A + 1B for w† computation)
│       ├── slp_linearisation.py             (BicycleLinearisation + SLP convergence metric)
│       ├── rule_encoder.py                  (16 active rule encoders + 3 stubs)
│       ├── compliance_checker.py            (runtime b_ε(z_ws) vs b_ε(z_lex*) comparison)
│       ├── calibration_cache.py             (JSON-backed (scenario_class, penalty_form, ε) → w† cache)
│       ├── map_lifter.py                    (per-tick lifting of map data into ego-local frame)
│       ├── config/planner/two_level_mpc_planner.yaml   (Hydra config)
│       ├── docs/full_rule_wiring_plan.md    (design doc for the 16-rule encoder phase)
│       └── tests/                           ← 12 test files / 98 tests
├── examples/                                ← runnable demos + analysis pipeline
│   ├── README.md
│   ├── simulation.py                        (synthetic closed-loop harness, 10 Hz)
│   ├── planners.py                          (6 pedagogical planners: constant / IDM / aggressive / overtake / urban / lane-change)
│   ├── scenarios.py                         (canned synthetic worlds)
│   ├── visualizer.py                        (IEEE-styled MP4 + summary PNG + per-tick CSV renderer)
│   ├── 01_…py … 13_run_protocol.py          (13 numbered demos)
│   ├── analyze_protocol.py                  (per-level decomposition + lex-Pareto dominance + F1–F3 figures)
│   ├── metrics_smoothness.py                (msgpack-replay smoothness extractor for F7)
│   └── outputs/                             (generated artefacts; see outputs/README.md)
├── scripts/                                 ← post-processing pipeline
│   ├── README.md
│   ├── ieee_style.py                        (single source of truth for IEEE typography)
│   ├── generate_artifacts.py                (118 per-scenario + aggregate + theory figures)
│   └── generate_violation_snapshots.py      (159 peak-violation snapshots + per-scenario galleries)
└── tests/                                   (top-level placeholder; per-package tests live under lexicone/)
```

## How the two halves fit together

There are two ways the rule observer and the planner can be combined:

```text
                  Synthetic demos (01–10)
                  ┌──────────────────────────┐
World + Planner → │ examples/simulation.py   │ → list[SceneSnapshot] ──┐
                  └──────────────────────────┘                          │
                                                                        ▼
                                                              examples/visualizer.py
                                                                        ▲
                                                                        │
                  nuPlan simulator                                       │
                  ┌──────────────────────────┐                          │
TwoLevelMPCPlanner│ run_simulation.py (Hydra)│ → SimulationLog.pkl/    │
                  └──────────────────────────┘   msgpack.xz ───────────▶ NuPlanSimulationLogSource
                                                                        │
                                                                        ▼
                                                              list[SceneSnapshot]
                                                                        │
                                                                        ▼
                                                              examples/visualizer.py
```

Both paths produce the **same** `list[SceneSnapshot]` data structure, so the visualiser does not need to know whether the snapshots came from a synthetic world or from a recorded nuPlan log.

## Quick start

### Run a synthetic demo

```bash
cd workspace
python examples/04_simulated_idm_following.py
# Writes MP4/PNG/CSV to workspace/examples/outputs/04_simulated_idm_following/
```

### Run the MPC planner inside the nuPlan simulator

```bash
cd workspace
python examples/11_simulated_two_level_mpc_planner.py --seed 11
# Picks a random dynamic scenario, runs nuPlan's run_simulation.py via
# subprocess with our planner, then renders the resulting SimulationLog.
```

### Run the planner over 16 curated scenario types (batch)

```bash
cd workspace
python examples/12_batch_two_level_mpc_planner.py --seed 7
# Or with a custom list:
python examples/12_batch_two_level_mpc_planner.py \
    --types stopping_with_lead,traversing_crosswalk \
    --label-offset 17 --seed 23
```

### Run the planner's unit tests

```bash
PYTHONPATH=/workspace/nuplan-project/workspace \
    pytest workspace/lexicone/planning/tests/
```

## Utility scripts

[`mp4_to_gif_recursive.sh`](mp4_to_gif_recursive.sh) — bash helper that recursively converts every MP4 under a directory to a high-quality GIF (using a two-pass `ffmpeg` filter graph: palette generation + paletteuse). Useful when sharing per-scenario simulation playback for slides / web / GitHub READMEs where MP4 embedding is awkward.

```bash
# default: fps=20, width=960px, skip files whose .gif already exists
./mp4_to_gif_recursive.sh examples/outputs/12_batch_two_level_mpc_planner

# override defaults
FPS=15 WIDTH=720 OVERWRITE=1 ./mp4_to_gif_recursive.sh examples/outputs/13_protocol/C4_cascade_l1
```

Requires `ffmpeg` on PATH.

## Dependencies

All third-party packages used by `workspace/` are already installed by `nuplan-devkit`:

- `casadi 3.7.2` (nonlinear MPC, IPOPT)
- `shapely`, `numpy`, `scipy` (geometry, interpolation, bootstrap CIs)
- `matplotlib`, `ffmpeg` (rendering + MP4 / GIF conversion)
- `hydra-core` (config selection for nuPlan's simulator)

There are no new third-party dependencies introduced by this workspace.

## Where to read next

| If you want to … | Start at … |
|---|---|
| Understand the rule engine | [`lexicone/observer/README.md`](lexicone/observer/README.md) |
| Understand the 25 rules in detail | [`lexicone/observer/rules/README.md`](lexicone/observer/rules/README.md) |
| Understand the MPC planner | [`lexicone/planning/README.md`](lexicone/planning/README.md) |
| See how the demos are wired up | [`examples/README.md`](examples/README.md) |
