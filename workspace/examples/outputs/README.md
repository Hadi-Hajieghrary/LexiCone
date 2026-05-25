# `examples/outputs/` — generated artefacts from every demo / batch / experiment

This directory is the project's **dump ground for everything the planners produce**: simulation MP4s + GIFs, per-tick rule-evaluation CSVs, episode-summary PNGs, comparative-effectiveness plots, and the IEEE Transactions-grade figure set. Nothing here is hand-authored; every file is written by a script in [examples/](..) or [scripts/](../../scripts/).

## Teaser — the LCP planner traversing an intersection

This is what the operational LCP-WS-$L_1$ planner does when it crosses a full nuPlan-mini intersection (scenario 16 of the batch). The four-row visualisation layout — header band, map with ego (red) + agents + planned trajectory (dashed) + traffic-light markers, sidebar with active-rules panel, and the bottom violation strip — is consistent across all 16 scenarios.

![LCP planner traversing an intersection](12_batch_two_level_mpc_planner/16_traversing_intersection__l1/16_traversing_intersection.gif)

Full visual gallery (all 16 scenarios with embedded GIFs + per-scenario narrative) at [`12_batch_two_level_mpc_planner/README.md`](12_batch_two_level_mpc_planner/README.md).

The directory is **git-ignored** because individual artefacts are large (multi-MB MP4s × 16 scenarios × N conditions). Re-generate by running the scripts noted in each subdirectory README.

## Directory map

```text
examples/outputs/
├── README.md                                       ← you are here
├── 12_batch_two_level_mpc_planner/                 [LCP-WS-L1, first iteration of the 16-scenario batch]
│   ├── README.md                                   ← parent index: viz key + 16-scenario gallery table
│   ├── batch_summary_l1.csv                        ← top-level per-scenario summary
│   ├── batch_summary.csv  (legacy from earlier)
│   ├── batch_summary_extra_17.csv  (custom --types)
│   ├── batch_summary_extra_18.csv  (custom --types)
│   └── <label>__l1/                                ← one subdir per scenario (16 total)
│       ├── README.md                               ← scenario narrative + embedded GIF + observed violations
│       ├── <label>.mp4                             ← per-tick simulation animation (IEEE-styled, 130 dpi)
│       ├── <label>.gif                             ← GitHub-embedded GIF (20 fps, 960 px wide)
│       ├── <label>_log.csv                         ← per-tick per-rule evaluations (~3750 rows)
│       └── <label>_summary.png                     ← episode summary plot
│
├── 13_protocol/                                    [comparative-effectiveness protocol, see plan/README]
│   ├── README.md
│   ├── C0_legacy/seed_<n>/<label>/                 ← legacy flat-weight MPC baseline
│   ├── C1_ws_l1/seed_<n>/<label>__l1/              ← LCP weighted-sum, L₁ penalty
│   ├── C2_ws_l1_slp3/seed_<n>/<label>__l1/         ← C1 with 3 SLP iterations
│   ├── C3_ws_l2/seed_<n>/<label>__l2/              ← LCP weighted-sum, L₂ penalty
│   ├── C4_cascade_l1/seed_<n>/<label>__l1_cascade/ ← full lex cascade per tick
│   └── figures/                                    ← F1–F8 IEEE figures + per_cell_metrics.csv
│
└── artifacts/                                      [277 plots/figures supporting the LCP paper]
    ├── README.md
    ├── per_scenario/<label>/                       ← P1–P6 per-scenario detail (6 × 16 = 96)
    ├── aggregate/                                  ← A1–A10 cross-scenario plots (10)
    ├── theory/                                     ← T1, C2–C4, B1/B2/B4, D1/D2, G1 (12)
    └── violations/                                 ← peak-violation snapshots per rule (159)
```

## What writes what

| Directory | Producer script | Trigger |
|---|---|---|
| `12_batch_two_level_mpc_planner/` | [examples/12_batch_two_level_mpc_planner.py](../12_batch_two_level_mpc_planner.py) | `python examples/12_batch_two_level_mpc_planner.py --penalty-form l1` |
| `13_protocol/<cond>/seed_<n>/` | [examples/13_run_protocol.py](../13_run_protocol.py) | `python examples/13_run_protocol.py --conditions … --seeds …` |
| `13_protocol/figures/` | [examples/analyze_protocol.py](../analyze_protocol.py) | `python examples/analyze_protocol.py --protocol-root examples/outputs/13_protocol` |
| `artifacts/per_scenario/`, `artifacts/aggregate/`, `artifacts/theory/` | [scripts/generate_artifacts.py](../../scripts/generate_artifacts.py) | `python scripts/generate_artifacts.py` |
| `artifacts/violations/` | [scripts/generate_violation_snapshots.py](../../scripts/generate_violation_snapshots.py) | `python scripts/generate_violation_snapshots.py` |

## How to navigate this tree

If you want to **see what the LCP planner does on one scenario**, open the per-scenario `README.md` under `12_batch_two_level_mpc_planner/<label>__l1/README.md` — it embeds the GIF, narrates what's happening, and explains which rules fired and why. The 16 readmes are linked from a gallery table in [`12_batch_two_level_mpc_planner/README.md`](12_batch_two_level_mpc_planner/README.md). The MP4, summary PNG, and per-tick CSV are companions to the GIF in the same directory.

If you want to **see the evidence that LCP is better than legacy**, open `13_protocol/figures/F1_per_level_per_scenario.png` after the comparative batch completes (`F3_lex_dominance_heatmap.png` is the headline structural claim).

If you want to **understand the framework or the rules**, browse `artifacts/theory/` (paper-section figures) and `artifacts/violations/<label>__gallery.png` (which rules each scenario violated and exactly when).

Each subdirectory has its own README with the same level of detail.
