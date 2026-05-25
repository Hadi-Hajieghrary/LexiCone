# `13_protocol/` — comparative-effectiveness protocol output tree

This is the **rigorous evidence base** for the paper's "LCP MPC is better than the legacy MPC" claim. The protocol is described in full at [`/home/vscode/.claude/plans/i-would-like-to-fuzzy-flamingo.md`](../../../../home/vscode/.claude/plans/i-would-like-to-fuzzy-flamingo.md) and summarised at §14.9 of [References/lex_constraint_programming_report_v10_2.md](../../../../References/lex_constraint_programming_report_v10_2.md).

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

## Status of this directory

While the comprehensive batch is running, this directory grows incrementally. Each cell's subdir materialises only after its `12_batch_…py` subprocess finishes. Check progress with:

```bash
ls examples/outputs/13_protocol/*/seed_*/batch_summary*.csv | wc -l   # cells done
```

(should be 25 when the full sweep finishes; analysis pipeline can run partially with whatever's done so far).
