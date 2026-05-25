# `scripts/` — figure generators and shared style

This directory contains the **post-processing pipeline** that turns simulation outputs into paper-grade figures. None of these scripts run the simulator themselves — they consume the per-tick CSVs and msgpack logs produced by [`examples/`](../examples/).

All figures honour IEEE Transactions publication style (Times serif, 10 pt body, 300 dpi, 3.50 in / 7.16 in standard column widths).

## Files

| File | Role |
|---|---|
| [`ieee_style.py`](ieee_style.py) | Single source of truth for typography + line weights + DPI + column widths. `import ieee_style; ieee_style.apply()` configures `matplotlib.rcParams` once at script start. Exposes `COL_1 = 3.50`, `COL_2 = 7.16` for figure sizing. |
| [`generate_artifacts.py`](generate_artifacts.py) | Produces the **118-PNG static artefact set** under [`../examples/outputs/artifacts/`](../examples/outputs/artifacts/): 96 per-scenario detail figures (P1–P6 × 16 scenarios), 10 cross-scenario aggregates (A1–A10), 12 conceptual figures for the paper (T1, T2, T3, C2, C3, C4, B1, B2, B4, D1, D2, G1). Reads from [`../examples/outputs/12_batch_two_level_mpc_planner/`](../examples/outputs/12_batch_two_level_mpc_planner/). |
| [`generate_violation_snapshots.py`](generate_violation_snapshots.py) | Produces the **159-PNG violation set** under [`../examples/outputs/artifacts/violations/`](../examples/outputs/artifacts/violations/): 143 per-rule peak-violation snapshots + 16 per-scenario galleries. Reads msgpack logs directly from `${NUPLAN_EXP_ROOT}/exp/demo_12_batch__<label>__l1/...`, renders clean frames from scratch (no MP4 extraction). |

## Re-generating everything

```bash
cd workspace
python scripts/generate_artifacts.py             # ~3 min
python scripts/generate_violation_snapshots.py   # ~3 min
```

Both scripts are idempotent — running them again overwrites existing outputs. They depend only on simulation outputs already on disk; if those have been deleted, re-run the relevant batch first ([examples/12_batch_two_level_mpc_planner.py](../examples/12_batch_two_level_mpc_planner.py)).

## Style customisation

To change typography globally (e.g. switch from 10 pt to 11 pt body), edit only [`ieee_style.py`](ieee_style.py); the rest of the pipeline picks it up via `import ieee_style; ieee_style.apply()` calls.

To change figure sizes per artefact, edit the `figsize=(W, H)` arguments inside the relevant `fig_*` function in `generate_artifacts.py`. The width should be `COL_1` (3.50") for single-column or `COL_2` (7.16") for double-column.

## Related downstream pipeline

The comparative-effectiveness analysis ([`../examples/analyze_protocol.py`](../examples/analyze_protocol.py)) and the smoothness extractor ([`../examples/metrics_smoothness.py`](../examples/metrics_smoothness.py)) both depend on `ieee_style.apply()`, but they live under `examples/` because they consume data from `examples/outputs/13_protocol/` and produce figures under the same tree.

## Related utility scripts

[`../mp4_to_gif_recursive.sh`](../mp4_to_gif_recursive.sh) at the workspace root is a separate bash helper that recursively converts every MP4 under a directory to a high-quality GIF (two-pass `ffmpeg` filter graph: palette generation + paletteuse, configurable FPS + width). Not part of the figure pipeline, but useful for sharing scenario-playback MP4s where GIF embedding is preferred.
