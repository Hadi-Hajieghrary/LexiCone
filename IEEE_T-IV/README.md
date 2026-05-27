# IEEE T-IV manuscript: Closed-Loop Deployment of LCP-MPC

This directory contains the IEEE Transactions on Intelligent Vehicles submission *Closed-Loop Deployment of Lexicographic Constraint Programming for Trajectory Model Predictive Control: A Comparative-Effectiveness Study on the nuPlan Simulator*.

## Layout

- [Main.tex](Main.tex) — the manuscript root. IEEEtran `[journal, twocolumn]`. `\subfile{}` wires §I–§IX from [Sections/](Sections/).
- [Sections/](Sections/) — 9 body section subfiles + 6 appendix subfiles. Each is a `\subfile{}`-compatible fragment (no `\documentclass` wrapper).
- [References/references.bib](References/references.bib) — 24 IEEEtran-style bibliography entries grouped by literature line (equivalent-weight theory, prioritised control, rulebook formalism, multi-objective optimisation, numerical methods, statistical methodology).
- [Figures/](Figures/) — inline figures; symlinks to source artefacts under [../workspace/examples/outputs/artifacts/](../workspace/examples/outputs/artifacts/). Naming convention `figNN_<id>_<name>.png` (no extension in `\includegraphics{}` so pdflatex auto-detects).
- [IEEEtran.cls](IEEEtran.cls), [IEEEtran.bst](IEEEtran.bst), [ieeeconf.cls](ieeeconf.cls) — IEEEtran class + bibliography style bundled in-repo.
- [Makefile](Makefile) — build chain (`make paper`, `make figures`, `make analysis`, `make protocol`, `make ablations`, `make all`).

## Build chain

```
protocol  →  analysis  →  figures  →  paper
                ↓                       ↑
            supplement-data        supplement
```

- **`make protocol`** — runs the 400-cell comparative protocol via [../workspace/examples/13_run_protocol.py](../workspace/examples/13_run_protocol.py).
- **`make ablations`** — runs the four Phase-0.5 ablation arms + external baseline (see Section IV of the plan; not yet implemented in the Makefile).
- **`make analysis`** — runs `analyze_protocol.py`, `metrics_smoothness.py`, and the new `analyze_*.py` scripts. Emits long-form CSVs at [../workspace/examples/outputs/13_protocol/figures/](../workspace/examples/outputs/13_protocol/figures/).
- **`make figures`** — regenerates every Fig.* and Table.* CSV input.
- **`make paper`** — `pdflatex Main.tex` ×3 + `bibtex` for the cross-references and bibliography.
- **`make supplement`** — builds [supplement.tex](supplement.tex) as a separate PDF.

End-to-end reproduction (`make all`) takes approximately 128 cell-hours of compute plus 5 days of authorial work. See [Sections/Appendix_E_Reproducibility.tex](Sections/Appendix_E_Reproducibility.tex) for the per-target breakdown.

## Per-folder cross-references

This README is intentionally short. The deeper per-folder documentation lives in the planner repository:

- [../workspace/lexicone/planning/README.md](../workspace/lexicone/planning/README.md) — two-level MPC architecture, the 16 rule encoders, the LCP OCP.
- [../workspace/lexicone/observer/README.md](../workspace/lexicone/observer/README.md) — 25-rule observer subsystem.
- [../workspace/examples/](../workspace/examples/) — 13 numbered demos + the protocol driver + analysis scripts.
- [../workspace/scripts/](../workspace/scripts/) — `ieee_style.py` (typography), `generate_artifacts.py` (277 figures), `generate_violation_snapshots.py` (159 PNGs).
- [../References/](../References/) — the foundation paper [v10_2] plus the workshop documents.

## Citation key for the foundation paper

The foundation paper [v10_2] is cited as `\cite{lcp2025}` throughout this manuscript. The bibtex entry in [References/references.bib](References/references.bib) currently uses `@unpublished` with the note "Manuscript in preparation"; update to an `@article` (or `@misc` with arXiv ID) before submission once the foundation paper is posted.

## Pre-submission checklist

1. **Protocol complete.** All 25 cells of [../workspace/examples/outputs/13_protocol/](../workspace/examples/outputs/13_protocol/) have a `batch_summary*.csv`.
2. **Phase 0.5 complete.** Ablation arms ×4, external baseline, calibration-weight sensitivity all run.
3. **Analysis pipelines run.** `per_cell_metrics.csv` ≥ 2,400 rows; `lex_dominance.csv` ≥ 4 rows; `smoothness.csv` ≥ 380 rows; `*_walltime.csv`, `*_ipopt.csv`, `*_compliance.csv` present for every instrumented cell.
4. **Figures populated.** All 26 inline figures resolved under [Figures/](Figures/). No `[Missing figure: ...]` boxes in the built PDF.
5. **Tables populated.** No `[TBP]` placeholders remain in any section subfile.
6. **§13 quirks pass.** `grep -E 'Reimannian|the the |close loop|nonholonomic' Sections/*.tex` returns no matches.
7. **Build clean.** `make paper` produces `Main.pdf` with no LaTeX errors, no `Warning: Reference ... undefined`, no orphan figures.
8. **Supplement built.** `make supplement` produces `supplement.pdf` ≤ 50 MB (or externally hosted with a DOI).
9. **Independent reviewer pass.** Spawn `code-reviewer` agent (cf. the project plan) over the full manuscript.
10. **Voice pass.** Spawn `hajieghrary-voice` agent for the first-person-plural register check.

Submission target: IEEE Transactions on Intelligent Vehicles.
