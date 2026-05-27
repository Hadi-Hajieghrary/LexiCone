#!/bin/bash
# Re-generate all manuscript figures + tables from the latest C1 + C4 batch
# outputs. Run after both batches under
# workspace/examples/outputs/manuscript/{C1_instrumented,C4_cascade}/ have
# completed. Idempotent.
set -e
cd "$(dirname "$0")/.."
echo "[regen] re-running diagnostics scan..."
python3 examples/scan_diagnostics.py
echo "[regen] re-running necessity scan..."
python3 examples/scan_necessity.py
echo "[regen] re-running validation report (Examples 1+2 + framework dimensions)..."
python3 examples/validate_v10_2.py \
    --out examples/outputs/manuscript/validation_report.txt > /dev/null
echo "[regen] re-rendering validation-summary figure..."
python3 examples/plot_validation_summary.py
echo "[regen] re-generating figures + tables..."
python3 examples/make_manuscript_artefacts.py
echo "[regen] done."
echo "        figures: IEEE_T-IV/Figures/fig{10,11,12,18,22,28,_validation_summary}*.pdf"
echo "        tables:  IEEE_T-IV/Sections/tables/table_*.tex"
echo "        re-build paper:  cd IEEE_T-IV && pdflatex Main && pdflatex Main && bibtex Main && pdflatex Main"
