# Makefile for the IEEE T-IV LCP-MPC deployment paper (Scope B).
#
# Three orthogonal phases (mirroring Appendix E):
#   A. Framework regression  — verifies the lcp/ reference implementation
#   B. Dual-condition protocol — runs the 16-scenario C1 and C4 batches
#   C. Artefact + paper build — generates figures/tables and compiles PDF
#
# Most targets do not invoke each other; chain them via `make all`.

PROJECT_ROOT  := $(abspath ..)
WORKSPACE     := $(PROJECT_ROOT)/workspace
PY            := $(shell command -v python3 || command -v python)
PYTHONPATH    := $(WORKSPACE)

MANUSCRIPT_OUT := $(WORKSPACE)/examples/outputs/manuscript
C1_OUT         := $(MANUSCRIPT_OUT)/C1_instrumented
C4_OUT         := $(MANUSCRIPT_OUT)/C4_cascade

NUPLAN_EXP_C1  := /workspace/nuplan-project/exp
NUPLAN_EXP_C4  := /workspace/exp_worker_c4/exp

PDFLATEX := pdflatex -interaction=nonstopmode -halt-on-error
BIBTEX   := bibtex

.PHONY: help all framework-regression \
        protocol-c1 protocol-c4 protocol \
        scans artefacts paper clean

help:
	@echo "v10_2 LCP-MPC deployment paper — build targets"
	@echo ""
	@echo "  framework-regression  Verify lcp/ against v10_2 worked Examples (~1 s)"
	@echo "  protocol-c1           Run C1 (operational WS-L1) on 16 scenarios (~1 h)"
	@echo "  protocol-c4           Run C4 (lex cascade) on 16 scenarios (~4 h)"
	@echo "  protocol              protocol-c1 + protocol-c4 (sequential)"
	@echo "  scans                 §8.5 diagnostics + §10.2 necessity scans on C1 logs"
	@echo "  artefacts             Generate figures + tables; mirror to IEEE_T-IV/"
	@echo "  paper                 Build Main.pdf (pdflatex x3 + bibtex)"
	@echo "  all                   framework-regression → protocol → scans → artefacts → paper"
	@echo "  clean                 Remove LaTeX build artefacts"

# ------------------------------------------------------------------
# Phase A: framework regression
# ------------------------------------------------------------------

framework-regression:
	cd $(WORKSPACE) && $(PY) -m pytest lcp/tests/ -q
	cd $(WORKSPACE) && $(PY) examples/lcp_demo_v10_2.py \
	    > $(MANUSCRIPT_OUT)/example_1_demo.txt
	cd $(WORKSPACE) && $(PY) examples/lcp_demo_example2.py \
	    > $(MANUSCRIPT_OUT)/example_2_demo.txt
	@echo "✓ framework regression: 55 tests pass; Example 1 + 2 demos captured"

# ------------------------------------------------------------------
# Phase B: dual-condition protocol
# ------------------------------------------------------------------

protocol-c1:
	cd $(WORKSPACE) && NUPLAN_EXP_ROOT=$(NUPLAN_EXP_C1) PYTHONPATH=$(PYTHONPATH) \
	    $(PY) examples/12_batch_two_level_mpc_planner.py \
	        --penalty-form l1 --runtime-mode ws --seed 7 \
	        --output-dir $(C1_OUT)

protocol-c4:
	cd $(WORKSPACE) && NUPLAN_EXP_ROOT=$(NUPLAN_EXP_C4) PYTHONPATH=$(PYTHONPATH) \
	    $(PY) examples/12_batch_two_level_mpc_planner.py \
	        --penalty-form l1 --runtime-mode cascade --seed 7 \
	        --output-dir $(C4_OUT)

protocol: protocol-c1 protocol-c4

# ------------------------------------------------------------------
# Phase C: artefact generation + paper build
# ------------------------------------------------------------------

scans:
	cd $(WORKSPACE) && $(PY) examples/scan_diagnostics.py
	cd $(WORKSPACE) && $(PY) examples/scan_necessity.py

artefacts: scans
	cd $(WORKSPACE) && $(PY) examples/make_manuscript_artefacts.py
	@echo "✓ artefacts regenerated and mirrored to IEEE_T-IV/Figures + IEEE_T-IV/Sections/tables"

paper:
	$(PDFLATEX) Main.tex
	-$(BIBTEX) Main
	$(PDFLATEX) Main.tex
	$(PDFLATEX) Main.tex

all: framework-regression protocol scans artefacts paper

clean:
	rm -f Main.aux Main.bbl Main.blg Main.log Main.out Main.toc \
	      Main.pdf Sections/*.aux

# ------------------------------------------------------------------
# Convenience: regenerate only the data-derived figures after a fresh
# protocol run, without re-running the long batches.
# ------------------------------------------------------------------

regen: scans artefacts
