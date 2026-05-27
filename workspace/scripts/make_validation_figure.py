#!/usr/bin/env python3
"""Generate the 7-dimension validation summary figure for the manuscript.

Renders a horizontal bar chart annotated with the headline outcome of each of
the seven validation dimensions exercised by ``validate_v10_2.py``. The figure
is intended for §VII (Results) or as a one-glance front-piece of the
supplement.
"""
from __future__ import annotations

import sys
from pathlib import Path

_WORKSPACE = Path(__file__).resolve().parents[1]
if str(_WORKSPACE) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from scripts.ieee_style import apply, COL_2
    apply()
except ImportError:
    COL_2 = 7.16   # IEEEtran twocolumn double-column width (inches)


DIMENSIONS = [
    ("Correctness",        "Examples 1 \\& 2 reproduce paper to $10^{-6}$",
     "PASS", "55 / 55 unit tests"),
    ("Completeness",       "Every public symbol importable",
     "PASS", "51 / 51 symbols"),
    ("Performance",        "Algorithm 1A LP wall-time vs depth $L$",
     "PASS", "0.8 ms ($L{=}2$) -- 1.0 ms ($L{=}12$)"),
    ("Effectiveness",      "Theorem 4.1 in random in-$\\Omega$ weights",
     "PASS", "200 / 200 recover $z_{\\rm lex}$"),
    ("Utility",            "Calibrated vs uncalibrated weight strategies",
     "PASS", "2 / 5 strategies recover lex"),
    ("Perturbation stab.", "Theorem 7.1 stability under in-/out-$\\Omega$ noise",
     "PASS", "in-$\\Omega$: 200 / 200; out: 0 / 19"),
    ("Proc.\\ 10.1 conv.",  "Necessity + utility relaxation",
     "PASS", "converged in 2 iters"),
]


def main() -> int:
    n = len(DIMENSIONS)
    fig, ax = plt.subplots(figsize=(COL_2, 2.9))
    y = np.arange(n)

    # Bar: green if PASS, red if FAIL (here all PASS).
    colours = ["#2ca02c" if d[2] == "PASS" else "#d62728" for d in DIMENSIONS]
    bars = ax.barh(y, [1.0] * n, color=colours, edgecolor="black", linewidth=0.4, height=0.7)

    # Label each bar with the dimension name, the test description, and the result.
    for i, (name, descr, verdict, detail) in enumerate(DIMENSIONS):
        ax.text(0.02, i, f"  {i+1}. {name}", va="center", ha="left",
                color="white", fontsize=8, weight="bold")
        ax.text(0.30, i, descr, va="center", ha="left", color="white", fontsize=7)
        ax.text(0.98, i, f"{verdict}\\,$\\bullet$\\,{detail}",
                va="center", ha="right", color="white", fontsize=7, weight="bold")

    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlim(0, 1.0)
    ax.invert_yaxis()
    for spine in ("top", "right", "bottom", "left"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout(pad=0.2)

    out = Path("/workspace/nuplan-project/IEEE_T-IV/Figures/fig_validation_summary.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")

    # Also drop a copy into workspace/examples/outputs/manuscript/figures/
    out2 = _WORKSPACE / "examples" / "outputs" / "manuscript" / "figures" / "fig_validation_summary.pdf"
    out2.parent.mkdir(parents=True, exist_ok=True)
    import shutil; shutil.copy2(out, out2)
    print(f"mirrored to {out2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
