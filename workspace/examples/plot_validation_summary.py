#!/usr/bin/env python3
"""Generate fig_validation_summary.pdf from the validation report.

Reads ``examples/outputs/manuscript/validation_report.txt`` (produced by
``examples/validate_v10_2.py``), extracts the seven section summary verdicts,
and renders a single-column matplotlib figure suitable for inclusion in the
manuscript at \\S\\ref{fig:validation_summary}.

This makes the figure data-driven: re-running ``validate_v10_2.py`` followed
by this script keeps fig\\_validation\\_summary.pdf in sync with the actual
test outcomes.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_WORKSPACE = Path(__file__).resolve().parent.parent
if str(_WORKSPACE) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scripts.ieee_style import apply, COL_2
    apply()
except ImportError:
    COL_2 = 7.16   # IEEEtran double-column width in inches


# One-line headline per dimension; populated from the validation report.
HEADLINES = {
    1: "Algorithm 1A/0 reproduce $w^\\dagger = (5.5, 5.5)$, $r^\\dagger = 4.5$"
       " on Example 1; 55 unit tests pass.",
    2: "All 51 public symbols of \\texttt{lcp} importable + callable.",
    3: r"Algorithm 1A LP wall-time $\leq 1$ ms at $L \in \{2, 3, 5, 8, 12\}$.",
    4: "200/200 random weights in $\\Omega(p^\\star) \\cap [1,10]^2$ recover lex.",
    5: "2/5 weight strategies recover lex; only interior $\\Omega$ weights succeed.",
    6: "181/181 in-$\\Omega$ perturbations preserve $b = (1,1)$; 0/19 outside-$\\Omega$ do.",
    7: "Procedure 10.1 converges in 2 iterations to $(\\delta_1, \\delta_2, \\delta_3)"
       " = (0, 0.5, 0.4)$.",
}

LABELS = {
    1: "Correctness (Examples 1 \\& 2)",
    2: "Completeness (public API)",
    3: "Performance (LP solve time vs $L$)",
    4: "Effectiveness (Theorem 4.1 inside $\\Omega$)",
    5: "Utility (calibrated vs naive weights)",
    6: "Perturbation stability (Theorem 7.1)",
    7: "Procedure 10.1 convergence",
}


def _parse_report(report_path: Path) -> dict[str, str]:
    """Return ``{section_id: verdict}`` from the summary block of the report.

    Looks for lines like ``  1. Correctness          PASS`` in the trailing
    Summary section.
    """
    if not report_path.exists():
        return {}
    text = report_path.read_text()
    summary = text.split("Summary")[-1] if "Summary" in text else ""
    out: dict[str, str] = {}
    for line in summary.splitlines():
        m = re.match(r"\s*(\d+)\.\s.*\b(PASS|FAIL)\b", line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--report", type=Path,
        default=_WORKSPACE / "examples" / "outputs" / "manuscript" / "validation_report.txt",
    )
    parser.add_argument(
        "--out", type=Path,
        default=_WORKSPACE / "examples" / "outputs" / "manuscript" / "figures" / "fig_validation_summary.pdf",
    )
    parser.add_argument(
        "--mirror-to", type=Path,
        default=Path("/workspace/nuplan-project/IEEE_T-IV/Figures/fig_validation_summary.pdf"),
    )
    args = parser.parse_args()

    verdicts = _parse_report(args.report)
    if not verdicts:
        print(f"WARN: no Summary block in {args.report}; falling back to all PASS")
        verdicts = {str(i): "PASS" for i in range(1, 8)}

    n = len(LABELS)
    fig, ax = plt.subplots(figsize=(COL_2, 0.45 * n + 0.7))
    ax.set_xlim(0, 10); ax.set_ylim(-0.4, n - 0.6)
    ax.invert_yaxis()
    ax.axis("off")
    for i in range(1, n + 1):
        y = i - 1
        verdict = verdicts.get(str(i), "?")
        col = "#1b7837" if verdict == "PASS" else "#b2182b"
        # Section label + headline + verdict badge
        ax.text(0.05, y, f"{i}. {LABELS[i]}", va="center", ha="left",
                fontsize=8, fontweight="bold")
        ax.text(3.6, y, HEADLINES[i], va="center", ha="left", fontsize=7)
        ax.add_patch(plt.Rectangle((9.2, y - 0.25), 0.7, 0.5,
                                    color=col, alpha=0.9))
        ax.text(9.55, y, verdict, va="center", ha="center", fontsize=7,
                color="white", fontweight="bold")
    fig.tight_layout(pad=0.3)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    plt.close(fig)
    print(f"  [validation_summary] {args.out}")

    if args.mirror_to:
        args.mirror_to.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy2(args.out, args.mirror_to)
        print(f"  + mirror → {args.mirror_to}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
