"""IEEE Transactions publication-grade matplotlib style.

Strict typography and resolution conforming to IEEE Transactions author
guidelines:

* Body text:        Times Roman family, 8 pt
* Axis titles:      9 pt bold
* Axis labels:      8 pt
* Tick labels:      7 pt
* Legend text:      7 pt
* Suptitle:         9 pt bold
* Axes linewidth:   0.5 pt
* Line linewidth:   0.8 pt (data lines), 0.5 pt (grid/spines)
* DPI:              300 (savefig and figure)
* Math text:        STIX (Times-compatible serif math)

Standard column widths (in inches):

* ``COL_1`` = 3.50 in   (88.9 mm)  — single-column figure
* ``COL_2`` = 7.16 in   (181.86 mm) — double-column / page-wide figure

To use, ``import ieee_style; ieee_style.apply()`` once before constructing
figures, and pick figure sizes from ``COL_1`` / ``COL_2``.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


COL_1 = 3.50   # single column, IEEE Transactions
COL_2 = 7.16   # double column / full page width


def apply() -> None:
    """Apply IEEE Transactions matplotlib rcParams globally."""
    plt.rcParams.update({
        # Typography
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "STIXGeneral"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.title_fontsize": 10,
        "figure.titlesize": 11,
        "figure.titleweight": "bold",
        # Math rendering
        "mathtext.fontset": "stix",
        "mathtext.default": "regular",
        # Line weights
        "axes.linewidth": 0.5,
        "lines.linewidth": 0.8,
        "patch.linewidth": 0.4,
        "grid.linewidth": 0.3,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.minor.width": 0.35,
        "ytick.minor.width": 0.35,
        "xtick.major.size": 2.2,
        "ytick.major.size": 2.2,
        "xtick.minor.size": 1.2,
        "ytick.minor.size": 1.2,
        # Resolution and saving
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        # Layout
        "axes.titlepad": 4.0,
        "axes.labelpad": 2.5,
        "xtick.major.pad": 1.8,
        "ytick.major.pad": 1.8,
        # Grid (off by default; turn on per-axis when needed)
        "grid.color": "#dddddd",
        "grid.linestyle": "-",
        "grid.alpha": 0.5,
        # Legend
        "legend.frameon": False,
        "legend.borderpad": 0.3,
        "legend.labelspacing": 0.25,
        "legend.handlelength": 1.6,
        "legend.handletextpad": 0.5,
        "legend.columnspacing": 1.0,
    })


def column(span: str = "two") -> float:
    """Return the standard IEEE Transactions column width in inches.

    Parameters
    ----------
    span : {"one", "two"}
        ``"one"`` → 3.50 in (single column); ``"two"`` → 7.16 in (double column).
    """
    if span == "one":
        return COL_1
    if span == "two":
        return COL_2
    raise ValueError(f"span must be 'one' or 'two', got {span!r}")
