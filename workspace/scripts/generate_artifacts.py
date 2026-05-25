"""Generate plots, figures, and diagrams supporting the LCP paper.

Produces three artifact groups under ``examples/outputs/artifacts/``:

* **per_scenario/<label>/** — 5 figures derived from each scenario's per-tick
  rule-evaluation CSV. Augments the existing ``<label>_summary.png``.

* **aggregate/** — ~10 figures aggregating across all 16 benchmark scenarios.

* **theory/** — ~10 conceptual figures derived from synthetic toy problems
  (worked-example reproductions, algorithm flowcharts as block diagrams,
  rulebook hierarchy, upper-image geometry).

Run from the workspace root:

    python scripts/generate_artifacts.py
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy.optimize import linprog
from scipy.spatial import ConvexHull


# ----------------------------------------------------------------------
# Paths and global style
# ----------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
BATCH_DIR = ROOT / "examples" / "outputs" / "12_batch_two_level_mpc_planner"
OUT = ROOT / "examples" / "outputs" / "artifacts"
PER = OUT / "per_scenario"
AGG = OUT / "aggregate"
THEORY = OUT / "theory"
for d in (OUT, PER, AGG, THEORY):
    d.mkdir(parents=True, exist_ok=True)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ieee_style  # noqa: E402
ieee_style.apply()
COL_1 = ieee_style.COL_1   # 3.50 in
COL_2 = ieee_style.COL_2   # 7.16 in

VIOLATION_CMAP = LinearSegmentedColormap.from_list(
    "violation", [(1, 1, 1), (1, 0.92, 0.62), (1, 0.36, 0.36), (0.55, 0, 0)]
)
APPLY_CMAP = LinearSegmentedColormap.from_list(
    "apply", [(0.97, 0.97, 0.97), (0.27, 0.45, 0.77)]
)

# Priority-level → colour
LEVEL_COLOURS = {
    10: "#7f1d1d", 9: "#c0392b", 8: "#e67e22", 7: "#f1c40f",
    3: "#16a085", 2: "#2980b9", 1: "#8e44ad", 0: "#7f8c8d",
}

LEVEL_FROM_RID: Dict[str, int] = {
    "10r0": 10, "10r3": 10, "10r4": 10, "10r5": 10,
    "9r0": 9, "9r1": 9,
    "8r0": 8, "8r1": 8,
    "7r0": 7, "7r1": 7, "7r2": 7, "7r3": 7, "7r4": 7, "7r5": 7,
    "3r0": 3, "3r3": 3, "3r5": 3, "3r6": 3,
    "2r2": 2,
    "1r0": 1, "1r2": 1, "1r5": 1, "1r11": 1,
    "0r2": 0, "0r3": 0,
}


# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------


@dataclass
class ScenarioLog:
    label: str
    rule_ids: List[str]
    rule_names: Dict[str, str]
    timestamps: np.ndarray         # shape (T,)
    applies: np.ndarray            # shape (R, T)
    rates: np.ndarray              # shape (R, T)
    is_violated: np.ndarray        # shape (R, T) bool

    @property
    def n_ticks(self) -> int:
        return len(self.timestamps)

    @property
    def dts(self) -> np.ndarray:
        if len(self.timestamps) < 2:
            return np.array([0.1])
        d = np.diff(self.timestamps)
        return np.concatenate([[d[0]], d])

    def integrated_violation(self) -> np.ndarray:
        """Per-rule ∫ rate · dt over applicable ticks. Shape (R,)."""
        return ((self.rates * self.applies) * self.dts[None, :]).sum(axis=1)

    def applicable_counts(self) -> np.ndarray:
        return self.applies.sum(axis=1)

    def violation_counts(self) -> np.ndarray:
        return self.is_violated.sum(axis=1)


def load_scenario(label: str) -> ScenarioLog:
    sd = BATCH_DIR / f"{label}__l1"
    csv_path = next(sd.glob("*_log.csv"))
    by_tick: Dict[float, Dict[str, Tuple[int, int, float, str]]] = {}
    rule_names: Dict[str, str] = {}
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            t = float(row["t_s"])
            rid = row["rule_id"]
            by_tick.setdefault(t, {})[rid] = (
                int(row["applies"]), int(row["is_violated"]),
                float(row["violation_rate"]), row["rule_name"],
            )
            rule_names[rid] = row["rule_name"]
    timestamps = np.array(sorted(by_tick.keys()))
    rule_ids = sorted(rule_names.keys(), key=lambda r: -LEVEL_FROM_RID.get(r, -1))
    R, T = len(rule_ids), len(timestamps)
    applies = np.zeros((R, T), dtype=np.int8)
    rates = np.zeros((R, T))
    violated = np.zeros((R, T), dtype=bool)
    for j, t in enumerate(timestamps):
        for i, rid in enumerate(rule_ids):
            cell = by_tick[t].get(rid)
            if cell is None:
                continue
            ap, vi, rt, _ = cell
            applies[i, j] = ap
            rates[i, j] = rt
            violated[i, j] = bool(vi)
    return ScenarioLog(label, rule_ids, rule_names, timestamps - timestamps[0],
                       applies, rates, violated)


# ----------------------------------------------------------------------
# Per-scenario figures
# ----------------------------------------------------------------------


def fig_p1_top_rules_timeline(log: ScenarioLog, out: Path) -> None:
    """P1 — Top-N rules' violation-rate timeline (line plot)."""
    integ = log.integrated_violation()
    top_idx = np.argsort(-integ)[:6]
    top_idx = [i for i in top_idx if integ[i] > 0][:6] or list(top_idx[:1])
    fig, ax = plt.subplots(figsize=(7.16, 2.92), constrained_layout=True)
    for i in top_idx:
        rid = log.rule_ids[i]
        ax.plot(log.timestamps, log.rates[i], lw=1.6,
                color=LEVEL_COLOURS.get(LEVEL_FROM_RID.get(rid, 0), "#666"),
                label=f"{rid}  {log.rule_names[rid][:32]}", alpha=0.9)
    ax.set_xlabel("t (s)"); ax.set_ylabel("violation rate")
    ax.set_title(f"{log.label}  ·  top rules — violation-rate timeline", loc="left", fontweight="bold")
    ax.legend(loc="upper right", frameon=False, ncol=1)
    ax.grid(True, color="#eee", lw=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.savefig(out / "P1_top_rules_timeline.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_p2_applicability_heatmap(log: ScenarioLog, out: Path) -> None:
    """P2 — Per-rule applicability heatmap (binary, rule × tick)."""
    fig, ax = plt.subplots(figsize=(7.16, 2.98), constrained_layout=True)
    im = ax.imshow(log.applies, aspect="auto", cmap=APPLY_CMAP, vmin=0, vmax=1, interpolation="nearest")
    ax.set_yticks(range(len(log.rule_ids)))
    ax.set_yticklabels(log.rule_ids)
    ax.set_xlabel("tick"); ax.set_ylabel("rule")
    ax.set_title(f"{log.label}  ·  per-rule applicability mask", loc="left", fontweight="bold")
    cb = fig.colorbar(im, ax=ax, fraction=0.018, pad=0.01, ticks=[0, 1])
    cb.set_label("applies")
    fig.savefig(out / "P2_applicability_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_p3_violation_heatmap(log: ScenarioLog, out: Path) -> None:
    """P3 — Per-rule violation-rate heatmap (rule × tick)."""
    fig, ax = plt.subplots(figsize=(7.16, 3.58), constrained_layout=True)
    vmax = max(log.rates.max(), 1e-3)
    im = ax.imshow(log.rates, aspect="auto", cmap=VIOLATION_CMAP, vmin=0, vmax=vmax, interpolation="nearest")
    ax.set_yticks(range(len(log.rule_ids)))
    ax.set_yticklabels(log.rule_ids)
    ax.set_xlabel("tick"); ax.set_ylabel("rule")
    ax.set_title(f"{log.label}  ·  per-rule violation rate", loc="left", fontweight="bold")
    cb = fig.colorbar(im, ax=ax, fraction=0.018, pad=0.01)
    cb.set_label("rate")
    fig.savefig(out / "P3_violation_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_p4_cumulative_violation(log: ScenarioLog, out: Path) -> None:
    """P4 — Cumulative integrated violation over time for top-N rules."""
    integ = log.integrated_violation()
    top_idx = np.argsort(-integ)[:6]
    top_idx = [i for i in top_idx if integ[i] > 0][:6] or list(top_idx[:1])
    fig, ax = plt.subplots(figsize=(7.16, 2.92), constrained_layout=True)
    for i in top_idx:
        rid = log.rule_ids[i]
        cum = np.cumsum(log.rates[i] * log.applies[i] * log.dts)
        ax.plot(log.timestamps, cum, lw=1.8,
                color=LEVEL_COLOURS.get(LEVEL_FROM_RID.get(rid, 0), "#666"),
                label=f"{rid}  → {cum[-1]:.2f}", alpha=0.95)
    ax.set_xlabel("t (s)"); ax.set_ylabel(r"$\int_0^t \mathrm{rate}\,d\tau$")
    ax.set_title(f"{log.label}  ·  cumulative violation by rule", loc="left", fontweight="bold")
    ax.legend(loc="upper left", frameon=False)
    ax.grid(True, color="#eee", lw=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.savefig(out / "P4_cumulative_violation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_p5_violation_event_timeline(log: ScenarioLog, out: Path) -> None:
    """P5 — Scatter of violation events: (tick, rule) for every is_violated."""
    fig, ax = plt.subplots(figsize=(7.16, 3.58), constrained_layout=True)
    y_pos = {rid: i for i, rid in enumerate(log.rule_ids)}
    for i, rid in enumerate(log.rule_ids):
        ticks_violating = np.where(log.is_violated[i])[0]
        if len(ticks_violating) == 0:
            continue
        rates_v = log.rates[i, ticks_violating]
        color = LEVEL_COLOURS.get(LEVEL_FROM_RID.get(rid, 0), "#666")
        ax.scatter(ticks_violating, [i] * len(ticks_violating),
                   s=12 + 80 * (rates_v / max(rates_v.max(), 1e-6)),
                   c=color, alpha=0.7, edgecolors="white", linewidths=0.4)
    ax.set_yticks(range(len(log.rule_ids)))
    ax.set_yticklabels(log.rule_ids)
    ax.set_xlabel("tick"); ax.set_ylabel("rule")
    ax.set_title(f"{log.label}  ·  violation events (size ∝ rate, colour = level)",
                 loc="left", fontweight="bold")
    ax.grid(True, axis="x", color="#eee", lw=0.4)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.savefig(out / "P5_violation_events.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_p6_priority_level_stacks(log: ScenarioLog, out: Path) -> None:
    """P6 — Per-tick stacked-area chart of violation rate summed by level."""
    levels = sorted({LEVEL_FROM_RID.get(r, 0) for r in log.rule_ids}, reverse=True)
    per_level = np.zeros((len(levels), log.n_ticks))
    for i, rid in enumerate(log.rule_ids):
        lv = LEVEL_FROM_RID.get(rid, 0)
        per_level[levels.index(lv)] += log.rates[i] * log.applies[i]
    fig, ax = plt.subplots(figsize=(7.16, 2.92), constrained_layout=True)
    bottom = np.zeros(log.n_ticks)
    for k, lv in enumerate(levels):
        ax.fill_between(log.timestamps, bottom, bottom + per_level[k],
                        color=LEVEL_COLOURS.get(lv, "#888"),
                        alpha=0.85, label=f"L{lv}")
        bottom = bottom + per_level[k]
    ax.set_xlabel("t (s)"); ax.set_ylabel("Σ rate (per level)")
    ax.set_title(f"{log.label}  ·  total violation rate stacked by priority level",
                 loc="left", fontweight="bold")
    ax.legend(loc="upper right", frameon=False, ncol=4)
    ax.grid(True, color="#eee", lw=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.savefig(out / "P6_priority_level_stacks.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# Aggregate figures
# ----------------------------------------------------------------------


def fig_a1_top_violation_per_scenario(logs: List[ScenarioLog], out: Path) -> None:
    """A1 — Top integrated violation per scenario (sorted bar chart)."""
    labels = [l.label for l in logs]
    integ = [l.integrated_violation().max() for l in logs]
    top_rid = [l.rule_ids[int(np.argmax(l.integrated_violation()))] for l in logs]
    order = np.argsort(integ)[::-1]
    labels_s = [labels[i] for i in order]
    integ_s = [integ[i] for i in order]
    top_rid_s = [top_rid[i] for i in order]
    colors = [LEVEL_COLOURS.get(LEVEL_FROM_RID.get(r, 0), "#666") for r in top_rid_s]
    fig, ax = plt.subplots(figsize=(7.16, 3.58), constrained_layout=True)
    bars = ax.barh(range(len(labels_s)), integ_s, color=colors, alpha=0.9)
    for i, (v, r) in enumerate(zip(integ_s, top_rid_s)):
        ax.text(v, i, f"  {v:.2f}  ({r})", va="center", fontsize=11)
    ax.set_yticks(range(len(labels_s))); ax.set_yticklabels(labels_s)
    ax.invert_yaxis()
    ax.set_xlabel(r"top-rule integrated violation  $\int \mathrm{rate}\,dt$  [unit·s]")
    ax.set_title("Per-scenario top-rule integrated violation (sorted)", loc="left", fontweight="bold")
    ax.grid(True, axis="x", color="#eee", lw=0.5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.savefig(out / "A1_top_violation_per_scenario.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_a2_per_rule_aggregate(logs: List[ScenarioLog], out: Path) -> None:
    """A2 — Aggregate integrated violation per rule, across all scenarios."""
    all_rids = sorted(set(rid for l in logs for rid in l.rule_ids),
                      key=lambda r: -LEVEL_FROM_RID.get(r, 0))
    totals = {rid: 0.0 for rid in all_rids}
    for l in logs:
        integ = l.integrated_violation()
        for i, rid in enumerate(l.rule_ids):
            totals[rid] += integ[i]
    rids_sorted = sorted(all_rids, key=lambda r: -totals[r])
    vals = [totals[r] for r in rids_sorted]
    colors = [LEVEL_COLOURS.get(LEVEL_FROM_RID.get(r, 0), "#666") for r in rids_sorted]
    fig, ax = plt.subplots(figsize=(7.16, 4.55), constrained_layout=True)
    ax.barh(range(len(rids_sorted)), vals, color=colors, alpha=0.9)
    for i, v in enumerate(vals):
        if v > 0:
            ax.text(v, i, f"  {v:.2f}", va="center", fontsize=11)
    ax.set_yticks(range(len(rids_sorted)))
    ax.set_yticklabels([f"{r}  L{LEVEL_FROM_RID.get(r,0)}" for r in rids_sorted], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel(r"aggregate integrated violation across 16 scenarios")
    ax.set_title("Per-rule aggregate violation across the benchmark", loc="left", fontweight="bold")
    ax.grid(True, axis="x", color="#eee", lw=0.5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.savefig(out / "A2_per_rule_aggregate.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_a3_rule_x_scenario_heatmap(logs: List[ScenarioLog], out: Path) -> None:
    """A3 — Rule × scenario integrated-violation heatmap."""
    all_rids = sorted(set(rid for l in logs for rid in l.rule_ids),
                      key=lambda r: -LEVEL_FROM_RID.get(r, 0))
    M = np.zeros((len(all_rids), len(logs)))
    for j, l in enumerate(logs):
        integ = l.integrated_violation()
        for i, rid in enumerate(l.rule_ids):
            row = all_rids.index(rid)
            M[row, j] = integ[i]
    fig, ax = plt.subplots(figsize=(7.16, 4.30), constrained_layout=True)
    im = ax.imshow(M, aspect="auto", cmap=VIOLATION_CMAP, vmin=0,
                   vmax=max(M.max(), 1e-3), interpolation="nearest")
    ax.set_yticks(range(len(all_rids)))
    ax.set_yticklabels([f"{r}  L{LEVEL_FROM_RID.get(r,0)}" for r in all_rids], fontsize=10)
    ax.set_xticks(range(len(logs)))
    ax.set_xticklabels([l.label for l in logs], rotation=45, ha="right")
    ax.set_title("Rule × scenario integrated violation", loc="left", fontweight="bold")
    cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    cb.set_label(r"$\int \mathrm{rate}\,dt$  [unit·s]")
    fig.savefig(out / "A3_rule_x_scenario_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_a4_violation_count_heatmap(logs: List[ScenarioLog], out: Path) -> None:
    """A4 — Rule × scenario violation-tick-count heatmap."""
    all_rids = sorted(set(rid for l in logs for rid in l.rule_ids),
                      key=lambda r: -LEVEL_FROM_RID.get(r, 0))
    M = np.zeros((len(all_rids), len(logs)), dtype=int)
    for j, l in enumerate(logs):
        for i, rid in enumerate(l.rule_ids):
            M[all_rids.index(rid), j] = int(l.is_violated[i].sum())
    fig, ax = plt.subplots(figsize=(7.16, 4.30), constrained_layout=True)
    im = ax.imshow(M, aspect="auto", cmap="Reds", vmin=0,
                   vmax=max(M.max(), 1), interpolation="nearest")
    ax.set_yticks(range(len(all_rids)))
    ax.set_yticklabels([f"{r}" for r in all_rids])
    ax.set_xticks(range(len(logs)))
    ax.set_xticklabels([l.label for l in logs], rotation=45, ha="right")
    ax.set_title("Rule × scenario violation-tick count", loc="left", fontweight="bold")
    cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    cb.set_label("# violation ticks")
    fig.savefig(out / "A4_violation_count_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_a5_priority_level_breakdown(logs: List[ScenarioLog], out: Path) -> None:
    """A5 — Aggregate violation per priority level (stacked across scenarios)."""
    labels = [l.label for l in logs]
    levels_sorted = sorted({LEVEL_FROM_RID.get(r, 0) for l in logs for r in l.rule_ids},
                           reverse=True)
    matrix = np.zeros((len(levels_sorted), len(logs)))
    for j, l in enumerate(logs):
        integ = l.integrated_violation()
        for i, rid in enumerate(l.rule_ids):
            lv = LEVEL_FROM_RID.get(rid, 0)
            matrix[levels_sorted.index(lv), j] += integ[i]
    fig, ax = plt.subplots(figsize=(7.16, 3.30), constrained_layout=True)
    bottom = np.zeros(len(logs))
    for k, lv in enumerate(levels_sorted):
        ax.bar(range(len(logs)), matrix[k], bottom=bottom,
               color=LEVEL_COLOURS.get(lv, "#888"),
               label=f"L{lv}", alpha=0.9)
        bottom = bottom + matrix[k]
    ax.set_xticks(range(len(logs)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel(r"Σ integrated violation by level")
    ax.set_title("Per-scenario violation breakdown by priority level", loc="left", fontweight="bold")
    ax.legend(loc="upper right", frameon=False, ncol=4)
    ax.grid(True, axis="y", color="#eee", lw=0.5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.savefig(out / "A5_priority_level_breakdown.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_a6_applicability_frequency(logs: List[ScenarioLog], out: Path) -> None:
    """A6 — Per-rule applicability frequency across the benchmark."""
    all_rids = sorted(set(rid for l in logs for rid in l.rule_ids),
                      key=lambda r: -LEVEL_FROM_RID.get(r, 0))
    total_appl = {r: 0 for r in all_rids}
    total_viol = {r: 0 for r in all_rids}
    total_ticks = 0
    for l in logs:
        total_ticks += l.n_ticks
        for i, rid in enumerate(l.rule_ids):
            total_appl[rid] += int(l.applies[i].sum())
            total_viol[rid] += int(l.is_violated[i].sum())
    x = np.arange(len(all_rids))
    appl_pct = [total_appl[r] / max(total_ticks, 1) * 100 for r in all_rids]
    viol_pct = [total_viol[r] / max(total_ticks, 1) * 100 for r in all_rids]
    fig, ax = plt.subplots(figsize=(7.16, 3.03), constrained_layout=True)
    ax.bar(x - 0.2, appl_pct, width=0.4, color="#4472c4", label="% ticks applicable")
    ax.bar(x + 0.2, viol_pct, width=0.4, color="#c0392b", label="% ticks violating")
    ax.set_xticks(x); ax.set_xticklabels(all_rids, rotation=70)
    ax.set_ylabel(f"% of {total_ticks} benchmark ticks")
    ax.set_title("Per-rule applicability and violation frequency across 16 scenarios",
                 loc="left", fontweight="bold")
    ax.legend(loc="upper right", frameon=False)
    ax.grid(True, axis="y", color="#eee", lw=0.5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.savefig(out / "A6_applicability_frequency.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_a7_rule_cooccurrence(logs: List[ScenarioLog], out: Path) -> None:
    """A7 — Rule × rule co-occurrence matrix (both violating same tick)."""
    all_rids = sorted(set(rid for l in logs for rid in l.rule_ids),
                      key=lambda r: -LEVEL_FROM_RID.get(r, 0))
    R = len(all_rids); M = np.zeros((R, R), dtype=int)
    idx = {r: k for k, r in enumerate(all_rids)}
    for l in logs:
        for t in range(l.n_ticks):
            v = [l.rule_ids[i] for i in range(len(l.rule_ids)) if l.is_violated[i, t]]
            for a in v:
                for b in v:
                    M[idx[a], idx[b]] += 1
    fig, ax = plt.subplots(figsize=(7.16, 6.51), constrained_layout=True)
    diag = np.diag(M).copy()
    np.fill_diagonal(M, 0)
    vmax = max(M.max(), 1)
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=vmax, interpolation="nearest")
    ax.set_xticks(range(R)); ax.set_xticklabels(all_rids, rotation=70)
    ax.set_yticks(range(R)); ax.set_yticklabels(all_rids)
    for i in range(R):
        if diag[i] > 0:
            ax.text(i, i, f"{diag[i]}", ha="center", va="center", fontsize=9,
                    color="#444", fontweight="bold")
    ax.set_title("Rule × rule co-violation count (off-diag = co-occur; diag annotated)",
                 loc="left", fontweight="bold")
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("# ticks co-violated")
    fig.savefig(out / "A7_rule_cooccurrence.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_a8_violation_duration_distribution(logs: List[ScenarioLog], out: Path) -> None:
    """A8 — Distribution of contiguous-violation-burst durations per rule."""
    bursts: Dict[str, List[int]] = defaultdict(list)
    for l in logs:
        for i, rid in enumerate(l.rule_ids):
            cur = 0
            for t in range(l.n_ticks):
                if l.is_violated[i, t]:
                    cur += 1
                else:
                    if cur > 0:
                        bursts[rid].append(cur)
                    cur = 0
            if cur > 0:
                bursts[rid].append(cur)
    rules_with_bursts = sorted(bursts, key=lambda r: -sum(bursts[r]))[:12]
    if not rules_with_bursts:
        return
    fig, ax = plt.subplots(figsize=(7.16, 3.58), constrained_layout=True)
    data = [bursts[r] for r in rules_with_bursts]
    positions = range(1, len(rules_with_bursts) + 1)
    bp = ax.boxplot(data, positions=positions, widths=0.55, patch_artist=True,
                    showfliers=True, flierprops=dict(marker=".", markersize=4))
    for patch, rid in zip(bp["boxes"], rules_with_bursts):
        patch.set_facecolor(LEVEL_COLOURS.get(LEVEL_FROM_RID.get(rid, 0), "#888"))
        patch.set_alpha(0.7)
    ax.set_xticks(positions); ax.set_xticklabels(rules_with_bursts, rotation=45)
    ax.set_ylabel("burst length (# consecutive violation ticks)")
    ax.set_title("Violation-burst duration distribution per rule (top-12 by total burst time)",
                 loc="left", fontweight="bold")
    ax.grid(True, axis="y", color="#eee", lw=0.5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.savefig(out / "A8_violation_duration_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_a9_aggregate_dashboard(logs: List[ScenarioLog], out: Path) -> None:
    """A9 — One-figure summary dashboard."""
    labels = [l.label for l in logs]
    tot_violated_rules = [int((l.integrated_violation() > 0).sum()) for l in logs]
    tot_integ = [l.integrated_violation().sum() for l in logs]
    n_ticks = [l.n_ticks for l in logs]
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 4.30), constrained_layout=True)
    ax = axes[0, 0]
    ax.bar(range(len(logs)), tot_integ, color="#c0392b", alpha=0.8)
    ax.set_xticks(range(len(logs)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel(r"Σ integrated violation (all rules)")
    ax.set_title("Total integrated violation per scenario", fontweight="bold", loc="left")
    ax.grid(True, axis="y", color="#eee", lw=0.5)
    ax = axes[0, 1]
    ax.bar(range(len(logs)), tot_violated_rules, color="#4472c4", alpha=0.8)
    ax.set_xticks(range(len(logs)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("# violating rules")
    ax.set_title("Number of distinct violating rules per scenario", fontweight="bold", loc="left")
    ax.grid(True, axis="y", color="#eee", lw=0.5)
    ax = axes[1, 0]
    ax.bar(range(len(logs)), n_ticks, color="#16a085", alpha=0.8)
    ax.set_xticks(range(len(logs)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("# ticks")
    ax.set_title("Episode length per scenario", fontweight="bold", loc="left")
    ax.grid(True, axis="y", color="#eee", lw=0.5)
    ax = axes[1, 1]
    txt = (
        f"Benchmark: {len(logs)} scenarios\n"
        f"Total ticks: {sum(n_ticks):,}\n"
        f"Aggregate ∫ violation: {sum(tot_integ):.2f}\n"
        f"Mean violating rules / scenario: {np.mean(tot_violated_rules):.1f}\n"
        f"Top scenario (by top-rule violation):\n"
        f"  {logs[int(np.argmax([l.integrated_violation().max() for l in logs]))].label}"
    )
    ax.text(0.02, 0.5, txt, transform=ax.transAxes, fontsize=10,
            family="DejaVu Sans Mono", va="center")
    ax.set_axis_off()
    fig.suptitle("Aggregate benchmark dashboard", fontsize=11, fontweight="bold")
    fig.savefig(out / "A9_aggregate_dashboard.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_a10_scenario_similarity(logs: List[ScenarioLog], out: Path) -> None:
    """A10 — Pairwise scenario similarity by violation profile (cosine)."""
    all_rids = sorted(set(rid for l in logs for rid in l.rule_ids))
    M = np.zeros((len(logs), len(all_rids)))
    for j, l in enumerate(logs):
        for i, rid in enumerate(l.rule_ids):
            M[j, all_rids.index(rid)] = l.integrated_violation()[i]
    norms = np.linalg.norm(M, axis=1, keepdims=True) + 1e-9
    sim = (M @ M.T) / (norms @ norms.T)
    fig, ax = plt.subplots(figsize=(7.16, 6.44), constrained_layout=True)
    im = ax.imshow(sim, cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks(range(len(logs)))
    ax.set_xticklabels([l.label for l in logs], rotation=45, ha="right")
    ax.set_yticks(range(len(logs)))
    ax.set_yticklabels([l.label for l in logs])
    ax.set_title("Scenario × scenario violation-profile cosine similarity",
                 loc="left", fontweight="bold")
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("cosine similarity")
    fig.savefig(out / "A10_scenario_similarity.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# Theory / algorithm / worked-example / architecture figures
# ----------------------------------------------------------------------


def fig_t1_upper_image_l1(out: Path) -> None:
    """T1 — Upper image for a 2-priority L₁ LP (polygon with lex vertex)."""
    # Toy upper image: {(V1, V2, J): V1+V2 >= 0, V1 >= a*J, V2 >= b*J, J >= 1}.
    # We plot the 2D slice at J=1 to keep it readable.
    fig, ax = plt.subplots(figsize=(3.50, 3.27), constrained_layout=True)
    pts = np.array([[0, 0], [0, 3.5], [1.4, 3.5], [3.5, 1.4], [3.5, 0]])
    ax.fill(pts[:, 0], pts[:, 1], facecolor="#e9eff8", edgecolor="#2980b9", lw=1.8, alpha=0.9)
    # Lex point: minimise V1, then V2 → lex = (0, lowest-V2-with-V1=0) = (0, 0)
    lex = np.array([0.0, 0.0])
    ax.plot(*lex, "o", color="#c0392b", markersize=5, zorder=5)
    ax.annotate(r"$p^\star = (V_1^\star, V_2^\star)$", lex, xytext=(0.25, 0.4),
                textcoords="data", fontsize=10, color="#c0392b")
    # WS support hyperplane for some w
    w = np.array([1.0, 1.5])
    # support line  w·x = w·p_star = 0
    xs = np.linspace(-0.5, 4, 50)
    ax.plot(xs, -(w[0] / w[1]) * xs, "--", color="#27ae60", lw=1.6, alpha=0.8,
            label=fr"WS hyperplane  $w \cdot p = 0,\ w = ({w[0]:.1f},{w[1]:.1f})$")
    ax.set_xlim(-0.5, 4); ax.set_ylim(-0.5, 4); ax.set_aspect("equal")
    ax.set_xlabel(r"$V_1$"); ax.set_ylabel(r"$V_2$")
    ax.set_title(r"Upper image $\overline{\mathcal{P}}$ at $J=1$ slice (2-priority $L_1$)",
                 loc="left", fontweight="bold")
    ax.grid(True, color="#eee", lw=0.4)
    ax.legend(loc="upper right", frameon=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.savefig(out / "T1_upper_image_l1.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_t2_equivalence_cone(out: Path) -> None:
    """T2 — Equivalence cone and unit-performance slice in weight space."""
    fig, ax = plt.subplots(figsize=(3.50, 3.27), constrained_layout=True)
    # Toy: equivalence region for some lex point — say w1 ∈ [1, 10], w2 ∈ [1, 10],
    # with constraint w1 + 2*w2 <= 22 (polytope vertex at (10, 6))
    bx = np.array([[1, 1], [10, 1], [10, 6], [2, 10], [1, 10]], dtype=float)
    ax.fill(bx[:, 0], bx[:, 1], facecolor="#fff3d6", edgecolor="#e67e22", lw=1.8, alpha=0.95,
            label=r"$\Omega(p^\star)\cap\mathrm{box}$")
    # Chebyshev centre
    from scipy.spatial import ConvexHull
    hull = ConvexHull(bx)
    # Approximate Chebyshev: solve LP for largest inscribed circle
    # max r s.t. n_i^T x + r <= b_i  for each edge
    A_ub, b_ub = [], []
    for s in hull.simplices:
        p, q = bx[s[0]], bx[s[1]]
        n = np.array([q[1] - p[1], -(q[0] - p[0])])
        n /= np.linalg.norm(n)
        # ensure outward normal
        c = bx.mean(axis=0)
        if np.dot(n, c - p) > 0:
            n = -n
        # constraint: n·x <= n·p
        # for radius r: n·x + r ||n||  <= n·p  → here ||n||=1
        A_ub.append([n[0], n[1], 1.0])
        b_ub.append(np.dot(n, p))
    res = linprog(c=[0, 0, -1], A_ub=A_ub, b_ub=b_ub, bounds=[(None, None)] * 3, method="highs")
    cx, cy, rr = res.x
    th = np.linspace(0, 2 * np.pi, 100)
    ax.plot(cx + rr * np.cos(th), cy + rr * np.sin(th), "-", color="#27ae60", lw=1.8,
            label=fr"inscribed $\ell_2$-ball, $w^\dagger=({cx:.2f},{cy:.2f}),\,r^\dagger={rr:.2f}$")
    ax.plot(cx, cy, "o", color="#27ae60", markersize=5)
    # Box
    ax.plot([1, 10, 10, 1, 1], [1, 1, 10, 10, 1], "-", color="#7f8c8d", lw=1.0, alpha=0.7,
            label=r"weight box $[1,10]^2$")
    ax.set_xlim(0.5, 10.5); ax.set_ylim(0.5, 10.5); ax.set_aspect("equal")
    ax.set_xlabel(r"$w_1$"); ax.set_ylabel(r"$w_2$")
    ax.set_title(r"Equivalence region $\Omega(p^\star)$ and Chebyshev-centre $w^\dagger$",
                 loc="left", fontweight="bold")
    ax.legend(loc="upper right", frameon=False)
    ax.grid(True, color="#eee", lw=0.4)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.savefig(out / "T2_equivalence_cone.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_t3_l1_vs_l2_contrast(out: Path) -> None:
    """T3 — L1 (polyhedral) vs L2 (smooth) upper image contrast."""
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.42), constrained_layout=True)
    # L1: polyhedral
    pts1 = np.array([[0, 0], [0, 3], [1.5, 3], [3, 1.5], [3, 0]])
    axes[0].fill(pts1[:, 0], pts1[:, 1], facecolor="#e9eff8", edgecolor="#2980b9", lw=1.8, alpha=0.9)
    axes[0].plot(0, 0, "o", color="#c0392b", markersize=5, zorder=5,
                 label=r"$p^\star$ (sharp vertex)")
    axes[0].set_xlim(-0.4, 3.4); axes[0].set_ylim(-0.4, 3.4); axes[0].set_aspect("equal")
    axes[0].set_xlabel(r"$V_1$"); axes[0].set_ylabel(r"$V_2$")
    axes[0].set_title(r"$L_1$: polyhedral upper image", loc="left", fontweight="bold")
    axes[0].legend(loc="upper right", frameon=False)
    axes[0].grid(True, color="#eee", lw=0.4)
    # L2: smooth (quadratic boundary)
    th = np.linspace(np.pi, 1.5 * np.pi, 100)
    cx, cy, r = 3, 3, 3
    xs = cx + r * np.cos(th); ys = cy + r * np.sin(th)
    poly = np.column_stack([np.concatenate([[0], xs, [3]]),
                            np.concatenate([[3], ys, [0]])])
    axes[1].fill(poly[:, 0], poly[:, 1], facecolor="#f6e8e8", edgecolor="#c0392b",
                 lw=1.8, alpha=0.9)
    axes[1].plot(0, 0, "o", color="#c0392b", markersize=5, zorder=5,
                 label=r"$p^\star$ (smooth boundary point)")
    axes[1].set_xlim(-0.4, 3.4); axes[1].set_ylim(-0.4, 3.4); axes[1].set_aspect("equal")
    axes[1].set_xlabel(r"$V_1$"); axes[1].set_ylabel(r"$V_2$")
    axes[1].set_title(r"$L_2$: smooth upper image", loc="left", fontweight="bold")
    axes[1].legend(loc="upper right", frameon=False)
    axes[1].grid(True, color="#eee", lw=0.4)
    for ax in axes:
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.suptitle(r"$L_1$ vs $L_2$: equivalence-region geometry differs at $p^\star$",
                 fontsize=11, fontweight="bold")
    fig.savefig(out / "T3_l1_vs_l2_contrast.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_c2_example1_chebyshev(out: Path) -> None:
    """C2 — Example 1 (paper §11.1) Chebyshev centre in weight space."""
    fig, ax = plt.subplots(figsize=(3.50, 3.27), constrained_layout=True)
    # Box [1,10]^2, Chebyshev centre (5.5, 5.5), radius 4.5
    ax.plot([1, 10, 10, 1, 1], [1, 1, 10, 10, 1], "-", color="#2c3e50", lw=1.0,
            label=r"$\Omega(p^\star)\cap[1,10]^2$")
    ax.fill_between([1, 10], 1, 10, facecolor="#eef5fd", alpha=0.85)
    cx, cy, rr = 5.5, 5.5, 4.5
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(cx + rr * np.cos(th), cy + rr * np.sin(th), "-", color="#27ae60", lw=1.0,
            label=fr"inscribed ball  $r^\dagger={rr}$")
    ax.plot(cx, cy, "o", color="#c0392b", markersize=5,
            label=fr"$w^\dagger=({cx},{cy})$")
    ax.set_xlim(0, 11); ax.set_ylim(0, 11); ax.set_aspect("equal")
    ax.set_xlabel(r"$w_1$"); ax.set_ylabel(r"$w_2$")
    ax.set_title(r"Example 1 (paper §11.1): Chebyshev centre on $[1,10]^2$",
                 loc="left", fontweight="bold")
    ax.legend(loc="lower right", frameon=False)
    ax.grid(True, color="#eee", lw=0.4)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.savefig(out / "C2_example1_chebyshev.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_c3_example2_decay(out: Path) -> None:
    """C3 — Example 2 (paper §11.2) asymptotic V_i decay (log-log)."""
    w = np.logspace(0, 4, 50)
    V_l2 = 2.25 / (w / 10.0) ** 2
    V_l1 = 2.25 / (w / 10.0)
    fig, ax = plt.subplots(figsize=(3.50, 2.47), constrained_layout=True)
    ax.loglog(w, V_l2, lw=1.0, color="#c0392b", label=r"$V_i \sim O(1/w_i^2)$  (Theorem 6.1, $L_2$)")
    ax.loglog(w, V_l1, lw=1.0, color="#2980b9", linestyle="--",
              label=r"$V_i \sim O(1/w_i)$  (raw rate, $L_2$)")
    ax.set_xlabel(r"weight $w_i$"); ax.set_ylabel(r"$V_i^\star(w_i)$")
    ax.set_title(r"Example 2 (paper §11.2): asymptotic decay of $V_i$ with $w_i$",
                 loc="left", fontweight="bold")
    ax.grid(True, which="both", color="#eee", lw=0.4)
    ax.legend(loc="upper right", frameon=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.savefig(out / "C3_example2_Vi_decay.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_c4_example2_threshold(out: Path) -> None:
    """C4 — Example 2 threshold curve W_1(ε_1) at fixed w_3."""
    eps = np.logspace(-4, 0, 100)
    W1 = 7.75 / np.sqrt(eps)
    fig, ax = plt.subplots(figsize=(3.50, 2.47), constrained_layout=True)
    ax.loglog(eps, W1, lw=1.0, color="#c0392b")
    ax.scatter([0.01], [77.5], s=110, color="#27ae60", zorder=5,
               label=r"closed-form check  $W_1(0.01,1)\approx 77.5$")
    ax.axhline(77.5, color="#27ae60", linestyle=":", lw=1.0, alpha=0.5)
    ax.axvline(0.01, color="#27ae60", linestyle=":", lw=1.0, alpha=0.5)
    ax.set_xlabel(r"$\epsilon_1$"); ax.set_ylabel(r"$W_1(\epsilon_1,\, w_3{=}1)$")
    ax.set_title(r"Example 2: threshold scales as $W_1\sim 1/\sqrt{\epsilon_1}$",
                 loc="left", fontweight="bold")
    ax.grid(True, which="both", color="#eee", lw=0.4)
    ax.legend(loc="upper right", frameon=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.savefig(out / "C4_example2_threshold_W1.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _bbox(ax, xy, w, h, text, fc="#fbfaf6", ec="#7f8c8d", fontsize=10, fontweight="normal"):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.01,rounding_size=0.02",
                                facecolor=fc, edgecolor=ec, lw=1.0))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center",
            fontsize=fontsize, fontweight=fontweight)


def _arrow(ax, p, q, color="#34495e"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="->", mutation_scale=14,
                                 lw=1.2, color=color))


def fig_b1_algorithm_1a_flow(out: Path) -> None:
    """B1 — Algorithm 1A flowchart (block diagram)."""
    fig, ax = plt.subplots(figsize=(7.16, 3.58), constrained_layout=True)
    ax.set_xlim(0, 13); ax.set_ylim(0, 6); ax.set_axis_off()
    blocks = [
        ((0.3, 4.0), 2.2, 1.3, "Lex cascade\n$\\to z_{\\text{lex}}^\\star,\\ p^\\star$", "#e9eff8"),
        ((2.8, 4.0), 2.2, 1.3, "Gradient eval\n$\\nabla J,\\,\\nabla g_{i,j}$\nat $z_{\\text{lex}}^\\star$", "#fef9e7"),
        ((5.3, 4.0), 2.2, 1.3, "Lex KKT system\n(stationarity, complementarity)", "#fff0e0"),
        ((7.8, 4.0), 2.2, 1.3, "Fourier–Motzkin\nelimination of\nlex multipliers", "#fde9e9"),
        ((10.3, 4.0), 2.4, 1.3, "Box-bounded\nChebyshev LP\n$\\to (w^\\dagger, r^\\dagger)$", "#e9f7e9"),
        ((4.4, 1.5), 4.2, 1.3, "Compliance check at runtime:\n$b(z_{\\text{ws}}(w^\\dagger))\\;=\\;b(z_{\\text{lex}}^\\star)$  ?", "#fbfaf6"),
    ]
    for (xy, w, h, txt, fc) in blocks:
        _bbox(ax, xy, w, h, txt, fc=fc, fontsize=10, fontweight="bold")
    for src, dst in [
        ((2.5, 4.65), (2.8, 4.65)), ((5.0, 4.65), (5.3, 4.65)),
        ((7.5, 4.65), (7.8, 4.65)), ((10.0, 4.65), (10.3, 4.65)),
        ((11.5, 4.0), (8.4, 2.8)),
    ]:
        _arrow(ax, src, dst)
    ax.text(0.5, 5.6, "Offline: one-time calibration per scenario class",
            fontsize=10, fontweight="bold", color="#27ae60")
    ax.text(0.5, 0.5, "Online: per-tick WS solve at $w^\\dagger$ + runtime check.\n"
                       "Mismatch  $\\Rightarrow$ fallback to cascade or recalibrate.",
            fontsize=10, color="#34495e")
    fig.suptitle("Algorithm 1A ($L_1$ exact equivalence) — data flow",
                 fontsize=11, fontweight="bold")
    fig.savefig(out / "B1_algorithm_1a_flow.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_b2_algorithm_1b_flow(out: Path) -> None:
    """B2 — Algorithm 1B flowchart (block diagram)."""
    fig, ax = plt.subplots(figsize=(7.16, 3.58), constrained_layout=True)
    ax.set_xlim(0, 13); ax.set_ylim(0, 6); ax.set_axis_off()
    blocks = [
        ((0.3, 4.0), 2.2, 1.3, "Lex cascade\n$\\to z_{\\text{lex}}^\\star,\\ p^\\star$", "#e9eff8"),
        ((2.8, 4.0), 2.4, 1.3, "Reduced-KKT\nsensitivity\n$\\to (c_i, \\kappa_i)$", "#fef9e7"),
        ((5.4, 4.0), 2.4, 1.3, "Pointwise threshold\n$W_i(\\epsilon_i, w_{-i})$", "#fff0e0"),
        ((7.9, 4.0), 2.4, 1.3, "Coupled-linear\nChebyshev LP\n$\\to (w^\\dagger, r^\\dagger)$", "#fde9e9"),
        ((10.4, 4.0), 2.3, 1.3, "WS verify:\n$b_{\\epsilon}(z_{ws}(w^\\dagger))$\n$=b_{\\epsilon}(z_{\\text{lex}}^\\star)$", "#e9f7e9"),
        ((4.4, 1.5), 4.2, 1.3, "Runtime: WS solve + tolerance compliance check",
         "#fbfaf6"),
    ]
    for (xy, w, h, txt, fc) in blocks:
        _bbox(ax, xy, w, h, txt, fc=fc, fontsize=10, fontweight="bold")
    for src, dst in [
        ((2.5, 4.65), (2.8, 4.65)), ((5.2, 4.65), (5.4, 4.65)),
        ((7.8, 4.65), (7.9, 4.65)), ((10.3, 4.65), (10.4, 4.65)),
        ((11.5, 4.0), (8.4, 2.8)),
    ]:
        _arrow(ax, src, dst)
    ax.text(0.5, 5.6, "Offline: per-class operator-supplied tolerance $\\epsilon$",
            fontsize=10, fontweight="bold", color="#27ae60")
    fig.suptitle("Algorithm 1B ($L_2$ tolerance compliance) — data flow",
                 fontsize=11, fontweight="bold")
    fig.savefig(out / "B2_algorithm_1b_flow.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_b4_operational_state_machine(out: Path) -> None:
    """B4 — Offline/online operational state machine."""
    fig, ax = plt.subplots(figsize=(7.16, 3.85), constrained_layout=True)
    ax.set_xlim(0, 13); ax.set_ylim(0, 7); ax.set_axis_off()
    # Offline track
    _bbox(ax, (0.3, 5.0), 3.0, 1.3, "Encounter new\nscenario class", fc="#e9eff8",
          fontweight="bold")
    _bbox(ax, (3.7, 5.0), 3.0, 1.3, "Lex cascade\n+ Algorithm 1A/1B", fc="#fff0e0",
          fontweight="bold")
    _bbox(ax, (7.1, 5.0), 3.0, 1.3, "Cache  $w^\\dagger$  by\nscenario-class key", fc="#e9f7e9",
          fontweight="bold")
    # Online track
    _bbox(ax, (0.3, 2.5), 3.0, 1.3, "Per-tick:\nPlannerInput", fc="#fbfaf6", fontweight="bold")
    _bbox(ax, (3.7, 2.5), 3.0, 1.3, "Lookup  $w^\\dagger$\nfrom cache", fc="#fbfaf6", fontweight="bold")
    _bbox(ax, (7.1, 2.5), 3.0, 1.3, "WS solve at $w^\\dagger$", fc="#fbfaf6", fontweight="bold")
    _bbox(ax, (10.5, 2.5), 2.3, 1.3, "Output\ntrajectory", fc="#e9f7e9", fontweight="bold")
    # Branching
    _bbox(ax, (3.7, 0.3), 6.5, 1.3,
          "Compliance check  $b(z_{ws})=b(z_{\\text{lex}}^\\star)$ ?\n"
          "  match: accept    mismatch: log / fall back to cascade",
          fc="#fde9e9", fontweight="bold")
    for src, dst in [
        ((3.3, 5.65), (3.7, 5.65)), ((6.7, 5.65), (7.1, 5.65)),
        ((3.3, 3.15), (3.7, 3.15)), ((6.7, 3.15), (7.1, 3.15)),
        ((10.1, 3.15), (10.5, 3.15)), ((8.6, 2.5), (8.6, 1.6)),
        ((7.1, 5.0), (5.0, 1.6)),
    ]:
        _arrow(ax, src, dst)
    ax.text(0.3, 6.6, "Offline (per scenario class)", fontsize=11, fontweight="bold",
            color="#27ae60")
    ax.text(0.3, 4.1, "Online (per 0.1 s tick)", fontsize=11, fontweight="bold", color="#c0392b")
    fig.suptitle("LCP operational state machine", fontsize=11, fontweight="bold")
    fig.savefig(out / "B4_operational_state_machine.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_d1_two_level_pipeline(out: Path) -> None:
    """D1 — Two-level MPC pipeline diagram with SLP inner loop."""
    fig, ax = plt.subplots(figsize=(7.16, 3.32), constrained_layout=True)
    ax.set_xlim(0, 14); ax.set_ylim(0, 6); ax.set_axis_off()
    _bbox(ax, (0.3, 3.5), 2.4, 1.4, "Global planner\n(BFS over lane graph)", fc="#e9eff8")
    _bbox(ax, (3.1, 3.5), 2.4, 1.4, "Reference path\n(arc-length)", fc="#e9eff8")
    _bbox(ax, (5.9, 3.5), 2.4, 1.4, "Rule encoder\n(per-tick, 16 rules)", fc="#fff0e0")
    _bbox(ax, (8.7, 3.5), 3.0, 1.4, "LCP MPC\n4 levels × epigraph slacks", fc="#fde9e9",
          fontweight="bold")
    _bbox(ax, (12.1, 3.5), 1.6, 1.4, "IPOPT\nsolve", fc="#fbfaf6")
    _bbox(ax, (5.5, 1.0), 4.5, 1.3, "SLP outer loop:\nlinearise → solve → re-warm-start",
          fc="#e9f7e9", fontweight="bold")
    _bbox(ax, (10.5, 1.0), 3.2, 1.3, "Trajectory →\nsimulator (10 Hz)", fc="#e9eff8")
    for src, dst in [
        ((2.7, 4.2), (3.1, 4.2)), ((5.5, 4.2), (5.9, 4.2)),
        ((8.3, 4.2), (8.7, 4.2)), ((11.7, 4.2), (12.1, 4.2)),
        ((12.9, 3.5), (12.9, 2.3)), ((10.2, 1.65), (10.5, 1.65)),
        ((7.75, 2.3), (9.5, 3.5)),
    ]:
        _arrow(ax, src, dst)
    fig.suptitle("Two-level LCP MPC pipeline (with SLP outer loop)",
                 fontsize=11, fontweight="bold")
    fig.savefig(out / "D1_two_level_pipeline.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_d2_rule_level_mapping(out: Path) -> None:
    """D2 — Rule-to-level mapping table."""
    levels = {
        "L1 — Safety": ["9r0  vehicle collision", "10r0  VRU collision",
                        "7r0  lane corridor", "7r5  sidewalk drive",
                        "(10r5 bike lane: stub)"],
        "L2 — Legal": ["3r0  speed limit", "7r1  traffic light",
                       "7r2  opposing lane", "7r3  one-way",
                       "(7r4 stop in crosswalk: stub)"],
        "L3 — Comfort": ["3r3  safe headway", "1r11  lat. acceleration",
                         "0r2  long. comfort", "3r5  lat. clearance",
                         "0r3  lat. comfort (proxy)",
                         "(3r6 lane intrusion: stub)"],
        "L4 — Efficiency": ["weight_pos  position tracking",
                            "weight_heading  heading tracking",
                            "weight_speed  speed tracking",
                            "weight_control(_rate)  smoothness"],
    }
    fig, ax = plt.subplots(figsize=(7.16, 4.41), constrained_layout=True)
    ax.set_xlim(0, 12); ax.set_ylim(0, 10); ax.set_axis_off()
    colors = {"L1 — Safety": "#fde9e9", "L2 — Legal": "#fff0e0",
              "L3 — Comfort": "#e9f7e9", "L4 — Efficiency": "#e9eff8"}
    y_top = 9.5
    for name, rules in levels.items():
        ax.add_patch(FancyBboxPatch((0.3, y_top - 2.0), 11.4, 1.9,
                                     boxstyle="round,pad=0.01,rounding_size=0.02",
                                     facecolor=colors[name], edgecolor="#7f8c8d", lw=1.0))
        ax.text(0.6, y_top - 0.3, name, fontsize=11, fontweight="bold")
        for i, r in enumerate(rules):
            x = 0.6 + (i % 3) * 3.7
            y = y_top - 0.8 - (i // 3) * 0.5
            ax.text(x, y, "•  " + r, fontsize=10.5, family="DejaVu Sans Mono")
        y_top -= 2.3
    fig.suptitle("Rule-to-level mapping (16 promoted rules + 4 stubs)",
                 fontsize=11, fontweight="bold")
    fig.savefig(out / "D2_rule_level_mapping.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_g1_rulebook_hierarchy(out: Path) -> None:
    """G1 — 25-rule rulebook hierarchy tree."""
    fig, ax = plt.subplots(figsize=(7.16, 4.30), constrained_layout=True)
    ax.set_xlim(0, 15); ax.set_ylim(0, 10); ax.set_axis_off()
    levels = [
        (10, ["10r0", "10r3", "10r4", "10r5"], "#7f1d1d"),
        (9, ["9r0", "9r1"], "#c0392b"),
        (8, ["8r0", "8r1"], "#e67e22"),
        (7, ["7r0", "7r1", "7r2", "7r3", "7r4", "7r5"], "#f1c40f"),
        (3, ["3r0", "3r3", "3r5", "3r6"], "#16a085"),
        (2, ["2r2"], "#2980b9"),
        (1, ["1r0", "1r2", "1r5", "1r11"], "#8e44ad"),
        (0, ["0r2", "0r3"], "#7f8c8d"),
    ]
    y_pos = 9.3
    for lv, rules, c in levels:
        ax.add_patch(FancyBboxPatch((0.2, y_pos - 0.7), 1.4, 0.55,
                                     boxstyle="round,pad=0.01,rounding_size=0.02",
                                     facecolor=c, edgecolor=c, lw=1.0, alpha=0.7))
        ax.text(0.9, y_pos - 0.42, f"L{lv}", ha="center", va="center", color="white",
                fontsize=11, fontweight="bold")
        x = 2.0
        for r in rules:
            ax.add_patch(FancyBboxPatch((x, y_pos - 0.7), 1.4, 0.55,
                                         boxstyle="round,pad=0.005,rounding_size=0.02",
                                         facecolor="white", edgecolor=c, lw=1.4))
            ax.text(x + 0.7, y_pos - 0.42, r, ha="center", va="center", fontsize=10)
            x += 1.55
        y_pos -= 1.0
    ax.text(0.2, 9.7, "Lexicographic priority (10 = highest)",
            fontsize=10, fontweight="bold", color="#34495e")
    fig.suptitle("25-rule rulebook hierarchy by lexicographic level", fontsize=11, fontweight="bold")
    fig.savefig(out / "G1_rulebook_hierarchy.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------


SCENARIO_LABELS = [
    "01_following_slow_lead", "02_near_long_vehicle", "03_near_multiple_vehicles",
    "04_changing_lane", "05_changing_lane_left", "06_changing_lane_right",
    "07_starting_left_turn", "08_starting_right_turn", "09_high_speed_turn",
    "10_low_speed_turn", "11_protected_cross", "12_unprotected_cross",
    "13_high_magnitude_speed", "14_medium_magnitude_speed",
    "15_near_high_speed_vehicle", "16_traversing_intersection",
]


def main() -> None:
    # Load all 16
    logs: List[ScenarioLog] = []
    for lab in SCENARIO_LABELS:
        try:
            log = load_scenario(lab)
        except StopIteration:
            print(f"  skip {lab} (no log found)")
            continue
        logs.append(log)
        print(f"  loaded {lab}: {log.n_ticks} ticks, "
              f"{int((log.integrated_violation() > 0).sum())} violating rules")

    # Per-scenario figures
    for log in logs:
        scen_dir = PER / log.label
        scen_dir.mkdir(parents=True, exist_ok=True)
        print(f"  -> per_scenario/{log.label}/")
        fig_p1_top_rules_timeline(log, scen_dir)
        fig_p2_applicability_heatmap(log, scen_dir)
        fig_p3_violation_heatmap(log, scen_dir)
        fig_p4_cumulative_violation(log, scen_dir)
        fig_p5_violation_event_timeline(log, scen_dir)
        fig_p6_priority_level_stacks(log, scen_dir)

    # Aggregate
    print(f"  -> aggregate/")
    fig_a1_top_violation_per_scenario(logs, AGG)
    fig_a2_per_rule_aggregate(logs, AGG)
    fig_a3_rule_x_scenario_heatmap(logs, AGG)
    fig_a4_violation_count_heatmap(logs, AGG)
    fig_a5_priority_level_breakdown(logs, AGG)
    fig_a6_applicability_frequency(logs, AGG)
    fig_a7_rule_cooccurrence(logs, AGG)
    fig_a8_violation_duration_distribution(logs, AGG)
    fig_a9_aggregate_dashboard(logs, AGG)
    fig_a10_scenario_similarity(logs, AGG)

    # Theory / algorithm / worked examples / architecture
    print(f"  -> theory/")
    fig_t1_upper_image_l1(THEORY)
    fig_t2_equivalence_cone(THEORY)
    fig_t3_l1_vs_l2_contrast(THEORY)
    fig_c2_example1_chebyshev(THEORY)
    fig_c3_example2_decay(THEORY)
    fig_c4_example2_threshold(THEORY)
    fig_b1_algorithm_1a_flow(THEORY)
    fig_b2_algorithm_1b_flow(THEORY)
    fig_b4_operational_state_machine(THEORY)
    fig_d1_two_level_pipeline(THEORY)
    fig_d2_rule_level_mapping(THEORY)
    fig_g1_rulebook_hierarchy(THEORY)

    # Index
    write_index(logs)
    print(f"\nAll artifacts under {OUT}")


def write_index(logs: List[ScenarioLog]) -> None:
    lines = [
        "# Generated artifacts",
        "",
        "Auto-generated by `scripts/generate_artifacts.py`. Three directories:",
        "",
        "## `per_scenario/<label>/` — 6 figures per scenario",
        "",
        "| File | Description |",
        "|---|---|",
        "| `P1_top_rules_timeline.png` | Top-6 rules' violation-rate timeline |",
        "| `P2_applicability_heatmap.png` | Per-rule applicability mask (binary, rule × tick) |",
        "| `P3_violation_heatmap.png` | Per-rule violation-rate heatmap (rule × tick) |",
        "| `P4_cumulative_violation.png` | Cumulative violation over time for top-6 rules |",
        "| `P5_violation_events.png` | Scatter of violation events (size ∝ rate, colour by level) |",
        "| `P6_priority_level_stacks.png` | Stacked total rate per priority level vs time |",
        "",
        "Scenarios:",
        "",
    ]
    for log in logs:
        lines.append(f"- `per_scenario/{log.label}/`")
    lines += [
        "",
        "## `aggregate/` — 10 cross-scenario figures",
        "",
        "| File | Description |",
        "|---|---|",
        "| `A1_top_violation_per_scenario.png` | Sorted bar: top-rule integrated violation per scenario |",
        "| `A2_per_rule_aggregate.png` | Aggregate integrated violation per rule across all 16 |",
        "| `A3_rule_x_scenario_heatmap.png` | Rule × scenario integrated-violation heatmap |",
        "| `A4_violation_count_heatmap.png` | Rule × scenario violation-tick-count heatmap |",
        "| `A5_priority_level_breakdown.png` | Per-scenario stacked violation by priority level |",
        "| `A6_applicability_frequency.png` | Per-rule applicability and violation % across benchmark |",
        "| `A7_rule_cooccurrence.png` | Rule × rule co-violation matrix |",
        "| `A8_violation_duration_distribution.png` | Burst-duration boxplot per rule |",
        "| `A9_aggregate_dashboard.png` | 4-panel benchmark dashboard |",
        "| `A10_scenario_similarity.png` | Scenario × scenario violation-profile cosine similarity |",
        "",
        "## `theory/` — 12 scenario-independent conceptual figures",
        "",
        "| File | Paper § | Description |",
        "|---|---|---|",
        "| `T1_upper_image_l1.png` | §3 | 2D upper-image polygon with lex vertex |",
        "| `T2_equivalence_cone.png` | §4 | Equivalence region + Chebyshev centre in weight space |",
        "| `T3_l1_vs_l2_contrast.png` | §5 vs §6 | Polyhedral vs smooth upper-image contrast |",
        "| `C2_example1_chebyshev.png` | §11.1 | Paper Example 1: w† = (5.5, 5.5), r† = 4.5 |",
        "| `C3_example2_Vi_decay.png` | §11.2 | Asymptotic V_i decay log-log plot |",
        "| `C4_example2_threshold_W1.png` | §11.2 | W_1(ε_1) ~ 1/√ε_1 threshold curve |",
        "| `B1_algorithm_1a_flow.png` | §9.3 | Algorithm 1A block-diagram flow |",
        "| `B2_algorithm_1b_flow.png` | §9.4 | Algorithm 1B block-diagram flow |",
        "| `B4_operational_state_machine.png` | §9.6, §12 | Offline+online operational state machine |",
        "| `D1_two_level_pipeline.png` | §12, §14.2 | Two-level MPC pipeline with SLP outer loop |",
        "| `D2_rule_level_mapping.png` | §12.1, §14.1 | 16 rules mapped to 4 priority levels |",
        "| `G1_rulebook_hierarchy.png` | §12.1 | 25-rule rulebook hierarchy by level |",
        "",
        "## Counts",
        "",
        f"- Per-scenario: {len(logs)} scenarios × 6 figures = {len(logs) * 6}",
        "- Aggregate: 10",
        "- Theory: 12",
        f"- **Total: {len(logs) * 6 + 10 + 12}**",
    ]
    (OUT / "README.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
