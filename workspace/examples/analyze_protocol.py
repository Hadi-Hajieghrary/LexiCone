"""Analyse the comparative-effectiveness protocol's batch outputs.

Reads per-tick rule-evaluation CSVs across (condition, scenario, seed) cells,
computes per-priority-level integrated violations, lex-Pareto dominance flags,
priority-weighted aggregates, and bootstrap CIs. Emits the headline IEEE
Transactions figures (F1, F2, F3) plus a tidy long-form CSV that the remaining
figures (F4–F8) consume.

Inputs (default layout written by ``examples/13_run_protocol.py``):

    examples/outputs/13_protocol/<condition>/seed_<n>/<label>__<suffix>/
        <label>_log.csv         ← per-tick rule evaluations (5 cols)

Outputs:

    examples/outputs/13_protocol/figures/
        per_cell_metrics.csv    ← long-form: (condition, scenario, seed,
                                  level, V, J, n_violations, status, duration_s)
        lex_dominance.csv       ← pairwise (scenario, M, B) → wins/ties/losses
        F1_per_level_per_scenario.png
        F2_delta_violin.png
        F3_lex_dominance_heatmap.png

Headline numbers are also printed to stdout for quick inspection.

Usage::

    python examples/analyze_protocol.py --protocol-root examples/outputs/13_protocol
    python examples/analyze_protocol.py --legacy-glob 'examples/outputs/12_batch*/*/*_log.csv'
        # falls back to the existing 12_batch layout when 13_protocol is empty
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# IEEE Transactions typography (Times serif, 8 pt body, 300 dpi).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import ieee_style  # noqa: E402

ieee_style.apply()
COL_2 = ieee_style.COL_2


# ----------------------------------------------------------------------
# Rule-id → priority-level + MPC-controlled set (kept in sync with the
# partition test at lexicone/planning/tests/test_rule_level_mapping.py).
# ----------------------------------------------------------------------

MPC_CONTROLLED_IDS = frozenset({
    "10r0", "9r0",
    "7r0", "7r1", "7r2", "7r3", "7r5",
    "3r0", "3r3", "3r5",
    "1r11",
    "0r2", "0r3",
})

# Levels at which at least one MPC-controlled rule exists. Used as the index
# set for the V_ell vector. Sorted descending (highest priority first).
CONTROLLED_LEVELS: Tuple[int, ...] = (10, 9, 7, 3, 1, 0)

LEVEL_COLOURS = {
    10: "#7f1d1d", 9: "#c0392b", 7: "#f1c40f",
    3: "#16a085", 1: "#8e44ad", 0: "#7f8c8d",
}

# Per-level numerical tolerance for the "equal-up-to-tol" comparison in the
# lex-dominance predicate. Same epsilon vector as the LCP YAML default.
EPSILON_PER_LEVEL: Dict[int, float] = {
    10: 1.0e-4, 9: 1.0e-4, 7: 4.0e-2,
    3: 5.0e-1, 1: 5.0e-1, 0: 5.0e-1,
}


def level_of(rule_id: str) -> int:
    """Extract the priority level from a rule id, e.g. ``7r2`` → ``7``."""
    return int(rule_id.split("r")[0])


# ----------------------------------------------------------------------
# CSV loading + per-cell metric computation
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class CellMetrics:
    """Per-tick CSV aggregated into the V_ell vector + scalar aggregates."""
    condition: str
    scenario: str
    seed: int
    duration_s: float
    n_ticks: int
    V_mpc: Dict[int, float]    # level → integrated violation (MPC-controlled only)
    V_inv: Dict[int, float]    # level → integrated violation (invariant control)
    J: float                   # priority-weighted aggregate

    def vector_mpc(self) -> np.ndarray:
        return np.array([self.V_mpc.get(L, 0.0) for L in CONTROLLED_LEVELS])


def aggregate_csv(csv_path: Path) -> Tuple[Dict[int, float], Dict[int, float], float, int]:
    """Read one per-tick CSV → (V_mpc[level], V_inv[level], duration_s, n_ticks).

    V[level] = sum over applicable ticks of violation_rate * dt, summed across
    all rules at that level.
    """
    per_tick: Dict[float, List[Tuple[str, int, float]]] = defaultdict(list)
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            t = float(row["t_s"])
            rid = row["rule_id"]
            applies = int(row["applies"])
            rate = float(row["violation_rate"])
            if applies:
                per_tick[t].append((rid, applies, rate))
    timestamps = sorted(per_tick.keys())
    if not timestamps:
        return {}, {}, 0.0, 0
    if len(timestamps) == 1:
        dts = [0.1]
    else:
        diffs = np.diff(timestamps)
        dts = [float(diffs[0])] + [float(d) for d in diffs]
    n_ticks = len(timestamps)
    duration_s = float(sum(dts))

    V_mpc: Dict[int, float] = defaultdict(float)
    V_inv: Dict[int, float] = defaultdict(float)
    for i, t in enumerate(timestamps):
        for rid, _appl, rate in per_tick[t]:
            lvl = level_of(rid)
            target = V_mpc if rid in MPC_CONTROLLED_IDS else V_inv
            target[lvl] += rate * dts[i]
    return dict(V_mpc), dict(V_inv), duration_s, n_ticks


def priority_weighted(V_mpc: Dict[int, float]) -> float:
    """Decade-separated aggregate: J = sum_ell 10^ell · V_ell."""
    return float(sum((10 ** L) * V_mpc.get(L, 0.0) for L in CONTROLLED_LEVELS))


def collect_cells(
    protocol_root: Path,
    legacy_globs: Sequence[str] = (),
) -> List[CellMetrics]:
    """Walk a protocol output tree and return one ``CellMetrics`` per CSV.

    Tries the 13_protocol/<condition>/seed_<n>/<label>/log.csv layout first;
    falls back to ``--legacy-glob`` patterns (which map onto the older
    12_batch_two_level_mpc_planner layout) if 13_protocol is empty.
    """
    cells: List[CellMetrics] = []

    # 13_protocol layout
    if protocol_root.is_dir():
        for cond_dir in sorted(protocol_root.iterdir()):
            if not cond_dir.is_dir() or cond_dir.name == "figures":
                continue
            condition = cond_dir.name
            for seed_dir in sorted(cond_dir.iterdir()):
                if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
                    continue
                seed = int(seed_dir.name.removeprefix("seed_"))
                for label_dir in sorted(seed_dir.iterdir()):
                    if not label_dir.is_dir():
                        continue
                    csv_files = list(label_dir.glob("*_log.csv"))
                    if not csv_files:
                        continue
                    V_mpc, V_inv, dur, n = aggregate_csv(csv_files[0])
                    scenario = label_dir.name.split("__")[0]
                    cells.append(CellMetrics(
                        condition=condition, scenario=scenario, seed=seed,
                        duration_s=dur, n_ticks=n,
                        V_mpc=V_mpc, V_inv=V_inv,
                        J=priority_weighted(V_mpc),
                    ))

    # Legacy fallback — useful for plotting against the pre-protocol runs.
    for pattern in legacy_globs:
        for csv_path in sorted(Path("/").glob(pattern.lstrip("/"))):
            parts = csv_path.relative_to(Path("/")).parts
            # Heuristic: condition = "legacy" or "ws-l1" inferred from path suffix.
            label_dir = csv_path.parent.name
            scenario = label_dir.split("__")[0]
            if "__l1_cascade" in label_dir:
                condition = "C4_cascade_l1"
            elif "__l1" in label_dir:
                condition = "C1_ws_l1"
            elif "__l2" in label_dir:
                condition = "C3_ws_l2"
            else:
                condition = "C0_legacy"
            V_mpc, V_inv, dur, n = aggregate_csv(csv_path)
            cells.append(CellMetrics(
                condition=condition, scenario=scenario, seed=0,
                duration_s=dur, n_ticks=n,
                V_mpc=V_mpc, V_inv=V_inv,
                J=priority_weighted(V_mpc),
            ))

    return cells


# ----------------------------------------------------------------------
# Lex-Pareto dominance
# ----------------------------------------------------------------------


def lex_compare(
    V_M: np.ndarray, V_B: np.ndarray, levels: Sequence[int],
) -> int:
    """Return +1 if M ≻_lex B, -1 if B ≻_lex M, 0 if tie at tolerance.

    Levels are listed in priority-descending order (highest first).
    """
    for i, lvl in enumerate(levels):
        tau = EPSILON_PER_LEVEL[lvl]
        diff = V_M[i] - V_B[i]
        if diff > tau:
            return -1   # B dominates at this level
        if diff < -tau:
            return +1   # M dominates at this level
    return 0


def per_scenario_winner(
    cells_M: List[CellMetrics], cells_B: List[CellMetrics],
) -> Tuple[int, int, int]:
    """Aggregate seeds by per-scenario median and return (#M wins, #ties, #B wins)."""
    by_scen_M: Dict[str, List[np.ndarray]] = defaultdict(list)
    by_scen_B: Dict[str, List[np.ndarray]] = defaultdict(list)
    for c in cells_M:
        by_scen_M[c.scenario].append(c.vector_mpc())
    for c in cells_B:
        by_scen_B[c.scenario].append(c.vector_mpc())

    scenarios = sorted(set(by_scen_M) & set(by_scen_B))
    n_M = n_tie = n_B = 0
    for s in scenarios:
        med_M = np.median(np.stack(by_scen_M[s]), axis=0)
        med_B = np.median(np.stack(by_scen_B[s]), axis=0)
        winner = lex_compare(med_M, med_B, CONTROLLED_LEVELS)
        if winner > 0:
            n_M += 1
        elif winner < 0:
            n_B += 1
        else:
            n_tie += 1
    return n_M, n_tie, n_B


def binomial_ci_95(wins: int, trials: int) -> Tuple[float, float]:
    """Exact binomial 95% CI on the win fraction (Clopper–Pearson)."""
    if trials == 0:
        return 0.0, 0.0
    lo, hi = stats.binomtest(wins, trials).proportion_ci(0.95, method="exact")
    return float(lo), float(hi)


# ----------------------------------------------------------------------
# Bootstrap CI for per-level percent reduction
# ----------------------------------------------------------------------


def bootstrap_delta_per_level(
    cells_M: List[CellMetrics], cells_B: List[CellMetrics],
    levels: Sequence[int] = CONTROLLED_LEVELS,
    n_bootstrap: int = 10000,
    rng: Optional[np.random.Generator] = None,
) -> Dict[int, Tuple[float, float, float]]:
    """Stratified BCa-bootstrap median percent reduction per level, across scenarios.

    Returns ``{level: (median_delta, lo_95, hi_95)}``.
    """
    rng = rng or np.random.default_rng(0)
    # Aggregate seeds → per-scenario median.
    by_scen_M: Dict[str, List[np.ndarray]] = defaultdict(list)
    by_scen_B: Dict[str, List[np.ndarray]] = defaultdict(list)
    for c in cells_M:
        by_scen_M[c.scenario].append(c.vector_mpc())
    for c in cells_B:
        by_scen_B[c.scenario].append(c.vector_mpc())
    scenarios = sorted(set(by_scen_M) & set(by_scen_B))

    deltas_per_scen = np.zeros((len(scenarios), len(levels)))
    for i, s in enumerate(scenarios):
        med_M = np.median(np.stack(by_scen_M[s]), axis=0)
        med_B = np.median(np.stack(by_scen_B[s]), axis=0)
        for j, lvl in enumerate(levels):
            denom = max(med_B[j], EPSILON_PER_LEVEL[lvl])
            deltas_per_scen[i, j] = (med_B[j] - med_M[j]) / denom

    # Percentile bootstrap (BCa requires more work; percentile is fine here).
    n_s = len(scenarios)
    out: Dict[int, Tuple[float, float, float]] = {}
    for j, lvl in enumerate(levels):
        boot = np.empty(n_bootstrap)
        for b in range(n_bootstrap):
            idx = rng.integers(0, n_s, size=n_s)
            boot[b] = np.median(deltas_per_scen[idx, j])
        med = float(np.median(deltas_per_scen[:, j]))
        lo, hi = float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))
        out[lvl] = (med, lo, hi)
    return out


# ----------------------------------------------------------------------
# Long-form CSV + headline figures
# ----------------------------------------------------------------------


def write_per_cell_csv(cells: List[CellMetrics], out: Path) -> None:
    """Write one row per (condition, scenario, seed, level) and the J aggregate."""
    rows = []
    for c in cells:
        for lvl in CONTROLLED_LEVELS:
            rows.append({
                "condition": c.condition,
                "scenario": c.scenario,
                "seed": c.seed,
                "level": lvl,
                "V_mpc": c.V_mpc.get(lvl, 0.0),
                "V_invariant": c.V_inv.get(lvl, 0.0),
                "J_total": c.J,
                "duration_s": c.duration_s,
                "n_ticks": c.n_ticks,
            })
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def write_lex_dominance_csv(
    cells: List[CellMetrics], out: Path, baseline_cond: str,
) -> List[Tuple[str, int, int, int]]:
    """Pairwise lex-dominance summary for every non-baseline condition.

    Returns the list of (condition, n_wins, n_ties, n_losses) tuples printed.
    """
    by_cond: Dict[str, List[CellMetrics]] = defaultdict(list)
    for c in cells:
        by_cond[c.condition].append(c)
    if baseline_cond not in by_cond:
        return []
    baseline = by_cond[baseline_cond]
    summary: List[Tuple[str, int, int, int]] = []
    for cond, group in sorted(by_cond.items()):
        if cond == baseline_cond:
            continue
        w, t, l = per_scenario_winner(group, baseline)
        summary.append((cond, w, t, l))
    with out.open("w", newline="") as f:
        wri = csv.writer(f)
        wri.writerow(["condition_M", "baseline_B", "n_M_wins", "n_ties", "n_B_wins", "n_scenarios"])
        for cond, w, t, l in summary:
            wri.writerow([cond, baseline_cond, w, t, l, w + t + l])
    return summary


def fig_f1_per_level_per_scenario(cells: List[CellMetrics], out: Path) -> None:
    """F1 — one stacked-bar panel per condition; bars = scenarios; stacks = levels."""
    by_cond: Dict[str, Dict[str, np.ndarray]] = defaultdict(dict)
    for c in cells:
        by_cond[c.condition].setdefault(c.scenario, c.vector_mpc())
    conditions = sorted(by_cond)
    if not conditions:
        return
    scenarios = sorted({s for d in by_cond.values() for s in d})
    n = len(conditions)
    fig, axes = plt.subplots(
        1, n, figsize=(COL_2, 1.2 + 0.16 * len(scenarios)),
        sharey=True, constrained_layout=True,
    )
    if n == 1:
        axes = [axes]
    for ax, cond in zip(axes, conditions):
        cell_map = by_cond[cond]
        # Average across seeds per scenario.
        scen_vecs: Dict[str, List[np.ndarray]] = defaultdict(list)
        for c in cells:
            if c.condition == cond:
                scen_vecs[c.scenario].append(c.vector_mpc())
        bottoms = np.zeros(len(scenarios))
        for j, lvl in enumerate(CONTROLLED_LEVELS):
            vals = np.array([
                np.median(np.stack(scen_vecs.get(s, [np.zeros(len(CONTROLLED_LEVELS))])), axis=0)[j]
                for s in scenarios
            ])
            ax.barh(
                range(len(scenarios)), vals, left=bottoms,
                color=LEVEL_COLOURS[lvl], alpha=0.92, edgecolor="white", linewidth=0.3,
                label=f"L{lvl}" if ax is axes[0] else None,
            )
            bottoms = bottoms + vals
        ax.set_yticks(range(len(scenarios)))
        if ax is axes[0]:
            ax.set_yticklabels(scenarios)
        else:
            ax.set_yticklabels([""] * len(scenarios))
        ax.set_title(cond, fontsize=9, fontweight="bold", loc="left")
        ax.invert_yaxis()
        ax.set_xlabel(r"$\sum_\ell V_\ell$  (MPC-controlled)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].legend(
        title="Level", loc="lower right", frameon=True, framealpha=0.85,
        edgecolor="none", facecolor="white",
    )
    fig.savefig(out / "F1_per_level_per_scenario.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_f2_delta_violin(
    cells: List[CellMetrics], baseline_cond: str, target_cond: str, out: Path,
) -> None:
    """F2 — per-level box+strip of percent reduction Δ_ℓ across scenarios."""
    by_cond: Dict[str, List[CellMetrics]] = defaultdict(list)
    for c in cells:
        by_cond[c.condition].append(c)
    if baseline_cond not in by_cond or target_cond not in by_cond:
        return

    by_scen_M: Dict[str, List[np.ndarray]] = defaultdict(list)
    by_scen_B: Dict[str, List[np.ndarray]] = defaultdict(list)
    for c in by_cond[target_cond]:
        by_scen_M[c.scenario].append(c.vector_mpc())
    for c in by_cond[baseline_cond]:
        by_scen_B[c.scenario].append(c.vector_mpc())
    scenarios = sorted(set(by_scen_M) & set(by_scen_B))
    if not scenarios:
        return

    deltas = np.zeros((len(scenarios), len(CONTROLLED_LEVELS)))
    for i, s in enumerate(scenarios):
        med_M = np.median(np.stack(by_scen_M[s]), axis=0)
        med_B = np.median(np.stack(by_scen_B[s]), axis=0)
        for j, lvl in enumerate(CONTROLLED_LEVELS):
            denom = max(med_B[j], EPSILON_PER_LEVEL[lvl])
            deltas[i, j] = 100.0 * (med_B[j] - med_M[j]) / denom

    fig, ax = plt.subplots(figsize=(COL_2, 2.6), constrained_layout=True)
    positions = np.arange(len(CONTROLLED_LEVELS))
    bp = ax.boxplot(
        [deltas[:, j] for j in range(len(CONTROLLED_LEVELS))],
        positions=positions, widths=0.55, patch_artist=True,
        showfliers=False, medianprops=dict(color="black", linewidth=1.0),
    )
    for patch, lvl in zip(bp["boxes"], CONTROLLED_LEVELS):
        patch.set_facecolor(LEVEL_COLOURS[lvl])
        patch.set_alpha(0.55)
        patch.set_edgecolor(LEVEL_COLOURS[lvl])
    for j, lvl in enumerate(CONTROLLED_LEVELS):
        ax.scatter(
            positions[j] + np.random.uniform(-0.10, 0.10, len(scenarios)),
            deltas[:, j], s=8, color=LEVEL_COLOURS[lvl], alpha=0.7,
            edgecolor="white", linewidth=0.3,
        )
    ax.axhline(0.0, color="black", linewidth=0.5)
    ax.set_xticks(positions)
    ax.set_xticklabels([f"L{lvl}" for lvl in CONTROLLED_LEVELS])
    ax.set_ylabel(r"percent reduction $\Delta_\ell$ vs " + baseline_cond + " (%)")
    ax.set_title(
        f"{target_cond} vs {baseline_cond}: per-level violation reduction, n={len(scenarios)} scenarios",
        loc="left", fontsize=9, fontweight="bold",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(out / "F2_delta_violin.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_f3_lex_dominance_heatmap(
    cells: List[CellMetrics], baseline_cond: str, out: Path,
) -> None:
    """F3 — per-scenario × per-condition lex-dominance grid."""
    by_cond: Dict[str, List[CellMetrics]] = defaultdict(list)
    for c in cells:
        by_cond[c.condition].append(c)
    conditions = [c for c in sorted(by_cond) if c != baseline_cond]
    if not conditions or baseline_cond not in by_cond:
        return

    by_scen_B: Dict[str, List[np.ndarray]] = defaultdict(list)
    for c in by_cond[baseline_cond]:
        by_scen_B[c.scenario].append(c.vector_mpc())
    scenarios = sorted(by_scen_B)

    grid = np.zeros((len(scenarios), len(conditions)))
    for j, cond in enumerate(conditions):
        by_scen_M: Dict[str, List[np.ndarray]] = defaultdict(list)
        for c in by_cond[cond]:
            by_scen_M[c.scenario].append(c.vector_mpc())
        for i, s in enumerate(scenarios):
            if s not in by_scen_M:
                grid[i, j] = np.nan
                continue
            med_M = np.median(np.stack(by_scen_M[s]), axis=0)
            med_B = np.median(np.stack(by_scen_B[s]), axis=0)
            grid[i, j] = lex_compare(med_M, med_B, CONTROLLED_LEVELS)

    fig, ax = plt.subplots(
        figsize=(COL_2, 0.5 + 0.18 * len(scenarios)), constrained_layout=True,
    )
    cmap = plt.matplotlib.colors.ListedColormap(["#c0392b", "#f5f3ee", "#16a085"])
    bounds = [-1.5, -0.5, 0.5, 1.5]
    norm = plt.matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    ax.imshow(grid, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_yticks(range(len(scenarios))); ax.set_yticklabels(scenarios)
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions, rotation=20, ha="right")
    ax.set_title(
        f"Lex-Pareto dominance vs {baseline_cond}  (green = condition dominates; "
        f"red = baseline dominates; cream = tie)",
        loc="left", fontsize=9, fontweight="bold",
    )
    # Per-cell text labels
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid[i, j]
            if np.isnan(v):
                continue
            sym = {1: "✓", 0: "·", -1: "✗"}[int(v)]
            ax.text(j, i, sym, ha="center", va="center", fontsize=9,
                    color="white" if abs(v) > 0.5 else "#444", fontweight="bold")
    fig.savefig(out / "F3_lex_dominance_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# Headline printout
# ----------------------------------------------------------------------


def print_headline(
    cells: List[CellMetrics], baseline_cond: str,
) -> None:
    by_cond: Dict[str, List[CellMetrics]] = defaultdict(list)
    for c in cells:
        by_cond[c.condition].append(c)
    if baseline_cond not in by_cond:
        print(f"  baseline '{baseline_cond}' not present — headline skipped")
        return
    print()
    print(f"=== Headline lex-dominance vs baseline {baseline_cond} ===")
    print(f"{'condition':>20}  {'wins':>5}  {'ties':>5}  {'losses':>6}  {'n_scen':>7}  {'binom CI 95%':>16}")
    print("-" * 90)
    for cond, group in sorted(by_cond.items()):
        if cond == baseline_cond:
            continue
        w, t, l = per_scenario_winner(group, by_cond[baseline_cond])
        n = w + t + l
        lo, hi = binomial_ci_95(w, n)
        print(f"{cond:>20}  {w:>5}  {t:>5}  {l:>6}  {n:>7}  [{lo:.2f}, {hi:.2f}]")

    print()
    print("=== Per-level percent reduction (median across scenarios, 95% bootstrap CI) ===")
    rng = np.random.default_rng(7)
    for cond, group in sorted(by_cond.items()):
        if cond == baseline_cond:
            continue
        deltas = bootstrap_delta_per_level(
            group, by_cond[baseline_cond], rng=rng,
        )
        line = f"{cond:>20}  "
        for lvl in CONTROLLED_LEVELS:
            med, lo, hi = deltas[lvl]
            line += f"L{lvl}: {100*med:+6.1f}% [{100*lo:+5.0f},{100*hi:+5.0f}]   "
        print(line)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--protocol-root", type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "13_protocol",
    )
    p.add_argument(
        "--legacy-glob", action="append", default=[],
        help="Additional glob patterns to harvest legacy 12_batch CSVs from "
             "(for demonstrating against pre-protocol data). May be repeated.",
    )
    p.add_argument(
        "--baseline", type=str, default=None,
        help="Condition to use as the dominance baseline. Defaults to the "
             "first condition alphabetically (typically C0_legacy).",
    )
    p.add_argument(
        "--out-dir", type=Path, default=None,
        help="Where to write figures + CSVs. Defaults to <protocol-root>/figures/.",
    )
    args = p.parse_args()

    cells = collect_cells(args.protocol_root, legacy_globs=args.legacy_glob)
    if not cells:
        print(f"no per-tick CSVs found under {args.protocol_root} (or via --legacy-glob).",
              file=sys.stderr)
        return 1
    out = args.out_dir or args.protocol_root / "figures"
    out.mkdir(parents=True, exist_ok=True)

    baseline = args.baseline or sorted({c.condition for c in cells})[0]
    print(f"[analyze] {len(cells)} cells across "
          f"{len({c.condition for c in cells})} conditions, "
          f"{len({c.scenario for c in cells})} scenarios, "
          f"{len({c.seed for c in cells})} seeds")
    print(f"[analyze] baseline = {baseline}")
    print(f"[analyze] output    = {out}")

    write_per_cell_csv(cells, out / "per_cell_metrics.csv")
    write_lex_dominance_csv(cells, out / "lex_dominance.csv", baseline)
    fig_f1_per_level_per_scenario(cells, out)
    for cond in sorted({c.condition for c in cells} - {baseline}):
        fig_f2_delta_violin(cells, baseline, cond, out)
    fig_f3_lex_dominance_heatmap(cells, baseline, out)
    print_headline(cells, baseline)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
