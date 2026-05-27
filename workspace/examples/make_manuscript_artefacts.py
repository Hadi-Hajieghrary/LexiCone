#!/usr/bin/env python3
"""Generate manuscript figures + tables from completed C1 and C4 batch CSVs.

Produces in ``examples/outputs/manuscript/figures/``:

- ``fig10_top_violation.pdf``      — per-scenario top-rule integrated violation (C1)
- ``fig11_rule_scenario_heatmap.pdf`` — rule × scenario integrated-violation matrix (C1)
- ``fig12_level_breakdown.pdf``    — per-scenario stacked V_ell breakdown (C1)
- ``fig18_compliance_match.pdf``   — WS-vs-cascade compliance match per scenario
- ``fig22_walltime_bars.pdf``      — per-scenario wall-time C1 vs C4
- ``fig28_violation_timeseries.pdf`` — V_ell(t) overlay on scenario 01 (C1 vs C4)

And tables (LaTeX-fragments under ``examples/outputs/manuscript/tables/``):

- ``table_per_scenario_status.tex`` — per-scenario C1 status (Table 8)
- ``table_compliance.tex``           — aggregate compliance match-rate (Table 10)
- ``table_diagnostics.tex``          — per-scenario FM I/FM II pass (new)
- ``table_necessity.tex``            — per-scenario i*_nec, forced levels (new)
- ``table_walltime.tex``             — C1 vs C4 wall-time summary
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

_WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

try:
    from scripts.ieee_style import apply as _ieee_apply, COL_1, COL_2
    _ieee_apply()
except ImportError:
    COL_1, COL_2 = 3.5, 7.16   # inches — IEEEtran twocolumn defaults


# ----------------------------------------------------------------------
# Data loaders
# ----------------------------------------------------------------------

def _load_log_csv(path: Path) -> List[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def _rule_level(rule_id: str) -> int:
    """Parse the priority level from a rule_id like ``10r0`` or ``3r3``."""
    if "r" not in rule_id:
        return -1
    try:
        return int(rule_id.split("r", 1)[0])
    except ValueError:
        return -1


def _per_scenario_dirs(batch_root: Path) -> List[Path]:
    return sorted(p for p in batch_root.iterdir() if p.is_dir())


def _scenario_label(dir_path: Path) -> str:
    """Strip __l1 / __l1_cascade suffixes for clean labelling."""
    name = dir_path.name
    for suffix in ("__l1_cascade", "__l1"):
        if name.endswith(suffix):
            name = name[:-len(suffix)]
            break
    return name


def _per_tick_integrated_by_level(log_csv_rows: List[dict]) -> Dict[int, Dict[float, float]]:
    """Returns level -> {tick_t_s -> sum violation_rate} so we can plot V_ell(t)."""
    out: Dict[int, Dict[float, float]] = defaultdict(lambda: defaultdict(float))
    for row in log_csv_rows:
        lvl = _rule_level(row["rule_id"])
        if lvl < 0:
            continue
        if int(row["applies"]) != 1:
            continue
        t = float(row["t_s"])
        out[lvl][t] += float(row["violation_rate"])
    return {l: dict(out[l]) for l in sorted(out)}


# The 13 MPC-controlled rule IDs from §III.B of the manuscript.
# Source: lexicone.planning.rule_encoder.make_default_ruleset() — non-stub encoders.
MPC_CONTROLLED_IDS = frozenset({
    "10r0", "9r0", "7r0", "7r1", "7r2", "7r3", "7r5",
    "3r0", "3r3", "3r5", "1r11", "0r2", "0r3",
})


def _integrated_by_rule(log_csv_rows: List[dict],
                         mpc_only: bool = False) -> Dict[str, float]:
    """Returns rule_id -> sum violation_rate * dt (approximate ∫).

    When ``mpc_only=True``, filters to the 13 MPC-controlled rule IDs of
    Section III.B; otherwise includes all 25 observer-emitted rules.
    """
    out: Dict[str, float] = defaultdict(float)
    DT = 0.1
    for row in log_csv_rows:
        if int(row["applies"]) != 1:
            continue
        if mpc_only and row["rule_id"] not in MPC_CONTROLLED_IDS:
            continue
        out[row["rule_id"]] += float(row["violation_rate"]) * DT
    return dict(out)


def _integrated_by_level(rule_integrated: Dict[str, float]) -> Dict[int, float]:
    out: Dict[int, float] = defaultdict(float)
    for rid, v in rule_integrated.items():
        out[_rule_level(rid)] += v
    return dict(out)


def _load_batch_summary(batch_root: Path) -> List[dict]:
    summaries = list(batch_root.glob("batch_summary*.csv"))
    if not summaries:
        return []
    with summaries[0].open() as f:
        return list(csv.DictReader(f))


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------


def fig10_top_violation(c1_root: Path, out_path: Path) -> None:
    """Per-scenario top-*MPC*-rule integrated violation (single-column bar
    chart). Filters to the 13 MPC-controlled rule IDs since fig10 is the
    deployment claim — observer-only rules are not under planner control."""
    scenario_dirs = _per_scenario_dirs(c1_root)
    labels: List[str] = []
    integrated: List[float] = []
    top_rules: List[str] = []
    for d in scenario_dirs:
        csvs = list(d.glob("*_log.csv"))
        if not csvs:
            continue
        rule_int = _integrated_by_rule(_load_log_csv(csvs[0]), mpc_only=True)
        if not rule_int:
            continue
        top = max(rule_int, key=rule_int.get)
        labels.append(_scenario_label(d).replace("_", " "))
        integrated.append(rule_int[top])
        top_rules.append(top)
    if not labels:
        print(f"  [fig10] no per-tick logs in {c1_root}; skipping")
        return

    fig, ax = plt.subplots(figsize=(COL_2, 3.4))
    y = np.arange(len(labels))
    ax.barh(y, integrated, color="steelblue", edgecolor="black", linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    for yi, (val, rid) in enumerate(zip(integrated, top_rules)):
        ax.text(val, yi, f"  {rid}", va="center", fontsize=6)
    ax.set_xlabel("Top-rule integrated violation $\\int V \\, dt$")
    ax.invert_yaxis()
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig10] {out_path}")


def fig11_rule_scenario_heatmap(c1_root: Path, out_path: Path) -> None:
    """Rule × scenario integrated-violation heatmap (double-column)."""
    scenario_dirs = _per_scenario_dirs(c1_root)
    rule_set: set = set()
    per_scenario: Dict[str, Dict[str, float]] = {}
    for d in scenario_dirs:
        csvs = list(d.glob("*_log.csv"))
        if not csvs:
            continue
        rows = _load_log_csv(csvs[0])
        integrated = _integrated_by_rule(rows)
        # Keep only rules that violated anywhere (>0).
        integrated = {r: v for r, v in integrated.items() if v > 1e-9}
        per_scenario[_scenario_label(d)] = integrated
        rule_set.update(integrated.keys())
    if not per_scenario:
        print(f"  [fig11] no per-tick logs in {c1_root}; skipping")
        return

    rules = sorted(rule_set, key=lambda r: (_rule_level(r), r))
    scenarios = list(per_scenario.keys())
    mat = np.zeros((len(rules), len(scenarios)))
    for j, scn in enumerate(scenarios):
        for i, rid in enumerate(rules):
            mat[i, j] = per_scenario[scn].get(rid, 0.0)

    fig, ax = plt.subplots(figsize=(COL_2, 0.25 * len(rules) + 1.5))
    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_yticks(range(len(rules)))
    ax.set_yticklabels(rules, fontsize=6)
    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels([s.replace("_", " ") for s in scenarios], rotation=45, ha="right", fontsize=6)
    fig.colorbar(im, ax=ax, label="$\\int V_{rule} \\, dt$", fraction=0.025, pad=0.01)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig11] {out_path}")


def fig12_level_breakdown(c1_root: Path, out_path: Path) -> None:
    """Per-scenario stacked V_ell breakdown."""
    scenario_dirs = _per_scenario_dirs(c1_root)
    per_scenario: Dict[str, Dict[int, float]] = {}
    for d in scenario_dirs:
        csvs = list(d.glob("*_log.csv"))
        if not csvs:
            continue
        rows = _load_log_csv(csvs[0])
        per_scenario[_scenario_label(d)] = _integrated_by_level(_integrated_by_rule(rows))
    if not per_scenario:
        print(f"  [fig12] no logs; skipping")
        return

    all_levels = sorted({l for d in per_scenario.values() for l in d})
    cmap = plt.get_cmap("viridis", max(2, len(all_levels)))
    fig, ax = plt.subplots(figsize=(COL_2, 3.4))
    scenarios = list(per_scenario.keys())
    bottom = np.zeros(len(scenarios))
    for idx, lvl in enumerate(all_levels):
        heights = np.array([per_scenario[s].get(lvl, 0.0) for s in scenarios])
        ax.bar(range(len(scenarios)), heights, bottom=bottom,
               color=cmap(idx), label=f"L{lvl}", edgecolor="black", linewidth=0.3)
        bottom += heights
    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels([s.replace("_", " ") for s in scenarios], rotation=45, ha="right", fontsize=6)
    ax.set_ylabel("Integrated violation $\\int V_\\ell \\, dt$")
    ax.legend(fontsize=6, ncol=len(all_levels), loc="upper left", bbox_to_anchor=(0, 1.12))
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig12] {out_path}")


def fig18_compliance_match(c1_root: Path, c4_root: Path, out_path: Path,
                            eps: float = 1e-4) -> Tuple[Path, List[dict]]:
    """For each scenario and each priority level, compute the per-tick
    binary compliance vector b_l(t) = 1{V_l(t) <= eps} under C1 (WS) and
    C4 (cascade), then report the match rate."""
    c1_dirs = {_scenario_label(d): d for d in _per_scenario_dirs(c1_root)}
    c4_dirs = {_scenario_label(d): d for d in _per_scenario_dirs(c4_root)}
    common = sorted(set(c1_dirs) & set(c4_dirs))

    rows_summary: List[dict] = []
    per_scen_rates: Dict[str, Dict[int, float]] = {}

    for scn in common:
        log_c1 = next(c1_dirs[scn].glob("*_log.csv"), None)
        log_c4 = next(c4_dirs[scn].glob("*_log.csv"), None)
        if not log_c1 or not log_c4:
            continue
        v1 = _per_tick_integrated_by_level(_load_log_csv(log_c1))
        v4 = _per_tick_integrated_by_level(_load_log_csv(log_c4))
        per_level_rate: Dict[int, float] = {}
        for lvl in sorted(set(v1) | set(v4)):
            ticks_c1 = v1.get(lvl, {})
            ticks_c4 = v4.get(lvl, {})
            common_t = set(ticks_c1) & set(ticks_c4)
            if not common_t:
                continue
            n_match = sum(
                1 for t in common_t
                if (ticks_c1[t] <= eps) == (ticks_c4[t] <= eps)
            )
            per_level_rate[lvl] = n_match / len(common_t)
        per_scen_rates[scn] = per_level_rate
        for lvl, r in per_level_rate.items():
            rows_summary.append({"scenario": scn, "level": lvl, "match_rate": r,
                                  "n_ticks_compared": len(set(v1.get(lvl, {})) & set(v4.get(lvl, {})))})

    if not per_scen_rates:
        print("  [fig18] no overlap C1/C4; skipping")
        return out_path, []

    # Small multiples: one column per scenario, levels along x-axis.
    scenarios = list(per_scen_rates.keys())
    all_levels = sorted({l for d in per_scen_rates.values() for l in d})
    fig, ax = plt.subplots(figsize=(COL_2, 3.0))
    width = 0.8 / max(1, len(all_levels))
    cmap = plt.get_cmap("viridis", max(2, len(all_levels)))
    for idx, lvl in enumerate(all_levels):
        rates = [per_scen_rates[s].get(lvl, np.nan) for s in scenarios]
        x = np.arange(len(scenarios)) + (idx - len(all_levels)/2 + 0.5) * width
        ax.bar(x, rates, width=width * 0.95, color=cmap(idx),
               edgecolor="black", linewidth=0.3, label=f"L{lvl}")
    ax.axhline(1.0, color="black", linestyle=":", linewidth=0.5)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels([s.replace("_", " ") for s in scenarios], rotation=45, ha="right", fontsize=6)
    ax.set_ylabel("Compliance match rate $b_{\\rm WS} = b_{\\rm cascade}$")
    ax.legend(fontsize=6, ncol=len(all_levels), loc="upper left", bbox_to_anchor=(0, 1.12))
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig18] {out_path}")
    return out_path, rows_summary


def fig22_walltime_bars(c1_root: Path, c4_root: Path, out_path: Path) -> None:
    c1 = {r["label"]: float(r["duration_s"]) for r in _load_batch_summary(c1_root)}
    c4 = {r["label"]: float(r["duration_s"]) for r in _load_batch_summary(c4_root)}
    common = sorted(set(c1) & set(c4))
    if not common:
        print(f"  [fig22] no shared scenarios between C1 and C4; skipping")
        return
    x = np.arange(len(common))
    fig, ax = plt.subplots(figsize=(COL_2, 3.0))
    ax.bar(x - 0.2, [c1[s] for s in common], width=0.4, label="C1 (WS)",
           color="steelblue", edgecolor="black", linewidth=0.3)
    ax.bar(x + 0.2, [c4[s] for s in common], width=0.4, label="C4 (cascade)",
           color="firebrick", edgecolor="black", linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", " ") for s in common], rotation=45, ha="right", fontsize=6)
    ax.set_ylabel("Wall-clock duration (s)")
    ax.legend(fontsize=7)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig22] {out_path}  (median ratio C4/C1: {np.median([c4[s]/c1[s] for s in common]):.2f}x)")


def fig28_violation_timeseries(c1_root: Path, c4_root: Path, out_path: Path,
                                 scenario_substr: str = "01_following_slow_lead") -> None:
    """V_ell(t) overlay on a chosen scenario for C1 vs C4."""
    c1_scen = next((d for d in _per_scenario_dirs(c1_root) if scenario_substr in d.name), None)
    c4_scen = next((d for d in _per_scenario_dirs(c4_root) if scenario_substr in d.name), None)
    if c1_scen is None or c4_scen is None:
        print(f"  [fig28] scenario {scenario_substr} not found in both C1 and C4")
        return
    log_c1 = next(c1_scen.glob("*_log.csv"), None)
    log_c4 = next(c4_scen.glob("*_log.csv"), None)
    if not log_c1 or not log_c4:
        print(f"  [fig28] missing _log.csv")
        return
    v1 = _per_tick_integrated_by_level(_load_log_csv(log_c1))
    v4 = _per_tick_integrated_by_level(_load_log_csv(log_c4))
    all_levels = sorted(set(v1) | set(v4))

    fig, axes = plt.subplots(len(all_levels), 1, sharex=True,
                              figsize=(COL_2, 0.85 * len(all_levels) + 0.5))
    if len(all_levels) == 1:
        axes = [axes]
    for ax, lvl in zip(axes, all_levels):
        t_c1 = sorted(v1.get(lvl, {}).keys())
        y_c1 = [v1[lvl][t] for t in t_c1]
        t_c4 = sorted(v4.get(lvl, {}).keys())
        y_c4 = [v4[lvl][t] for t in t_c4]
        ax.plot(t_c1, y_c1, color="steelblue", lw=1.0, label="C1 (WS)")
        ax.plot(t_c4, y_c4, color="firebrick", lw=1.0, label="C4 (cascade)", linestyle="--")
        ax.set_ylabel(f"$V_{{{lvl}}}(t)$", fontsize=7)
        ax.grid(linestyle=":", alpha=0.5)
    axes[-1].set_xlabel("Time (s)")
    axes[0].legend(fontsize=7, loc="upper right")
    fig.suptitle(scenario_substr.replace("_", " "), fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig28] {out_path}")


# ----------------------------------------------------------------------
# Tables (LaTeX fragments)
# ----------------------------------------------------------------------


def _wrap_table(body_lines: str, caption: str, label: str,
                placement: str = "t", small: bool = True) -> str:
    size = "\\footnotesize" if small else ""
    return (
        f"\\begin{{table}}[{placement}]\n"
        f"    \\centering\n"
        f"    \\caption{{{caption}}}\n"
        f"    \\label{{{label}}}\n"
        f"    {size}\n"
        f"{body_lines}"
        f"\\end{{table}}\n"
    )


def table_per_scenario_status(c1_root: Path, out_path: Path) -> None:
    """Per-scenario status; ``Top rule`` is the worst-violating *MPC-controlled*
    rule (filtered to the 13 IDs of §III.B), not the worst-violating rule
    across MPC+observer. The original batch summary mixes observer-only rules
    (notably 2r2 Route Adherence and 1r0 Yield); the manuscript's deployment
    claim is about MPC-controlled rules, so we filter accordingly."""
    scenario_dirs = _per_scenario_dirs(c1_root)
    table_rows: List[dict] = []
    for d in scenario_dirs:
        csvs = list(d.glob("*_log.csv"))
        if not csvs:
            continue
        log_rows = _load_log_csv(csvs[0])
        n_ticks = len({float(r["t_s"]) for r in log_rows})
        mpc_int = _integrated_by_rule(log_rows, mpc_only=True)
        n_viol = sum(1 for v in mpc_int.values() if v > 1e-9)
        if mpc_int:
            top = max(mpc_int, key=mpc_int.get)
            top_v = mpc_int[top]
        else:
            top, top_v = "---", 0.0
        table_rows.append({
            "label": _scenario_label(d), "n_ticks": n_ticks,
            "n_viol": n_viol, "top": top, "top_v": top_v,
        })
    if not table_rows:
        return
    body = "\\begin{tabular}{lrrl}\n\\toprule\n"
    body += "Scenario & Ticks & $\\#$ viol.\\ MPC rules & Top MPC rule (integrated) \\\\\n\\midrule\n"
    for r in table_rows:
        label = r["label"].replace("_", "\\_")
        body += (f"{label} & {r['n_ticks']} & {r['n_viol']} & "
                 f"\\texttt{{{r['top']}}} ({r['top_v']:.2f}) \\\\\n")
    body += "\\bottomrule\n\\end{tabular}\n"
    out_path.write_text(_wrap_table(
        body,
        caption=(
            "Per-scenario status under the operational LCP planner $C_1$ on "
            "the 16-scenario \\texttt{nuPlan-mini} benchmark. The top-rule "
            "column is filtered to the 13 MPC-controlled rules of "
            "\\S\\ref{sec:rulebook:partition}; observer-only and stub rules "
            "are excluded since they are not under planner control."),
        label="tab:per_scenario_status",
    ))
    print(f"  [table_per_scenario] {out_path}")


def table_compliance(summary_rows: List[dict], out_path: Path) -> None:
    if not summary_rows:
        return
    by_level: Dict[int, List[float]] = defaultdict(list)
    for r in summary_rows:
        by_level[r["level"]].append(r["match_rate"])
    body = "\\begin{tabular}{rrrr}\n\\toprule\n"
    body += "Level & $\\#$ scenarios & Median $M_\\ell$ & Min $M_\\ell$ \\\\\n\\midrule\n"
    for lvl in sorted(by_level):
        rs = by_level[lvl]
        body += f"$L_{{{lvl}}}$ & {len(rs)} & {np.median(rs):.3f} & {min(rs):.3f} \\\\\n"
    body += "\\bottomrule\n\\end{tabular}\n"
    out_path.write_text(_wrap_table(
        body,
        caption=("Per-tick compliance match rate $M_\\ell^{(s)}$ "
                 "(Definition~\\ref{def:compliance-match}) aggregated by priority level across the benchmark."),
        label="tab:compliance",
    ))
    print(f"  [table_compliance] {out_path}")


def table_from_scan_csv(scan_csv: Path, out_path: Path, columns: List[str],
                          caption: str, label: str) -> None:
    if not scan_csv.exists():
        return
    with scan_csv.open() as f:
        rows = list(csv.DictReader(f))
    body = "\\begin{tabular}{" + "l" * len(columns) + "}\n\\toprule\n"
    body += " & ".join(col.replace("_", "\\_") for col in columns) + " \\\\\n\\midrule\n"
    for r in rows:
        body += " & ".join(str(r.get(c, "")).replace("_", "\\_") for c in columns) + " \\\\\n"
    body += "\\bottomrule\n\\end{tabular}\n"
    out_path.write_text(_wrap_table(body, caption=caption, label=label))
    print(f"  [{out_path.stem}] {out_path}")


def table_walltime(c1_root: Path, c4_root: Path, out_path: Path) -> None:
    c1 = {r["label"]: float(r["duration_s"]) for r in _load_batch_summary(c1_root)
          if r.get("status") == "ok"}
    c4 = {r["label"]: float(r["duration_s"]) for r in _load_batch_summary(c4_root)
          if r.get("status") == "ok"}
    common = sorted(set(c1) & set(c4))
    if not common:
        return
    ratios = [c4[s] / c1[s] for s in common]
    body = "\\begin{tabular}{lrrr}\n\\toprule\n"
    body += "Scenario & $T_{C_1}$ (s) & $T_{C_4}$ (s) & $\\rho^{(s)}$ \\\\\n\\midrule\n"
    for s in common:
        body += f"{s.replace('_', chr(92)+'_')} & {c1[s]:.1f} & {c4[s]:.1f} & {c4[s]/c1[s]:.2f} \\\\\n"
    body += "\\midrule\n"
    body += f"\\textbf{{Median}} & --- & --- & \\textbf{{{np.median(ratios):.2f}}} \\\\\n"
    body += "\\bottomrule\n\\end{tabular}\n"
    out_path.write_text(_wrap_table(
        body,
        caption=("Per-scenario wall-time of the operational $C_1$ and the cascade $C_4$ "
                 "and the ratio $\\rho^{(s)}$ (Definition~\\ref{def:walltime-ratio})."),
        label="tab:walltime",
    ))
    print(f"  [table_walltime] {out_path}  (median ratio: {np.median(ratios):.2f}x)")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--c1-root", type=Path,
        default=_WORKSPACE_ROOT / "examples" / "outputs" / "manuscript" / "C1_instrumented",
    )
    parser.add_argument(
        "--c4-root", type=Path,
        default=_WORKSPACE_ROOT / "examples" / "outputs" / "manuscript" / "C4_cascade",
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=_WORKSPACE_ROOT / "examples" / "outputs" / "manuscript",
    )
    parser.add_argument(
        "--manuscript-figures-dir", type=Path,
        default=Path("/workspace/nuplan-project/IEEE_T-IV/Figures"),
        help="Also drop generated figures here for direct LaTeX inclusion.",
    )
    parser.add_argument(
        "--manuscript-tables-dir", type=Path,
        default=Path("/workspace/nuplan-project/IEEE_T-IV/Sections/tables"),
        help="Also drop generated tables here for direct LaTeX inclusion.",
    )
    args = parser.parse_args()

    fig_dir = args.out_dir / "figures"
    tab_dir = args.out_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)
    args.manuscript_figures_dir.mkdir(parents=True, exist_ok=True)
    args.manuscript_tables_dir.mkdir(parents=True, exist_ok=True)

    import shutil

    def _emit_fig(builder, *args_, **kw):
        out_p = kw.pop("out_path")
        builder(*args_, out_path=out_p, **kw) if False else builder(*args_, out_p, **kw)
        # Mirror to manuscript figures dir if file was written.
        if out_p.exists():
            dest = args.manuscript_figures_dir / out_p.name
            shutil.copy2(out_p, dest)
            # Strip extension to ensure both .pdf and .png variants get picked up.
            print(f"      ↳ mirrored to {dest}")

    print(f"[make_manuscript_artefacts] writing to {args.out_dir}")
    print(f"  + mirror figures → {args.manuscript_figures_dir}")
    print(f"  + mirror tables → {args.manuscript_tables_dir}")

    print("--- figures (C1 only) ---")
    if args.c1_root.exists():
        for name, fn in [("fig10_top_violation", fig10_top_violation),
                          ("fig11_rule_scenario_heatmap", fig11_rule_scenario_heatmap),
                          ("fig12_level_breakdown", fig12_level_breakdown)]:
            fp = fig_dir / f"{name}.pdf"
            fn(args.c1_root, fp)
            if fp.exists():
                shutil.copy2(fp, args.manuscript_figures_dir / fp.name)
    else:
        print(f"  C1 root missing: {args.c1_root}")

    print("--- figures (C1 vs C4) ---")
    compliance_rows = []
    if args.c1_root.exists() and args.c4_root.exists():
        for name, fn in [("fig18_compliance_match", fig18_compliance_match),
                          ("fig22_walltime_bars", fig22_walltime_bars),
                          ("fig28_violation_timeseries", fig28_violation_timeseries)]:
            fp = fig_dir / f"{name}.pdf"
            result = fn(args.c1_root, args.c4_root, fp)
            if name == "fig18_compliance_match" and result is not None:
                _, compliance_rows = result
            if fp.exists():
                shutil.copy2(fp, args.manuscript_figures_dir / fp.name)
    else:
        print(f"  C4 root missing or incomplete; skipping C1-vs-C4 figures")

    print("--- tables ---")
    for name, args_ in [
        ("table_per_scenario_status",
         lambda p: table_per_scenario_status(args.c1_root, p)),
        ("table_compliance",
         lambda p: table_compliance(compliance_rows, p)),
        ("table_walltime",
         lambda p: table_walltime(args.c1_root, args.c4_root, p)),
    ]:
        tp = tab_dir / f"{name}.tex"
        args_(tp)
        if tp.exists():
            shutil.copy2(tp, args.manuscript_tables_dir / tp.name)

    diag_csv = args.out_dir / "diagnostics_scan.csv"
    nec_csv = args.out_dir / "necessity_scan.csv"
    for csv_path, name, cols, caption, label in [
        (diag_csv, "table_diagnostics",
         ["scenario", "peak_active_count", "licq_rank", "licq_n_columns",
          "licq_deficit", "framework_applies"],
         ("Pre-flight diagnostics (\\cite{lcp2025} \\S8.5) per benchmark scenario. "
          "LICQ rank reports the rank of the combined active-gradient matrix at "
          "the peak-violation tick of the $C_1$ run; framework applies iff FM~I "
          "(LICQ full rank) and FM~II (convexity) both hold."),
         "tab:diagnostics"),
        (nec_csv, "table_necessity",
         ["scenario", "i_star_nec", "forced_levels", "per_level_violation_frac"],
         ("Empirical \\S10.2 necessity scan per benchmark scenario. "
          "$i^\\star_{\\mathrm{nec}}$ is the largest priority level for which "
          "level $i^\\star_{\\mathrm{nec}}$ and every higher-priority level had "
          "$\\le 1\\%$ of applicable ticks violating; \\emph{forced\\_levels} "
          "lists the levels above $i^\\star_{\\mathrm{nec}}$ that exceeded the "
          "hardness tolerance."),
         "tab:necessity"),
    ]:
        tp = tab_dir / f"{name}.tex"
        table_from_scan_csv(csv_path, tp, cols, caption, label)
        if tp.exists():
            shutil.copy2(tp, args.manuscript_tables_dir / tp.name)

    print(f"\n[done] all artefacts in {args.out_dir}")
    print(f"        manuscript figures: {args.manuscript_figures_dir}")
    print(f"        manuscript tables: {args.manuscript_tables_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
