#!/usr/bin/env python3
"""§10 necessity-of-relaxation scan over the 16-scenario nuPlan-mini batch.

For each completed C1 scenario, identify the *forcibly relaxed* priority
levels per Definition 10.1: the lowest priority i* for which all higher-
priority levels j <= i* were jointly enforced by the planner without
violation across the whole episode.

Specifically, scan the per-tick ``_log.csv`` and for each priority level L
in {0, 1, ..., 7} (top of the bicycle-MPC stratification):

- Compute the fraction of ticks at which **any** rule at level L violated
  while applicable.
- Determine i*_nec = the largest level for which the cumulative violation
  fraction is below an empirical tolerance (default 1%, i.e. the planner
  effectively held that level hard) AND every higher level was also held
  effectively hard.
- Forced-relaxation levels = {L : L > i*_nec}.

This is the data-driven analogue of ``lcp.compute_necessary_relaxation_level``
applied to the empirical scenario rather than to a synthetic LP probe. It
tells us empirically which scenarios *required* relaxation of which levels.

Outputs ``necessity_scan.csv`` with one row per scenario.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

_WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_ROOT))


# 25-rule → 8-level mapping. Lower number = higher priority.
# Source: lexicone.observer rule taxonomy (level digit of rule_id prefix).
def _rule_level(rule_id: str) -> int:
    """Parse the priority level from a rule_id like ``10r0`` or ``3r3``.

    The id format is ``<level_int>r<slot_int>`` where ``level_int`` can be
    one or two digits. Higher integer = lower priority in the rulebook.
    """
    if "r" not in rule_id:
        return -1
    try:
        return int(rule_id.split("r", 1)[0])
    except ValueError:
        return -1


def _per_level_violation_fraction(log_csv: Path) -> dict[int, float]:
    """For each priority level, compute fraction of applicable ticks with
    violation_rate > 0."""
    by_level_applicable = defaultdict(int)
    by_level_violated = defaultdict(int)
    with log_csv.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            lvl = _rule_level(row["rule_id"])
            if lvl < 0:
                continue
            applies = int(row["applies"]) == 1
            vrate = float(row["violation_rate"])
            if applies:
                by_level_applicable[lvl] += 1
                if vrate > 0.0:
                    by_level_violated[lvl] += 1
    out: dict[int, float] = {}
    for lvl in sorted(by_level_applicable):
        n_app = by_level_applicable[lvl]
        n_viol = by_level_violated[lvl]
        out[lvl] = (n_viol / n_app) if n_app else 0.0
    return out


def _scan_one(scenario_dir: Path, tol: float = 0.01) -> dict:
    csvs = list(scenario_dir.glob("*_log.csv"))
    if not csvs:
        return {
            "scenario": scenario_dir.name, "i_star_nec": -1,
            "forced_levels": "", "per_level_violation_frac": "",
            "note": "no _log.csv",
        }
    fracs = _per_level_violation_fraction(csvs[0])
    if not fracs:
        return {
            "scenario": scenario_dir.name, "i_star_nec": -1,
            "forced_levels": "", "per_level_violation_frac": "",
            "note": "no applicable ticks",
        }

    # In the deployment rulebook, HIGHER ell digit = HIGHER priority. The
    # framework's cascade walks high → low priority and the §10.2 necessity
    # probe asks "what's the lowest-priority level we can still enforce hard
    # given that every higher-priority level is also held hard?" We walk
    # the full level spectrum 0..10 in descending order; a level with no
    # applicable ticks is vacuously hard. i*_nec is the lowest level that
    # was hard while every higher level was also hard.
    ALL_LEVELS = list(range(10, -1, -1))   # 10, 9, ..., 0
    i_star_nec = None
    for lvl in ALL_LEVELS:
        frac = fracs.get(lvl, 0.0)   # missing level = vacuously hard
        if frac <= tol:
            i_star_nec = lvl
        else:
            break
    i_star_nec_out = i_star_nec if i_star_nec is not None else -1
    forced = [str(lvl) for lvl in sorted(fracs.keys())
              if lvl < (i_star_nec if i_star_nec is not None else 999)
              and fracs.get(lvl, 0.0) > tol]
    i_star_nec = i_star_nec_out
    return {
        "scenario": scenario_dir.name,
        "i_star_nec": i_star_nec,
        "forced_levels": ",".join(forced),
        "per_level_violation_frac": ";".join(
            f"L{lvl}={fracs[lvl]:.3f}" for lvl in sorted(fracs.keys())
        ),
        "note": "ok",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--batch-dir", type=Path,
        default=_WORKSPACE_ROOT / "examples" / "outputs" / "manuscript" / "C1_instrumented",
    )
    parser.add_argument(
        "--out", type=Path,
        default=_WORKSPACE_ROOT / "examples" / "outputs" / "manuscript" / "necessity_scan.csv",
    )
    parser.add_argument(
        "--tol", type=float, default=0.01,
        help="Per-level violation fraction below which a level is considered effectively hard.",
    )
    args = parser.parse_args()

    if not args.batch_dir.exists():
        print(f"ERROR: batch dir not found: {args.batch_dir}", file=sys.stderr)
        return 2

    scenario_dirs = sorted(p for p in args.batch_dir.iterdir() if p.is_dir())
    rows = [_scan_one(d, tol=args.tol) for d in scenario_dirs]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"§10 necessity scan: {len(rows)} scenarios")
    print(f"  tolerance for 'effectively hard': {args.tol:.3f} (per-level violation fraction)")
    print(f"  CSV: {args.out}")
    print()
    print(f"  {'scenario':<40} {'i*_nec':>7} {'forced levels':>15}")
    print("  " + "-" * 70)
    for r in rows:
        print(f"  {r['scenario']:<40} {r['i_star_nec']:>7} "
              f"{r['forced_levels']:>15}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
