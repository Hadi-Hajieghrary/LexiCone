"""Multi-seed multi-condition driver for the comparative-effectiveness protocol.

Wraps :mod:`examples.12_batch_two_level_mpc_planner` so that the *same* 16
nuPlan-mini scenarios run under up to 5 planner conditions:

* **C0_legacy**   — flat-weight single-tier MPC
* **C1_ws_l1**    — LCP with L1 penalty, weighted-sum solve at calibrated w†
* **C2_ws_l1_slp3** — same as C1 but with 3 SLP outer iterations
* **C3_ws_l2**    — LCP with L2 penalty
* **C4_cascade_l1** — full lex cascade per tick

Each condition is repeated across N seeds (default ``{7, 17, 27, 37, 47}``).
Within a condition, the per-scenario seed advances by 1 from the base seed.

Outputs land in ``examples/outputs/13_protocol/<condition>/seed_<n>/``, matching
the layout the analysis pipeline (:mod:`examples.analyze_protocol`) expects.

Usage::

    # smoke test: 1 scenario × 3 conditions × 1 seed
    python examples/13_run_protocol.py --conditions C0,C1,C4 --seeds 7 --limit 1

    # comprehensive: 5 conditions × 16 scenarios × 5 seeds (~36 hr on 4 CPUs)
    python examples/13_run_protocol.py \\
        --conditions C0,C1,C2,C3,C4 --seeds 7,17,27,37,47 \\
        --parallel 4

    # resume after interruption (skips cells whose summary CSV already exists)
    python examples/13_run_protocol.py --conditions C0,C1,C4 --seeds 7,17 --resume
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import List, Optional, Tuple


WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
BATCH_SCRIPT = WORKSPACE_ROOT / "examples" / "12_batch_two_level_mpc_planner.py"
PROTOCOL_ROOT = WORKSPACE_ROOT / "examples" / "outputs" / "13_protocol"


@dataclass
class Condition:
    """One planner variant in the protocol."""
    code: str
    description: str
    cli_flags: List[str]
    expected_wall_min_per_scenario: float


CONDITIONS = {
    "C0": Condition(
        code="C0_legacy",
        description="Legacy flat-weight MPC (baseline)",
        cli_flags=[],
        expected_wall_min_per_scenario=3.0,
    ),
    "C1": Condition(
        code="C1_ws_l1",
        description="LCP L1 weighted-sum",
        cli_flags=["--penalty-form", "l1", "--runtime-mode", "ws",
                   "--slp-max-iterations", "1"],
        expected_wall_min_per_scenario=3.0,
    ),
    "C2": Condition(
        code="C2_ws_l1_slp3",
        description="LCP L1 WS with 3 SLP iterations",
        cli_flags=["--penalty-form", "l1", "--runtime-mode", "ws",
                   "--slp-max-iterations", "3"],
        expected_wall_min_per_scenario=6.0,
    ),
    "C3": Condition(
        code="C3_ws_l2",
        description="LCP L2 weighted-sum",
        cli_flags=["--penalty-form", "l2", "--runtime-mode", "ws",
                   "--slp-max-iterations", "1"],
        expected_wall_min_per_scenario=3.0,
    ),
    "C4": Condition(
        code="C4_cascade_l1",
        description="LCP L1 lex cascade (formally lex-optimal)",
        cli_flags=["--penalty-form", "l1", "--runtime-mode", "cascade",
                   "--slp-max-iterations", "1"],
        expected_wall_min_per_scenario=16.0,
    ),
}


def cell_output_dir(condition_code: str, seed: int) -> Path:
    return PROTOCOL_ROOT / condition_code / f"seed_{seed}"


def cell_already_done(condition_code: str, seed: int, expected_scenarios: int) -> bool:
    """Resume check: cell is done iff its summary CSV exists with N rows."""
    out_dir = cell_output_dir(condition_code, seed)
    if not out_dir.is_dir():
        return False
    csv_files = list(out_dir.glob("batch_summary*.csv"))
    if not csv_files:
        return False
    # Count data rows (subtract header).
    for csv_file in csv_files:
        n_rows = sum(1 for _ in csv_file.open()) - 1
        if n_rows >= expected_scenarios:
            return True
    return False


def run_one_cell(args: Tuple[Condition, int, int]) -> dict:
    """Run one ``(condition, seed)`` cell.

    Each call is one subprocess that runs ``12_batch_two_level_mpc_planner.py``
    over all scenarios for that condition+seed.
    """
    cond, seed, limit = args
    out_dir = cell_output_dir(cond.code, seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"

    # Each cell gets its OWN NUPLAN_EXP_ROOT (keyed on cond + seed) so that
    # parallel cells can never collide on the simulator's intermediate
    # experiment dirs. The directory is removed in the post-cell cleanup.
    # Using cell-unique (not worker-unique) keys avoids a race where two
    # cells dispatched to the same Pool worker slot — but not necessarily
    # the same Pool worker process — could share the same EXP_ROOT.
    env = os.environ.copy()
    base_exp_root = Path(env.get("NUPLAN_EXP_ROOT", "/workspace/exp"))
    cell_exp_root = base_exp_root.parent / f"{base_exp_root.name}__{cond.code}__seed_{seed}"
    cell_exp_root.mkdir(parents=True, exist_ok=True)
    env["NUPLAN_EXP_ROOT"] = str(cell_exp_root)

    cmd = [
        sys.executable, str(BATCH_SCRIPT),
        "--seed", str(seed),
        "--output-dir", str(out_dir),
    ]
    if limit is not None:
        cmd.extend(["--limit", str(limit)])
    cmd.extend(cond.cli_flags)

    t0 = time.time()
    print(f"[protocol] starting {cond.code}/seed_{seed}  (cmd: {' '.join(cmd[-6:])})", flush=True)
    with log_path.open("w") as logf:
        result = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)
    elapsed_s = time.time() - t0
    status = "ok" if result.returncode == 0 else f"exit_{result.returncode}"

    # Post-cell cleanup: drop the entire cell-unique EXP_ROOT. The cell's
    # per-tick CSVs, MP4s, and summary CSV are already in ``out_dir``
    # (the batch script wrote them); nothing under cell_exp_root is needed
    # downstream. Capping per-cell footprint = capping disk for the whole run.
    bytes_freed = 0
    try:
        if cell_exp_root.is_dir():
            bytes_freed = sum(
                f.stat().st_size for f in cell_exp_root.rglob("*") if f.is_file()
            )
            shutil.rmtree(cell_exp_root, ignore_errors=True)
    except Exception as exc:  # pragma: no cover - cleanup best-effort
        print(f"[protocol]   cleanup warning: {exc}", flush=True)

    print(f"[protocol] {cond.code}/seed_{seed} {status} in {elapsed_s/60:.1f} min"
          f"  (freed {bytes_freed/1e9:.1f} GB)", flush=True)
    return {"condition": cond.code, "seed": seed, "status": status,
            "elapsed_min": elapsed_s / 60, "log_path": str(log_path),
            "bytes_freed": bytes_freed}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--conditions", type=str, default="C0,C1,C4",
        help="Comma-separated condition codes (subset of C0,C1,C2,C3,C4).",
    )
    p.add_argument(
        "--seeds", type=str, default="7",
        help="Comma-separated base seeds.",
    )
    p.add_argument(
        "--limit", type=int, default=None,
        help="Cap on scenarios per cell (default: all 16).",
    )
    p.add_argument(
        "--parallel", type=int, default=1,
        help="Number of parallel cells (each cell gets its own NUPLAN_EXP_ROOT).",
    )
    p.add_argument(
        "--resume", action="store_true",
        help="Skip cells whose summary CSV already exists with the expected row count.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Just print the cells that would be run + estimated wall time.",
    )
    args = p.parse_args()

    cond_codes = [c.strip() for c in args.conditions.split(",") if c.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    unknown = [c for c in cond_codes if c not in CONDITIONS]
    if unknown:
        print(f"unknown conditions: {unknown}; valid: {list(CONDITIONS)}",
              file=sys.stderr)
        return 1

    conditions = [CONDITIONS[c] for c in cond_codes]
    n_scenarios = args.limit if args.limit is not None else 16

    # Estimate total wall time.
    seq_min = sum(c.expected_wall_min_per_scenario for c in conditions) * len(seeds) * n_scenarios
    par_min = seq_min / max(args.parallel, 1)
    print(f"[protocol] {len(conditions)} conditions × {len(seeds)} seeds "
          f"× {n_scenarios} scenarios = {len(conditions) * len(seeds)} cells "
          f"({len(conditions) * len(seeds) * n_scenarios} runs)")
    print(f"[protocol] est. wall: {seq_min/60:.1f} hr sequential, "
          f"{par_min/60:.1f} hr at {args.parallel}× parallel")
    print(f"[protocol] output root: {PROTOCOL_ROOT}")

    cells: List[Tuple[Condition, int, int]] = []
    for cond in conditions:
        for seed in seeds:
            if args.resume and cell_already_done(cond.code, seed, n_scenarios):
                print(f"[protocol] skip {cond.code}/seed_{seed} (already done)")
                continue
            cells.append((cond, seed, args.limit))

    if args.dry_run:
        for cond, seed, _ in cells:
            print(f"  would run: {cond.code} seed={seed}")
        return 0

    if not cells:
        print("[protocol] nothing to run")
        return 0

    t_start = time.time()
    if args.parallel <= 1:
        results = [run_one_cell(c) for c in cells]
    else:
        with Pool(args.parallel) as pool:
            results = pool.map(run_one_cell, cells)
    total_min = (time.time() - t_start) / 60

    print()
    print(f"[protocol] done. {len(results)} cells in {total_min:.1f} min")
    n_ok = sum(1 for r in results if r["status"] == "ok")
    print(f"[protocol] {n_ok}/{len(results)} succeeded")
    for r in results:
        if r["status"] != "ok":
            print(f"  FAILED {r['condition']}/seed_{r['seed']} → {r['log_path']}")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
