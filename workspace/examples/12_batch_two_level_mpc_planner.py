#!/usr/bin/env python3
"""Demo 12 — Batch evaluation of the two-level MPC planner across involved scenarios.

Runs the full pipeline from :mod:`examples.11_simulated_two_level_mpc_planner`
once per scenario type, on a curated list spanning the kinds of behaviour the
user asked about — overtake-style (slow lead, long vehicle, near multi-traffic),
lane changes (left, right, generic), turns (left, right, protected/unprotected,
high- and low-speed), and dynamic driving (high/medium magnitude speed, near
high-speed vehicles, construction zones). nuPlan's mini split doesn't include
a literal "u-turn" or "ramp exit" scenario type; the closest available proxies
are ``starting_left_turn`` (sharp turn) and ``starting_high_speed_turn``
(high-speed turn — analogous to ramp exits).

For each scenario the script invokes ``nuplan/planning/script/run_simulation.py``
with our planner config plus ``scenario_filter.scenario_types=[<type>]`` so the
simulator picks exactly one matching scenario from the readable mini DBs. The
resulting :class:`SimulationLog` is rendered via the lexicone visualiser to MP4
+ summary PNG + per-tick CSV. A combined table summarising every run is printed
at the end and dumped to ``batch_summary.csv``.

Expect ~1.5 minutes per scenario; the default 16-scenario list takes roughly
25 minutes wall-clock on a single CPU.

Usage::

    cd workspace
    python examples/12_batch_two_level_mpc_planner.py --seed 7

Outputs land in ``workspace/examples/outputs/12_batch_two_level_mpc_planner/<label>/``.
"""

from __future__ import annotations

import argparse
import csv
import os
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

# Make the workspace package and the DevContainers demo helpers importable.
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
DEMOS_ROOT = Path("/workspace/nuplan-project/DevContainers/demos").resolve()
DEVKIT_ROOT = Path(os.environ.get("NUPLAN_DEVKIT_ROOT", "/workspace/nuplan-devkit")).resolve()
PLANNER_CONFIG_DIR = WORKSPACE_ROOT / "lexicone" / "planning" / "config"

for path in (DEVKIT_ROOT, WORKSPACE_ROOT, DEMOS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common.nuplan_demo_utils import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    DEFAULT_EXP_ROOT,
    ensure_output_dir,
    find_db_files,
    list_tables,
)

from lexicone.observer import RuleEngine  # noqa: E402
from lexicone.observer.simulation_log_adapter import NuPlanSimulationLogSource  # noqa: E402

from examples.visualizer import render_episode  # noqa: E402


@dataclass
class ScenarioSpec:
    """One row in the batch list."""

    label: str           # short slug used for directory naming
    description: str     # human-readable category (used in summary table)
    scenario_type: str   # nuPlan scenario_tag.type to filter on


@dataclass
class BatchResult:
    spec: ScenarioSpec
    status: str
    duration_s: float
    n_ticks: int = 0
    scenario_label: str = ""
    mp4: Optional[Path] = None
    summary_png: Optional[Path] = None
    csv_path: Optional[Path] = None
    rule_violations: List[Tuple[str, int, int, float]] = field(default_factory=list)
    error: str = ""


# 16-scenario default list. Behaviour mix tracks the user's request:
# overtake-style (slow lead, long vehicle, multi-traffic), lane change
# (changing_lane{,_to_left,_to_right}), turns (left/right/high-speed/
# protected/unprotected), and dynamic driving (speed/near high-speed/
# construction zone).
DEFAULT_SCENARIOS: List[ScenarioSpec] = [
    ScenarioSpec("01_following_slow_lead", "Overtake-like: slow lead ahead", "following_lane_with_slow_lead"),
    ScenarioSpec("02_near_long_vehicle", "Overtake-like: near long vehicle", "near_long_vehicle"),
    ScenarioSpec("03_near_multiple_vehicles", "Overtake-like: multi-vehicle traffic", "near_multiple_vehicles"),
    ScenarioSpec("04_changing_lane", "Lane change (any)", "changing_lane"),
    ScenarioSpec("05_changing_lane_left", "Lane change (to left)", "changing_lane_to_left"),
    ScenarioSpec("06_changing_lane_right", "Lane change (to right)", "changing_lane_to_right"),
    ScenarioSpec("07_starting_left_turn", "Sharp left turn (proxy for U-turn)", "starting_left_turn"),
    ScenarioSpec("08_starting_right_turn", "Right turn", "starting_right_turn"),
    ScenarioSpec("09_high_speed_turn", "High-speed turn (proxy for ramp exit)", "starting_high_speed_turn"),
    ScenarioSpec("10_low_speed_turn", "Low-speed turn", "starting_low_speed_turn"),
    ScenarioSpec("11_protected_cross", "Protected cross turn", "starting_protected_cross_turn"),
    ScenarioSpec("12_unprotected_cross", "Unprotected cross turn", "starting_unprotected_cross_turn"),
    ScenarioSpec("13_high_magnitude_speed", "High-magnitude speed", "high_magnitude_speed"),
    ScenarioSpec("14_medium_magnitude_speed", "Medium-magnitude speed", "medium_magnitude_speed"),
    ScenarioSpec("15_near_high_speed_vehicle", "Near high-speed vehicle", "near_high_speed_vehicle"),
    ScenarioSpec("16_traversing_intersection", "Traversing intersection", "traversing_intersection"),
]


def db_is_readable(db_path: Path) -> bool:
    try:
        with sqlite3.connect(str(db_path)) as con:
            tables = list_tables(con)
            return "lidar_pc" in tables and "ego_pose" in tables and "scenario_tag" in tables
    except Exception:
        return False


def find_readable_log_stems(data_root: Path, max_logs: int = 12) -> list[str]:
    db_files = find_db_files(data_root, None)
    eligible = [db for db in db_files if db_is_readable(db)]
    if not eligible:
        raise RuntimeError(
            "No readable mini DBs were found. "
            "Run scripts/bootstrap_nuplan.sh --profile planner_mini."
        )
    return [db.stem for db in eligible[:max_logs]]


def find_latest_simulation_log(experiment_dir: Path) -> Optional[Path]:
    candidates = list(experiment_dir.rglob("*.msgpack.xz")) + list(experiment_dir.rglob("*.pkl.xz"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def run_one_simulation(
    *,
    spec: ScenarioSpec,
    log_names: list[str],
    seed: int,
    mpc_horizon_s: float,
    mpc_dt_s: float,
    replan_period_s: float,
    desired_speed_mps: float,
    penalty_form: Optional[str] = None,
    runtime_mode: str = "ws",
    slp_max_iterations: int = 1,
    slp_residual_tol_m: float = 0.05,
) -> Path:
    """Invoke ``run_simulation.py`` for a single scenario type, return the experiment dir.

    When ``penalty_form`` is set (``"l1"`` or ``"l2"``), the LCP-mode MPC is
    activated via Hydra override; otherwise the legacy single-tier MPC runs.
    The experiment-name suffix encodes the mode so legacy and LCP runs land in
    separate output directories and don't clobber each other.
    """
    runtime_suffix = f"_{runtime_mode}" if (penalty_form and runtime_mode != "ws") else ""
    mode_suffix = f"__{penalty_form}{runtime_suffix}" if penalty_form else ""
    experiment_name = f"demo_12_batch__{spec.label}{mode_suffix}"
    exp_root = Path(os.environ.get("NUPLAN_EXP_ROOT", str(DEFAULT_EXP_ROOT)))

    searchpath_override = (
        f"hydra.searchpath=[file://{PLANNER_CONFIG_DIR},"
        "pkg://nuplan.planning.script.config.common,"
        "pkg://nuplan.planning.script.experiments]"
    )
    log_names_csv = "[" + ",".join(log_names) + "]"

    command = [
        sys.executable,
        "nuplan/planning/script/run_simulation.py",
        "+simulation=closed_loop_reactive_agents",
        "planner=two_level_mpc_planner",
        f"planner.two_level_mpc_planner.mpc_horizon_s={mpc_horizon_s}",
        f"planner.two_level_mpc_planner.mpc_dt_s={mpc_dt_s}",
        f"planner.two_level_mpc_planner.replan_period_s={replan_period_s}",
        f"planner.two_level_mpc_planner.desired_speed_mps={desired_speed_mps}",
        "scenario_builder=nuplan_mini",
        "scenario_filter=all_scenarios",
        f"scenario_filter.scenario_types=[{spec.scenario_type}]",
        f"scenario_filter.log_names={log_names_csv}",
        "scenario_filter.limit_total_scenarios=1",
        "scenario_filter.shuffle=true",
        f"seed={seed}",
        "worker=sequential",
        f"experiment_name={experiment_name}",
        f"job_name={experiment_name}",
        searchpath_override,
    ]
    if penalty_form is not None:
        command.extend([
            f"planner.two_level_mpc_planner.penalty_form={penalty_form}",
            # Use "auto" weight sentinels — the calibration cache will resolve
            # to heuristic defaults on miss, which is the right behaviour for
            # an uncalibrated first run.
            "planner.two_level_mpc_planner.weights_per_level=[auto,auto,auto]",
            "planner.two_level_mpc_planner.epsilon_per_level=[1.0e-4,4.0e-2,5.0e-1]",
            f"planner.two_level_mpc_planner.scenario_class_hint={spec.scenario_type}",
            f"planner.two_level_mpc_planner.runtime_mode={runtime_mode}",
            f"planner.two_level_mpc_planner.slp_max_iterations={slp_max_iterations}",
            f"planner.two_level_mpc_planner.slp_residual_tol_m={slp_residual_tol_m}",
        ])
    env = os.environ.copy()
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join([str(WORKSPACE_ROOT)] + ([existing_pp] if existing_pp else []))
    # Quiet the per-run output; we surface the high-level results in the summary table.
    subprocess.run(command, cwd=str(DEVKIT_ROOT), check=True, env=env, stdout=subprocess.DEVNULL)
    return exp_root / "exp" / experiment_name / experiment_name


def render_log(
    *,
    spec: ScenarioSpec,
    log_path: Path,
    out_dir: Path,
    fps: int,
    map_margin_m: float,
    radius_m: float,
) -> Tuple[List[Tuple[str, int, int, float]], int, dict]:
    """Convert simulation log → SceneSnapshots → MP4/summary/csv. Returns rule rows + tick count + artefacts."""
    source = NuPlanSimulationLogSource.from_path(log_path, radius_m=radius_m)
    snapshots = list(source)
    if not snapshots:
        raise RuntimeError(f"empty snapshot stream from {log_path}")
    engine = RuleEngine()
    artefacts = render_episode(
        engine=engine,
        snapshots=snapshots,
        scenario_name=spec.label,
        output_dir=out_dir,
        fps=fps,
        map_margin_m=map_margin_m,
    )
    summary = engine.summary()
    rows = []
    for s in summary.rule_summaries.values():
        if s.n_steps_violated > 0:
            rows.append((s.rule_id, s.n_steps_violated, s.n_steps_applicable, s.integrated_violation))
    rows.sort(key=lambda r: -r[3])
    return rows, len(snapshots), artefacts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "12_batch_two_level_mpc_planner",
    )
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--margin", type=float, default=80.0)
    parser.add_argument("--radius", type=float, default=80.0)
    parser.add_argument("--seed", type=int, default=7, help="Hydra seed (advanced by 1 per scenario for variety).")
    parser.add_argument("--mpc-horizon-s", type=float, default=3.0)
    parser.add_argument("--mpc-dt-s", type=float, default=0.1)
    parser.add_argument("--replan-period-s", type=float, default=8.0)
    parser.add_argument("--desired-speed-mps", type=float, default=12.0)
    parser.add_argument("--max-logs", type=int, default=12, help="Cap on readable mini DBs to pass through Hydra.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of scenarios to actually run (defaults to all 16).",
    )
    parser.add_argument(
        "--types",
        type=str,
        default=None,
        help=(
            "Comma-separated list of nuPlan scenario_tag.type values to run instead of "
            "the default 16. Labels are auto-generated as ``<offset+i>_<type>``."
        ),
    )
    parser.add_argument(
        "--label-offset",
        type=int,
        default=17,
        help="Starting integer for auto-generated labels when --types is used (default: 17).",
    )
    parser.add_argument(
        "--penalty-form",
        type=str,
        default=None,
        choices=[None, "l1", "l2"],
        help=(
            "If set, switches the planner into LCP mode with the given L₁ or L₂ "
            "penalty. The legacy single-tier MPC runs when unset (default). "
            "Output directories and the summary CSV get a `__<penalty_form>` suffix "
            "so legacy and LCP runs don't clobber each other."
        ),
    )
    parser.add_argument(
        "--runtime-mode",
        type=str,
        default="ws",
        choices=["ws", "cascade"],
        help=(
            "Per-tick OCP strategy (only effective when --penalty-form is set): "
            "'ws' runs a single weighted-sum solve at calibrated weights; "
            "'cascade' runs the full L+1-stage lex cascade per tick (formally "
            "lex-optimal by construction, recommended for offline simulation)."
        ),
    )
    parser.add_argument("--slp-max-iterations", type=int, default=1)
    parser.add_argument("--slp-residual-tol-m", type=float, default=0.05)
    args = parser.parse_args()

    out_root = ensure_output_dir(args.output_dir)
    log_names = find_readable_log_stems(args.data_root, max_logs=args.max_logs)
    print(f"[batch] using {len(log_names)} readable mini DBs.")
    if args.penalty_form:
        print(f"[batch] LCP mode enabled: penalty_form={args.penalty_form}")

    if args.types:
        type_list = [t.strip() for t in args.types.split(",") if t.strip()]
        scenarios = [
            ScenarioSpec(
                label=f"{args.label_offset + idx:02d}_{stype}",
                description=stype.replace("_", " "),
                scenario_type=stype,
            )
            for idx, stype in enumerate(type_list)
        ]
    else:
        scenarios = DEFAULT_SCENARIOS if args.limit is None else DEFAULT_SCENARIOS[: args.limit]
    print(f"[batch] running {len(scenarios)} scenarios.")
    print(f"[batch] artefacts under: {out_root}")

    results: List[BatchResult] = []
    t_start_all = time.time()
    runtime_suffix = (
        f"_{args.runtime_mode}" if args.penalty_form and args.runtime_mode != "ws" else ""
    )
    mode_suffix = f"__{args.penalty_form}{runtime_suffix}" if args.penalty_form else ""
    for idx, spec in enumerate(scenarios, start=1):
        print(f"\n[batch] {idx:02d}/{len(scenarios)} {spec.label}{mode_suffix}  ({spec.description})")
        t0 = time.time()
        case_dir = ensure_output_dir(out_root / f"{spec.label}{mode_suffix}")
        seed = args.seed + idx
        try:
            experiment_dir = run_one_simulation(
                spec=spec,
                log_names=log_names,
                seed=seed,
                mpc_horizon_s=args.mpc_horizon_s,
                mpc_dt_s=args.mpc_dt_s,
                replan_period_s=args.replan_period_s,
                desired_speed_mps=args.desired_speed_mps,
                penalty_form=args.penalty_form,
                runtime_mode=args.runtime_mode,
                slp_max_iterations=args.slp_max_iterations,
                slp_residual_tol_m=args.slp_residual_tol_m,
            )
        except subprocess.CalledProcessError as exc:
            elapsed = time.time() - t0
            results.append(BatchResult(spec=spec, status="sim_failed", duration_s=elapsed, error=str(exc)))
            print(f"[batch]   simulation failed ({elapsed:.1f}s)")
            continue
        log_path = find_latest_simulation_log(experiment_dir)
        if log_path is None:
            elapsed = time.time() - t0
            results.append(BatchResult(spec=spec, status="no_log", duration_s=elapsed))
            print(f"[batch]   no simulation log produced ({elapsed:.1f}s)")
            continue
        try:
            rows, n_ticks, artefacts = render_log(
                spec=spec,
                log_path=log_path,
                out_dir=case_dir,
                fps=args.fps,
                map_margin_m=args.margin,
                radius_m=args.radius,
            )
        except Exception as exc:
            elapsed = time.time() - t0
            results.append(BatchResult(spec=spec, status="render_failed", duration_s=elapsed, error=str(exc)))
            print(f"[batch]   render failed: {exc}")
            continue
        elapsed = time.time() - t0
        scenario_label = log_path.parent.parent.name  # SimLog dir layout: .../<scenario_type>/<log>/<token>/
        results.append(
            BatchResult(
                spec=spec,
                status="ok",
                duration_s=elapsed,
                n_ticks=n_ticks,
                scenario_label=scenario_label,
                mp4=artefacts.get("mp4"),
                summary_png=artefacts.get("summary"),
                csv_path=artefacts.get("csv"),
                rule_violations=rows,
            )
        )
        top_rule = (
            f"top violation: {rows[0][0]} (integrated={rows[0][3]:.1f})"
            if rows
            else "no rule violated"
        )
        print(f"[batch]   ok ({elapsed:.1f}s, {n_ticks} ticks)  {top_rule}")

    total_elapsed = time.time() - t_start_all
    print(f"\n[batch] all done in {total_elapsed/60.0:.1f} min")

    # CSV + console summary. When the user passed a custom --types list or
    # enabled LCP mode, write to a suffixed file so each run keeps its own
    # summary alongside the others.
    name_bits = ["batch_summary"]
    if args.types:
        name_bits.append(f"extra_{args.label_offset:02d}")
    if args.penalty_form:
        name_bits.append(args.penalty_form)
    if args.penalty_form and args.runtime_mode != "ws":
        name_bits.append(args.runtime_mode)
    summary_name = "_".join(name_bits) + ".csv"
    summary_csv = out_root / summary_name
    with summary_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["label", "description", "scenario_type", "status", "duration_s", "n_ticks", "scenario_token",
             "n_violating_rules", "top_rule_id", "top_integrated_violation", "mp4_path"]
        )
        for r in results:
            top = r.rule_violations[0] if r.rule_violations else ("", 0, 0, 0.0)
            w.writerow([
                r.spec.label,
                r.spec.description,
                r.spec.scenario_type,
                r.status,
                f"{r.duration_s:.1f}",
                r.n_ticks,
                r.scenario_label,
                len(r.rule_violations),
                top[0],
                f"{top[3]:.3f}",
                str(r.mp4) if r.mp4 else "",
            ])
    print(f"\n[batch] summary csv: {summary_csv}\n")

    # Pretty console table
    print(f"{'label':<32} {'status':<9} {'ticks':>5} {'#viol':>5} {'top rule':>10} {'integrated':>10}  description")
    print("-" * 120)
    for r in results:
        top = r.rule_violations[0] if r.rule_violations else ("—", 0, 0, 0.0)
        print(
            f"{r.spec.label:<32} {r.status:<9} {r.n_ticks:>5} "
            f"{len(r.rule_violations):>5} {top[0]:>10} {top[3]:>10.2f}  {r.spec.description}"
        )

    n_ok = sum(1 for r in results if r.status == "ok")
    n_fail = len(results) - n_ok
    print(f"\n[batch] {n_ok} succeeded, {n_fail} failed/skipped.")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
