#!/usr/bin/env python3
"""Demo 11 — Two-level (global route + CasADi MPC) planner in the nuPlan simulator.

This demo wires our two-level motion planner (:mod:`lexicone.planning`) into
the official nuPlan closed-loop simulator. The flow is:

1. Pick a *dynamic* mini-split scenario (anything where the ego actually moves
   ≥ 20 m and starts above 3 m/s; we explicitly avoid the ``stationary`` types).
2. Subprocess into ``nuplan/planning/script/run_simulation.py`` with our
   planner Hydra config (``planner=two_level_mpc_planner``); the planner YAML
   lives in [workspace/lexicone/planning/config/planner](../lexicone/planning/config/planner) and is
   discovered via a ``hydra.searchpath`` override.
3. Load the saved :class:`SimulationLog` via
   :class:`~lexicone.observer.simulation_log_adapter.NuPlanSimulationLogSource`,
   convert each sample to a :class:`SceneSnapshot`, and hand the stream to the
   richer lexicone visualiser ([examples/visualizer.py](visualizer.py))
   that the other demos use. That gives us the full map (drivable area, lanes,
   crosswalks, traffic lights, lane direction arrows, …), the ego footprint,
   agents coloured by class, the planned trajectory, and the rule-engine
   strip.

Usage
-----
::

    cd workspace
    python examples/11_simulated_two_level_mpc_planner.py --seed 7

Outputs land in ``workspace/examples/outputs/11_two_level_mpc_planner/``:

- ``<scenario>.mp4`` — animated playback (lexicone visualiser).
- ``<scenario>_summary.png`` — episode totals + heatmap.
- ``<scenario>_log.csv`` — per-tick rule evaluations.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

# Make the workspace package and the DevContainers demo helpers importable.
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
DEMOS_ROOT = Path("/workspace/nuplan-project/DevContainers/demos").resolve()
DEVKIT_ROOT = Path(os.environ.get("NUPLAN_DEVKIT_ROOT", "/workspace/nuplan-devkit")).resolve()
PLANNER_CONFIG_DIR = WORKSPACE_ROOT / "lexicone" / "planning" / "config"

for path in (DEVKIT_ROOT, WORKSPACE_ROOT, DEMOS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common.nuplan_demo_utils import (  # noqa: E402  (sys.path mutation above)
    DEFAULT_DATA_ROOT,
    DEFAULT_EXP_ROOT,
    ensure_output_dir,
    find_db_files,
    list_tables,
)

from lexicone.observer import RuleEngine  # noqa: E402
from lexicone.observer.simulation_log_adapter import NuPlanSimulationLogSource  # noqa: E402

from examples.visualizer import render_episode  # noqa: E402


# Dynamic but well-behaved scenario types: the ego is moving at a normal speed
# on a single carriageway, optionally interacting with a lead vehicle. We bias
# toward these because pure intersection / opposing-lane scenarios stress the
# rule book heavily without a behavioural layer (lane-change selection,
# right-of-way arbitration). They produce a clean playback that highlights the
# core motion-planning loop without obscuring it with semantic edge cases.
DYNAMIC_SCENARIO_TYPES = [
    "following_lane_with_lead",
    "following_lane_with_slow_lead",
    "following_lane_without_lead",
    "behind_long_vehicle",
    "near_multiple_vehicles",
    "medium_magnitude_speed",
    "high_magnitude_speed",
]


def db_is_readable(db_path: Path) -> bool:
    """Skip DBs that fail a basic schema probe — at least one mini DB is corrupt
    on the bundled dataset; pointing the simulator at the full set would crash."""
    try:
        with sqlite3.connect(str(db_path)) as con:
            tables = list_tables(con)
            return "lidar_pc" in tables and "ego_pose" in tables and "scenario_tag" in tables
    except Exception:
        return False


def find_readable_log_stems(data_root: Path, max_logs: int = 12) -> list[str]:
    """Return up to ``max_logs`` readable mini-split log stems.

    We cap the list because pushing 40+ log names through Hydra's CLI override
    sometimes results in the filter returning zero scenarios — a dozen logs
    gives plenty of variety while staying well within the safe size range.
    """
    db_files = find_db_files(data_root, None)
    eligible = [db for db in db_files if db_is_readable(db)]
    if not eligible:
        raise RuntimeError(
            "No readable mini DBs were found. "
            "Run scripts/bootstrap_nuplan.sh --profile planner_mini."
        )
    return [db.stem for db in eligible[:max_logs]]


def find_latest_simulation_log(experiment_dir: Path) -> Path:
    candidates = list(experiment_dir.rglob("*.msgpack.xz")) + list(experiment_dir.rglob("*.pkl.xz"))
    if not candidates:
        raise FileNotFoundError(f"No simulation logs found under {experiment_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def run_planner_simulation(
    *,
    experiment_name: str,
    log_names: list[str],
    mpc_horizon_s: float,
    mpc_dt_s: float,
    replan_period_s: float,
    desired_speed_mps: float,
    seed: int,
) -> Path:
    exp_root = Path(os.environ.get("NUPLAN_EXP_ROOT", str(DEFAULT_EXP_ROOT)))
    # Hydra's ``searchpath`` accepts file:// URIs. Adding our config dir lets
    # ``planner=two_level_mpc_planner`` resolve to our YAML.
    searchpath_override = (
        f"hydra.searchpath=[file://{PLANNER_CONFIG_DIR},"
        "pkg://nuplan.planning.script.config.common,"
        "pkg://nuplan.planning.script.experiments]"
    )
    scenario_types_csv = "[" + ",".join(DYNAMIC_SCENARIO_TYPES) + "]"
    log_names_csv = "[" + ",".join(log_names) + "]"
    command = [
        sys.executable,
        "nuplan/planning/script/run_simulation.py",
        # Reactive IDM agents respond to the ego — without this, recorded traffic
        # is adversarial when our planner drives at a different speed than the
        # expert and we end up rear-ended or rear-ending recorded vehicles.
        "+simulation=closed_loop_reactive_agents",
        "planner=two_level_mpc_planner",
        f"planner.two_level_mpc_planner.mpc_horizon_s={mpc_horizon_s}",
        f"planner.two_level_mpc_planner.mpc_dt_s={mpc_dt_s}",
        f"planner.two_level_mpc_planner.replan_period_s={replan_period_s}",
        f"planner.two_level_mpc_planner.desired_speed_mps={desired_speed_mps}",
        "scenario_builder=nuplan_mini",
        # Override the default scenario_filter inline so we pick an actively-driving scenario
        # across the whole mini split (no log pinning — different logs hold different scenario types).
        "scenario_filter=all_scenarios",
        f"scenario_filter.scenario_types={scenario_types_csv}",
        f"scenario_filter.log_names={log_names_csv}",
        "scenario_filter.limit_total_scenarios=1",
        "scenario_filter.shuffle=true",
        f"seed={seed}",
        "worker=sequential",
        f"experiment_name={experiment_name}",
        f"job_name={experiment_name}",
        searchpath_override,
    ]
    env = os.environ.copy()
    # nuPlan instantiates the planner via _target_; our import path must resolve.
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join([str(WORKSPACE_ROOT)] + ([existing_pp] if existing_pp else []))

    print(f"[demo] simulation command: {' '.join(command)}")
    subprocess.run(command, cwd=str(DEVKIT_ROOT), check=True, env=env)
    return exp_root / "exp" / experiment_name / experiment_name


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "11_two_level_mpc_planner",
    )
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--margin", type=float, default=80.0, help="Map view margin around ego (m).")
    parser.add_argument("--radius", type=float, default=80.0, help="Map-query radius around the ego (m).")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--mpc-horizon-s", type=float, default=2.0)
    parser.add_argument("--mpc-dt-s", type=float, default=0.1)
    parser.add_argument("--replan-period-s", type=float, default=5.0)
    parser.add_argument("--desired-speed-mps", type=float, default=12.0)
    parser.add_argument(
        "--scenario-name",
        type=str,
        default=None,
        help="Override the label used in artefact filenames (default: log stem).",
    )
    args = parser.parse_args()

    out_dir = ensure_output_dir(args.output_dir)
    experiment_name = f"demo_11_two_level_mpc_v{args.desired_speed_mps:.1f}"
    readable_logs = find_readable_log_stems(args.data_root)
    print(f"[demo] {len(readable_logs)} readable mini DB(s) found.")

    print("[demo] two-level (global route + CasADi MPC) planner configuration:")
    print(f"[demo]   mpc_horizon_s:     {args.mpc_horizon_s}")
    print(f"[demo]   mpc_dt_s:          {args.mpc_dt_s}")
    print(f"[demo]   replan_period_s:   {args.replan_period_s}")
    print(f"[demo]   desired_speed_mps: {args.desired_speed_mps}")
    print(f"[demo]   seed:              {args.seed}")
    print(f"[demo]   experiment:        {experiment_name}")
    print("[demo] scenario filter shuffles across the readable mini DBs and picks one dynamic scenario.")

    experiment_dir = run_planner_simulation(
        experiment_name=experiment_name,
        log_names=readable_logs,
        mpc_horizon_s=args.mpc_horizon_s,
        mpc_dt_s=args.mpc_dt_s,
        replan_period_s=args.replan_period_s,
        desired_speed_mps=args.desired_speed_mps,
        seed=args.seed,
    )

    simulation_log_path = find_latest_simulation_log(experiment_dir)
    print(f"[demo] simulation log:               {simulation_log_path}")

    print("[demo] converting simulation log → SceneSnapshots …")
    source = NuPlanSimulationLogSource.from_path(simulation_log_path, radius_m=args.radius)
    snapshots = list(source)
    if not snapshots:
        print("ERROR: no snapshots produced from the simulation log.", file=sys.stderr)
        return 5
    scenario_name = args.scenario_name or simulation_log_path.stem
    print(f"[demo]   {len(snapshots)} snapshots → rendering via examples/visualizer.py …")

    engine = RuleEngine()
    artefacts = render_episode(
        engine=engine,
        snapshots=snapshots,
        scenario_name=scenario_name,
        output_dir=out_dir,
        fps=args.fps,
        map_margin_m=args.margin,
    )

    summary = engine.summary()
    print(
        f"\n== {scenario_name} ({len(snapshots)} ticks, {summary.duration_s:.1f}s)  planner=TwoLevelMPCPlanner =="
    )
    rows = sorted(
        (s for s in summary.rule_summaries.values() if s.n_steps_violated > 0),
        key=lambda s: -s.integrated_violation,
    )
    if not rows:
        print("  (no rule violated across the log)")
    for s in rows:
        print(
            f"  {s.rule_id:<6} L{s.rule_level:<2}  "
            f"viol_ticks={s.n_steps_violated:>3}/{s.n_steps_applicable:<3}  "
            f"max_rate={s.max_violation_rate:6.3f}  "
            f"integrated={s.integrated_violation:7.3f}   "
            f"{s.rule_name}"
        )
    print("\nArtefacts:")
    for kind, path in artefacts.items():
        print(f"  {kind:<7} {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
