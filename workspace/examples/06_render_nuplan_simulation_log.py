#!/usr/bin/env python3
"""Demo 06 — Render a real NuPlan SimulationLog through the rule engine.

Unlike demos 04/05 (self-contained closed-loop sim), this demo expects a
saved nuPlan :class:`SimulationLog` on disk — the artefact produced by
``SimulationLogCallback`` (see e.g. DevContainers/demos/13_idm_simulation_and_record.py).

The pipeline is:

1. ``NuPlanSimulationLogSource.from_path(log_path)`` loads the log and
   exposes an iterator of :class:`SceneSnapshot`s (one per sample), using
   the scenario's ``map_api`` for proximal map queries.
2. The :class:`RuleEngine` steps over those snapshots exactly as in the
   self-contained demos.
3. The visualiser unions the per-tick map queries into a coherent world
   view, then writes the same MP4 + summary PNG + CSV triple.

Requirements
------------
- ``nuplan-devkit`` must be importable.
- A valid SimulationLog file path (msgpack.xz or pickle, depending on your
  simulation config).

Usage
-----
    cd workspace
    python examples/06_render_nuplan_simulation_log.py \\
        --log-path /workspace/exp/<your-experiment>/simulation_log/<scenario>.msgpack.xz \\
        --radius 80

The script gracefully prints a hint if nuplan-devkit is not installed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_WORKSPACE = Path(__file__).resolve().parent.parent
if str(_WORKSPACE) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE))

from lexicone.observer import RuleEngine

from examples.visualizer import render_episode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--log-path",
        type=Path,
        required=True,
        help="Path to a saved nuPlan SimulationLog (msgpack.xz or pickle).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "06_nuplan_simulation_log",
    )
    parser.add_argument("--radius", type=float, default=80.0, help="Map-query radius around the ego (m).")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--margin", type=float, default=80.0, help="Map view margin around ego (m).")
    parser.add_argument(
        "--scenario-name",
        type=str,
        default=None,
        help="Override the scenario label used in artefact filenames. Defaults to the log file stem.",
    )
    args = parser.parse_args()

    if not args.log_path.exists():
        print(f"ERROR: log file not found: {args.log_path}", file=sys.stderr)
        return 2

    try:
        from lexicone.observer.simulation_log_adapter import NuPlanSimulationLogSource
    except Exception as e:  # nuplan-devkit may itself be the import target failure.
        print(
            "ERROR: could not import the SimulationLog adapter. Is nuplan-devkit installed?\n"
            f"  {type(e).__name__}: {e}",
            file=sys.stderr,
        )
        return 3

    try:
        source = NuPlanSimulationLogSource.from_path(args.log_path, radius_m=args.radius)
    except Exception as e:
        print(f"ERROR: failed to load SimulationLog: {type(e).__name__}: {e}", file=sys.stderr)
        return 4

    print(f"Loaded SimulationLog with {len(source)} samples from {args.log_path}")
    print("Converting samples → SceneSnapshots (this also runs map_api proximal queries) …")
    snapshots = list(source)
    print(f"  done ({len(snapshots)} snapshots).")

    scenario_name = args.scenario_name or args.log_path.stem
    engine = RuleEngine()
    artefacts = render_episode(
        engine=engine,
        snapshots=snapshots,
        scenario_name=scenario_name,
        output_dir=args.output_dir,
        fps=args.fps,
        map_margin_m=args.margin,
    )

    summary = engine.summary()
    print(f"\n== {scenario_name} ({len(snapshots)} ticks, {summary.duration_s:.1f}s) ==")
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
