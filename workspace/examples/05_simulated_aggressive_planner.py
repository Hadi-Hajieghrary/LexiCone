#!/usr/bin/env python3
"""Demo 05 — Same world template as demo 04, driven by an aggressive planner.

Re-uses the world layout from
:mod:`examples.04_simulated_idm_following` so the comparison is fair: same
lane, same leader, same posted limit. Only the planner changes.

Planner
-------
- :class:`AggressivePlanner` with ``target_speed=18 m/s`` (well above the
  11 m/s posted limit) and ``max_accel=2.5 m/s²``. It only brakes if it
  would otherwise collide within the next car length, so it routinely
  closes the gap below safe headway and eventually rear-ends the slow
  leader.

Collision handling
------------------
The simulator detects the rear-end collision and **freezes the ego** in
contact with the leader for the remainder of the episode. The rule engine
keeps evaluating, so ``9r0`` and ``3r3`` flag every post-collision tick —
this is the correct outcome (a rear-end crash is a sustained 9r0
violation, not a one-tick event).

Expected rule activity
----------------------
- ``3r0`` (speed limit) — sustained while ego is over the limit.
- ``3r3`` (safe headway) — fires as ego closes, then stays violated.
- ``0r2`` (longitudinal comfort) — fires from the hard acceleration and
  the collision impulse.
- ``9r0`` (vehicle collision) — fires from the impact tick onward.

Compare the resulting episode summary with demo 04: same scene, opposite
rule profile.

Output
------
``workspace/examples/outputs/05_simulated_aggressive_planner/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_WORKSPACE = Path(__file__).resolve().parent.parent
if str(_WORKSPACE) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE))

from lexicone.observer import RuleEngine

from examples.planners import AggressivePlanner

# Import the world builder directly so the two demos share the exact map.
import importlib.util

_demo04_path = Path(__file__).resolve().parent / "04_simulated_idm_following.py"
_spec = importlib.util.spec_from_file_location("_sim_demo04", _demo04_path)
_sim_demo04 = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
assert _spec and _spec.loader
_spec.loader.exec_module(_sim_demo04)  # type: ignore[union-attr]
_build_world = _sim_demo04._build_world  # noqa: SLF001 — intentional reuse

from examples.simulation import initial_ego, simulate
from examples.visualizer import render_episode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "05_simulated_aggressive_planner",
    )
    parser.add_argument("--ticks", type=int, default=80)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--margin", type=float, default=50.0, help="Map view margin around ego (m).")
    args = parser.parse_args()

    world = _build_world()
    planner = AggressivePlanner(target_speed_mps=18.0, max_accel_mps2=2.5, lateral_drift_radps=0.0)
    snapshots = simulate(
        world=world,
        planner=planner,
        initial=initial_ego(x=0.0, y=0.0, heading=0.0, speed=8.0),
        n_ticks=args.ticks,
    )
    engine = RuleEngine()
    artefacts = render_episode(
        engine=engine,
        snapshots=snapshots,
        scenario_name="aggressive_planner",
        output_dir=args.output_dir,
        fps=args.fps,
        map_margin_m=args.margin,
    )

    summary = engine.summary()
    print(f"\n== aggressive_planner ({len(snapshots)} ticks, {summary.duration_s:.1f}s) planner={planner.name} ==")
    rows = sorted(
        (s for s in summary.rule_summaries.values() if s.n_steps_violated > 0),
        key=lambda s: -s.integrated_violation,
    )
    if not rows:
        print("  (no rule violated)")
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
