#!/usr/bin/env python3
"""Demo 02 — Marginal cyclist overtake with bike-lane drift.

Scenario
--------
- A straight two-lane corridor (one vehicle lane + one bike lane) with a
  posted limit of 25 mph (≈11 m/s) and sidewalks on both sides.
- A cyclist rides in the bike lane at 3 m/s.
- The ego travels at ~12.5 m/s (over the limit) and drifts toward the bike
  lane while passing, leaving a narrow lateral gap.

Expected rule activity
----------------------
- ``3r0`` (speed limit) — ego is over the posted limit, expect a sustained
  violation.
- ``10r4`` (cyclist passing) — fires while the ego is alongside the cyclist
  with marginal lateral clearance.
- ``10r5`` (bike-lane encroachment) — fires once the ego footprint touches
  the bike-lane polygon.
- ``0r3`` (lateral comfort) — light flag from the drift's ay.

Output
------
Writes the MP4, episode summary PNG, and per-tick CSV log to
``workspace/examples/outputs/02_cyclist_overtake/``.

Usage
-----
    cd workspace
    python examples/02_cyclist_overtake.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_WORKSPACE = Path(__file__).resolve().parent.parent
if str(_WORKSPACE) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE))

from lexicone.observer import RuleEngine

from examples.scenarios import cyclist_overtake_episode
from examples.visualizer import render_episode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "02_cyclist_overtake",
    )
    parser.add_argument("--ticks", type=int, default=50)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--margin", type=float, default=40.0, help="Map view margin around ego (m).")
    args = parser.parse_args()

    episode = cyclist_overtake_episode(n_ticks=args.ticks)
    engine = RuleEngine()
    artefacts = render_episode(
        engine=engine,
        snapshots=episode.snapshots,
        scenario_name=episode.name,
        output_dir=args.output_dir,
        fps=args.fps,
        map_margin_m=args.margin,
    )

    summary = engine.summary()
    print(f"\n== {episode.name} ({len(episode.snapshots)} ticks, {summary.duration_s:.1f}s) ==")
    rows = sorted(
        (s for s in summary.rule_summaries.values() if s.n_steps_violated > 0),
        key=lambda s: -s.integrated_violation,
    )
    print("\nRules with non-zero integrated violation:")
    if not rows:
        print("  (none)")
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
