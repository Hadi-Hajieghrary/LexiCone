#!/usr/bin/env python3
"""Demo 01 — Ego rolls through a red light at a signalised intersection.

Scenario
--------
- 30 m straight entry lane → 10 m lane connector through a 10×10 m
  intersection → 30 m exit lane, plus a north-south cross lane.
- A marked crosswalk sits across the entry approach.
- The lane connector's traffic light is RED.
- A stop polyline is associated with the entry lane.
- Two cross-traffic vehicles wait on the perpendicular axis.
- Ego decelerates only mildly and crosses the stop line at ~12 m/s.

Expected rule activity
----------------------
- ``7r1`` (traffic-light compliance) — fires while the ego is inside the
  intersection under a RED connector.
- ``0r2`` (longitudinal comfort) — one-tick flag from the deceleration
  ramping in (the finite-difference jerk spikes when ax steps from 0 to
  ``-1.5`` m/s² between ticks).
- ``3r0`` (speed limit) — *should not* fire (lane limit 13.4 m/s, ego peaks
  at 14 m/s but within tolerance). Useful as a sanity check.
- ``8r0`` (mandatory stop) — *does not* fire here on purpose: by design the
  rule excludes ``TRAFFIC_LIGHT`` stop lines, deferring to 7r1.

Output
------
Writes the MP4, episode summary PNG, and per-tick CSV log to
``workspace/examples/outputs/01_intersection_red_light/``.

Usage
-----
    cd workspace
    python examples/01_intersection_red_light.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make ``lexicone`` and sibling ``examples`` importable when the script is
# invoked directly (e.g. ``python examples/01_intersection_red_light.py``).
_WORKSPACE = Path(__file__).resolve().parent.parent
if str(_WORKSPACE) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE))

from lexicone.observer import RuleEngine

from examples.scenarios import intersection_red_light_episode
from examples.visualizer import render_episode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "01_intersection_red_light",
    )
    parser.add_argument("--ticks", type=int, default=60, help="Number of ticks (default: 60).")
    parser.add_argument("--fps", type=int, default=10, help="MP4 frames per second.")
    parser.add_argument("--margin", type=float, default=40.0, help="Map view margin around ego (m).")
    args = parser.parse_args()

    episode = intersection_red_light_episode(n_ticks=args.ticks)
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
    print("\nRules with non-zero integrated violation:")
    rows = sorted(
        (s for s in summary.rule_summaries.values() if s.n_steps_violated > 0),
        key=lambda s: -s.integrated_violation,
    )
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
