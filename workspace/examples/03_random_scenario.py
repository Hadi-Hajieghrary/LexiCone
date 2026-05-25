#!/usr/bin/env python3
"""Demo 03 — Reproducible random scenario.

Given a ``--seed``, this generator builds a straight-road episode with a
random mix of map features (walkways, bike lanes, crosswalks) and agents
(leader vehicle, lateral neighbour, cyclist, pedestrian). The ego's target
speed, lateral drift, and longitudinal acceleration are also randomised.
Most seeds end up tripping a different mix of rules.

Run multiple seeds with ``--seeds 0 1 2 3 …`` to compare what different
random situations look like — each seed writes its own subdirectory.

Output
------
Writes one MP4, one summary PNG, and one CSV log per seed under
``workspace/examples/outputs/03_random_scenario/seed_<N>/``.

Usage
-----
    cd workspace
    python examples/03_random_scenario.py --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

_WORKSPACE = Path(__file__).resolve().parent.parent
if str(_WORKSPACE) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE))

from lexicone.observer import RuleEngine

from examples.scenarios import random_episode
from examples.visualizer import render_episode


def run_one(seed: int, n_ticks: int, fps: int, margin: float, root: Path) -> None:
    episode = random_episode(seed=seed, n_ticks=n_ticks)
    engine = RuleEngine()
    out_dir = root / f"seed_{seed}"
    artefacts = render_episode(
        engine=engine,
        snapshots=episode.snapshots,
        scenario_name=episode.name,
        output_dir=out_dir,
        fps=fps,
        map_margin_m=margin,
    )

    summary = engine.summary()
    print(f"\n== seed={seed}  ({len(episode.snapshots)} ticks, {summary.duration_s:.1f}s) ==")
    rows = sorted(
        (s for s in summary.rule_summaries.values() if s.n_steps_violated > 0),
        key=lambda s: -s.integrated_violation,
    )
    if not rows:
        print("  (no rule violated in this run)")
    for s in rows:
        print(
            f"  {s.rule_id:<6} L{s.rule_level:<2}  "
            f"viol_ticks={s.n_steps_violated:>3}/{s.n_steps_applicable:<3}  "
            f"max_rate={s.max_violation_rate:6.3f}  "
            f"integrated={s.integrated_violation:7.3f}   "
            f"{s.rule_name}"
        )
    print("  artefacts:")
    for kind, path in artefacts.items():
        print(f"    {kind:<7} {path}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "03_random_scenario",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--ticks", type=int, default=60)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--margin", type=float, default=50.0, help="Map view margin around ego (m).")
    args = parser.parse_args(argv)

    for seed in args.seeds:
        run_one(seed=seed, n_ticks=args.ticks, fps=args.fps, margin=args.margin, root=args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
