#!/usr/bin/env python3
"""Demo 04 — Closed-loop IDM follower behind a slow lead vehicle.

The same closed-loop harness will drive every "simulated" demo: a small
:class:`World` (map + scripted agents) plus a :class:`Planner` whose
``plan(ctx)`` is called once per tick. The simulator advances the ego
kinematically from the command, builds the next :class:`SceneSnapshot`, and
hands the stream to the rule engine — the rule engine and the visualiser
don't know whether the ego was scripted or simulated.

Scenario
--------
- A 300 m east-bound travel lane with a 25 mph (≈11 m/s) limit and walkways
  on both sides.
- A leader vehicle 25 m ahead, holding 4 m/s.
- Ego starts at 8 m/s.

Planner
-------
- :class:`IDMPlanner` with ``desired_speed=11 m/s``, ``time_headway=1.5 s``,
  ``min_gap=2 m``, ``max_accel=1.5 m/s²``, ``comfort_decel=2 m/s²``. The
  planner respects the posted speed limit.

Expected rule activity
----------------------
- ``3r3`` (safe headway) — *should not* fire: IDM is designed to maintain
  ≥ 1.5 s THW.
- ``3r0`` (speed limit) — *should not* fire: IDM is capped at the limit.
- Comfort rules (0r2, 0r3, 1r11) may flag the initial deceleration ramp.

Output
------
Writes the MP4, summary PNG, and per-tick CSV to
``workspace/examples/outputs/04_simulated_idm_following/``.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

_WORKSPACE = Path(__file__).resolve().parent.parent
if str(_WORKSPACE) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE))

from lexicone.observer import RuleEngine
from lexicone.observer.types import (
    AgentSnapshot,
    AgentType,
    DrivableAreaSnapshot,
    LaneSnapshot,
    Pose2D,
    WalkwaySnapshot,
)

from examples.planners import IDMPlanner
from examples.scenarios import _rect, _straight_lane
from examples.simulation import World, constant_velocity_agent, initial_ego, simulate
from examples.visualizer import render_episode


def _build_world() -> World:
    lane = _straight_lane("main", length=300.0, width=3.5, speed_limit_mps=11.0)
    drivable = DrivableAreaSnapshot(polygon=_rect(0.0, 0.0, 300.0, 8.0))
    walkway_n = WalkwaySnapshot("ww_n", polygon=_rect(0.0, 5.5, 300.0, 3.0))
    walkway_s = WalkwaySnapshot("ww_s", polygon=_rect(0.0, -5.5, 300.0, 3.0))
    leader = AgentSnapshot(
        track_id="leader",
        object_type=AgentType.VEHICLE,
        pose=Pose2D(x=25.0, y=0.0, heading=0.0),
        vx=4.0,
        vy=0.0,
        length=4.5,
        width=1.8,
    )
    return World(
        lanes=(lane,),
        drivable_area=(drivable,),
        walkways=(walkway_n, walkway_s),
        scripted_agents=(constant_velocity_agent(leader),),
        route_lane_ids=("main",),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "04_simulated_idm_following",
    )
    parser.add_argument("--ticks", type=int, default=80)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--margin", type=float, default=50.0, help="Map view margin around ego (m).")
    args = parser.parse_args()

    world = _build_world()
    planner = IDMPlanner(desired_speed_mps=11.0)
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
        scenario_name=f"idm_following",
        output_dir=args.output_dir,
        fps=args.fps,
        map_margin_m=args.margin,
    )

    summary = engine.summary()
    print(f"\n== idm_following ({len(snapshots)} ticks, {summary.duration_s:.1f}s) planner={planner.name} ==")
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
