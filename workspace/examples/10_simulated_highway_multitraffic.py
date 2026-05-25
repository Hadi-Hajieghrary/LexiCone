#!/usr/bin/env python3
"""Demo 10 — Closed-loop 3-lane highway with dense, multi-agent traffic.

Scenario
--------
- A 400 m east-bound 3-lane highway. Centerlines at ``y=0`` (right lane),
  ``y=3.5`` (centre), ``y=7.0`` (left/passing). 65 mph (≈29 m/s) posted
  limit on every lane. Walkways on both road edges.
- A slow truck in the ego's lane (centre, y=3.5) at 6 m/s, starting 35 m
  ahead.
- A faster vehicle in the left/passing lane catching up from behind
  (x=-25, v=18 m/s) — the planner must let it pass before initiating the
  lane change.
- A slower vehicle further ahead in the right lane (x=55, v=8 m/s) — not
  a viable overtake target.

Ego starts at x=0, ``y=3.5`` (centre lane), v=10 m/s.

Planner
-------
- :class:`LaneChangePlanner` — six-state machine
  ``follow → prepare_lane_change → merge → pass → return → follow``.
  Trigger: in-lane THW < 2 s against the slow truck. The merge only
  fires once the *target* lane is verified clear in a 30 m corridor
  around the ego; until then the planner stays behind the truck under
  IDM control.

Expected rule activity
----------------------
- ``3r3`` (safe headway) — fires during the follow phase while the ego
  closes on the slow truck.
- ``2r2`` (route adherence) — fires while the ego is in the left lane
  (route is the centre lane).
- ``0r3`` (lateral comfort) — brief peaks at the merge S-curve corners.
- ``3r5`` (lateral clearance) — may fire briefly while alongside the
  truck during the pass.
- ``9r0`` (vehicle collision) — *should not* fire (the gap check
  defers the merge until the fast left-lane traffic has cleared).

Output
------
``workspace/examples/outputs/10_simulated_highway_multitraffic/``.
"""

from __future__ import annotations

import argparse
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
    Pose2D,
    WalkwaySnapshot,
)

from examples.planners import LaneChangePlanner
from examples.scenarios import _rect, _straight_lane
from examples.simulation import World, constant_velocity_agent, initial_ego, simulate
from examples.visualizer import render_episode


def _build_world() -> World:
    right = _straight_lane("right", y_center=0.0, length=500.0, width=3.5, speed_limit_mps=29.0)
    center = _straight_lane("center", y_center=3.5, length=500.0, width=3.5, speed_limit_mps=29.0)
    left = _straight_lane("left", y_center=7.0, length=500.0, width=3.5, speed_limit_mps=29.0)

    drivable = DrivableAreaSnapshot(polygon=_rect(100.0, 3.5, 500.0, 12.0))
    walkway_s = WalkwaySnapshot("ww_s", polygon=_rect(100.0, -4.0, 500.0, 3.0))
    walkway_n = WalkwaySnapshot("ww_n", polygon=_rect(100.0, 11.0, 500.0, 3.0))

    truck = AgentSnapshot(
        track_id="truck",
        object_type=AgentType.VEHICLE,
        pose=Pose2D(x=35.0, y=3.5, heading=0.0),
        vx=6.0,
        vy=0.0,
        length=12.0,
        width=2.5,
    )
    fast_left = AgentSnapshot(
        track_id="fast_left",
        object_type=AgentType.VEHICLE,
        pose=Pose2D(x=-25.0, y=7.0, heading=0.0),
        vx=18.0,
        vy=0.0,
        length=4.5,
        width=1.8,
    )
    slow_right = AgentSnapshot(
        track_id="slow_right",
        object_type=AgentType.VEHICLE,
        pose=Pose2D(x=55.0, y=0.0, heading=0.0),
        vx=8.0,
        vy=0.0,
        length=4.5,
        width=1.8,
    )

    return World(
        lanes=(right, center, left),
        drivable_area=(drivable,),
        walkways=(walkway_s, walkway_n),
        scripted_agents=(
            constant_velocity_agent(truck),
            constant_velocity_agent(fast_left),
            constant_velocity_agent(slow_right),
        ),
        route_lane_ids=("center",),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "10_simulated_highway_multitraffic",
    )
    parser.add_argument("--ticks", type=int, default=200, help="Number of ticks at 10 Hz.")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--margin", type=float, default=70.0)
    args = parser.parse_args()

    world = _build_world()
    planner = LaneChangePlanner(
        cruise_speed_mps=13.0,
        overtake_speed_mps=15.0,
        # Merge to the left (y=+3.5 from centre at 3.5 → target y=7.0 left lane).
        lateral_offset_m=3.5,
    )
    snapshots = simulate(
        world=world,
        planner=planner,
        initial=initial_ego(x=0.0, y=3.5, heading=0.0, speed=10.0),
        n_ticks=args.ticks,
    )

    engine = RuleEngine()
    artefacts = render_episode(
        engine=engine,
        snapshots=snapshots,
        scenario_name="highway_multitraffic",
        output_dir=args.output_dir,
        fps=args.fps,
        map_margin_m=args.margin,
    )

    summary = engine.summary()
    print(
        f"\n== highway_multitraffic ({len(snapshots)} ticks, {summary.duration_s:.1f}s) "
        f"planner={planner.name}  final_phase={planner.phase} =="
    )
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
