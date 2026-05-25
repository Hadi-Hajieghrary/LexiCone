#!/usr/bin/env python3
"""Demo 09 — Closed-loop urban scene with a dynamically crossing pedestrian.

Scenario
--------
- A 200 m east-bound corridor with a travel lane (y=0, 25 mph limit), a
  bike lane (y=2.5, no posted limit), and walkways on both sides.
- A marked crosswalk at x=80, spanning the full road width.
- **Dynamic pedestrian**: starts on the south walkway at (80, -3.5),
  waits until t=4 s, then walks north across the crosswalk at 1.4 m/s,
  arriving at the north walkway at t≈9 s.
- **Cyclist**: in the bike lane, constant 3 m/s, starting at x=25.
- Ego: starts at x=0 at 8 m/s; route is the single travel lane.

Planner
-------
- :class:`UrbanDrivingPlanner` — IDM longitudinal control plus a
  pedestrian-yield term that brakes when any VRU is on or within
  ``ped_buffer_m`` of a crosswalk in the forward lookahead corridor.

Expected rule activity
----------------------
- ``8r1`` (crosswalk pedestrian yield) — applies while pedestrian is on
  the crosswalk; *should not* fire because ego stops in time.
- ``10r0`` (VRU collision) — *should not* fire (ego stops well clear).
- ``7r4`` (don't stop in crosswalks) — *should not* fire (planner stops
  upstream of the crosswalk, accounting for ego half-length).
- ``3r3`` (safe headway) — may apply against the cyclist but should not
  violate at the in-lane projection (cyclist is in the bike lane).
- ``0r2`` (longitudinal comfort) — fires during the braking ramp and
  the launch from rest.

Output
------
``workspace/examples/outputs/09_simulated_urban_crossing/``.
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
    CrosswalkSnapshot,
    DrivableAreaSnapshot,
    Pose2D,
    WalkwaySnapshot,
)

from examples.planners import UrbanDrivingPlanner
from examples.scenarios import _rect, _straight_lane
from examples.simulation import (
    World,
    constant_velocity_agent,
    crossing_pedestrian,
    initial_ego,
    simulate,
)
from examples.visualizer import render_episode


def _build_world() -> World:
    travel_lane = _straight_lane(
        "travel", y_center=0.0, length=300.0, width=3.5, speed_limit_mps=11.0
    )
    bike_lane = _straight_lane(
        "bike",
        y_center=2.5,
        length=300.0,
        width=1.5,
        speed_limit_mps=None,
        is_bike_lane=True,
    )
    drivable = DrivableAreaSnapshot(polygon=_rect(0.0, 1.0, 300.0, 7.0))
    walkway_south = WalkwaySnapshot("ww_s", polygon=_rect(100.0, -4.5, 300.0, 3.0))
    walkway_north = WalkwaySnapshot("ww_n", polygon=_rect(100.0, 6.5, 300.0, 3.0))

    crosswalk = CrosswalkSnapshot("cw_main", polygon=_rect(80.0, 1.0, 4.0, 8.0))

    cyclist = AgentSnapshot(
        track_id="cyclist",
        object_type=AgentType.BICYCLE,
        pose=Pose2D(x=25.0, y=2.5, heading=0.0),
        vx=3.0,
        vy=0.0,
        length=1.7,
        width=0.6,
    )

    ped = crossing_pedestrian(
        track_id="ped_alice",
        x=80.0,
        # Start on the south walkway, end well past the north walkway so the
        # inflated crosswalk no longer contains the pedestrian after they
        # finish crossing (planner releases the yield).
        y_start=-3.5,
        y_end=7.0,
        speed_mps=1.4,
        t_start_s=4.0,
    )

    return World(
        lanes=(travel_lane,),
        bike_lanes=(bike_lane,),
        crosswalks=(crosswalk,),
        drivable_area=(drivable,),
        walkways=(walkway_south, walkway_north),
        scripted_agents=(constant_velocity_agent(cyclist), ped),
        route_lane_ids=("travel",),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "09_simulated_urban_crossing",
    )
    parser.add_argument("--ticks", type=int, default=180, help="Number of ticks at 10 Hz.")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--margin", type=float, default=55.0)
    args = parser.parse_args()

    world = _build_world()
    planner = UrbanDrivingPlanner(desired_speed_mps=11.0)
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
        scenario_name="urban_crossing",
        output_dir=args.output_dir,
        fps=args.fps,
        map_margin_m=args.margin,
    )

    summary = engine.summary()
    print(
        f"\n== urban_crossing ({len(snapshots)} ticks, {summary.duration_s:.1f}s) "
        f"planner={planner.name} =="
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
