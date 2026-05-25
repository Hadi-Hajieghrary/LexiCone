#!/usr/bin/env python3
"""Demo 07 — Simulated overtake of a slow lead vehicle.

Scenario
--------
- Two parallel east-bound lanes (right travel lane at ``y=0``, passing lane
  at ``y=3.5``), both with a 11 m/s posted limit. Walkways on both sides.
- A slow leader cruises in the right lane at 4 m/s.
- Ego starts behind the leader in the right lane at 8 m/s.

Planner
-------
- :class:`OvertakePlanner` — a small five-state machine: approach → merge
  left → pass → merge right → cruise. Uses a cascade of P controllers
  (lateral error → target heading → yaw rate, plus P-on-speed for ax).
  Respects the posted limit.

Expected rule activity
----------------------
- ``3r3`` (safe headway) — fires briefly in *approach* while gap shrinks
  below 1.5 s before the merge begins.
- ``3r5`` (lateral clearance) — fires briefly when the ego is alongside
  the leader during *pass*, just inside the rule's 1 m dynamic threshold.
- ``1r11`` / ``0r3`` (lateral comfort) — fire transiently during the two
  merge phases from the yaw input.
- ``9r0`` (vehicle collision) — *should not* fire: the trigger gap and
  merge timing are set so the ego clears laterally before catching up.
- ``3r0`` (speed limit) — *should not* fire: the planner is capped at the
  lane limit.
- ``7r2`` / ``7r3`` (opposing lane / wrong-way) — *should not* fire: the
  passing lane shares the ego's direction.

Output
------
``workspace/examples/outputs/07_simulated_overtake/``.
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

from examples.planners import OvertakePlanner
from examples.scenarios import _rect, _straight_lane
from examples.simulation import World, constant_velocity_agent, initial_ego, simulate
from examples.visualizer import render_episode


def _build_world() -> World:
    right_lane = _straight_lane(
        "right", y_center=0.0, length=400.0, width=3.5, speed_limit_mps=11.0
    )
    left_lane = _straight_lane(
        "left", y_center=3.5, length=400.0, width=3.5, speed_limit_mps=11.0
    )
    # One drivable rectangle covering both lanes + a small shoulder.
    drivable = DrivableAreaSnapshot(polygon=_rect(0.0, 1.75, 400.0, 9.0))
    walkway_n = WalkwaySnapshot("ww_n", polygon=_rect(0.0, 7.5, 400.0, 3.0))
    walkway_s = WalkwaySnapshot("ww_s", polygon=_rect(0.0, -4.0, 400.0, 3.0))
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
        lanes=(right_lane, left_lane),
        drivable_area=(drivable,),
        walkways=(walkway_n, walkway_s),
        scripted_agents=(constant_velocity_agent(leader),),
        route_lane_ids=("right",),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "07_simulated_overtake",
    )
    parser.add_argument("--ticks", type=int, default=120)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--margin", type=float, default=60.0, help="Map view margin around ego (m).")
    args = parser.parse_args()

    world = _build_world()
    planner = OvertakePlanner()
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
        scenario_name="overtake",
        output_dir=args.output_dir,
        fps=args.fps,
        map_margin_m=args.margin,
    )

    summary = engine.summary()
    print(
        f"\n== overtake ({len(snapshots)} ticks, {summary.duration_s:.1f}s) "
        f"planner={planner.name}  final_phase={planner.phase} =="
    )
    rows = sorted(
        (s for s in summary.rule_summaries.values() if s.n_steps_violated > 0),
        key=lambda s: -s.integrated_violation,
    )
    if not rows:
        print("  (no rule violated — clean overtake)")
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
