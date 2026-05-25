#!/usr/bin/env python3
"""Demo 08 — Closed-loop signalized 4-way intersection.

Scenario
--------
- Full 4-way signalized intersection: east-bound corridor (entry → lane
  connector → exit) crosses a north-bound corridor (entry → lane connector
  → exit). 25 mph (≈11 m/s) limit on all approaches.
- Drivable area is a ``+``-shape covering both corridors; walkways line
  the four outer quadrants.
- Two marked crosswalks: one across the east approach, one across the
  north approach.
- **Cycling traffic light** — E-W connector runs green→yellow→red on a
  9-second cycle; N-S connector is 180° out of phase. So if ego arrives
  on red it must wait, then proceed on green.
- Cross-traffic: two N-S vehicles starting upstream of the intersection,
  moving steadily — they cross the box during their green phase, sit
  through E-W green.
- Ego starts 60 m west of the box at 10 m/s; route is the east-bound
  triple.

Planner
-------
- :class:`UrbanDrivingPlanner` — IDM longitudinal control plus
  traffic-light awareness and pedestrian yield. The TL term commands a
  hard-but-comfortable decel when a RED is in the lookahead corridor.

Expected rule activity
----------------------
- ``7r1`` (traffic-light compliance) — *should not* fire: the planner
  stops cleanly before the connector entry on red.
- ``8r0`` (mandatory stop) — *should not* fire: the stop polyline is
  TRAFFIC_LIGHT type, handled by 7r1.
- ``0r2`` (longitudinal comfort) — fires transiently during the brake-to-
  stop ramp and the launch from rest.
- ``3r3`` (safe headway) — may fire briefly if the planner closes inside
  1.5 s headway as it approaches the line; the deceleration term should
  pull headway back.
- ``1r0`` (yield priority) — may fire from the cross-traffic agents while
  they are inside the priority box during the ego's wait.
- ``9r0``/``10r0``/``2r2``/``3r0`` — should stay quiet.

Output
------
``workspace/examples/outputs/08_simulated_signalized_intersection/``.
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
    IntersectionSnapshot,
    Pose2D,
    StopLineSnapshot,
    StopType,
    WalkwaySnapshot,
)

from examples.planners import UrbanDrivingPlanner
from examples.scenarios import _rect, _straight_lane
from examples.simulation import (
    World,
    combine_traffic_light_schedules,
    constant_velocity_agent,
    initial_ego,
    simulate,
    traffic_light_cycle,
)
from examples.visualizer import render_episode


def _build_world() -> World:
    # East-west corridor: entry (-80→-5), connector (-5→+5), exit (+5→+80).
    ew_entry = _straight_lane(
        "ew_entry",
        x_center=-42.5,
        length=75.0,
        width=3.5,
        speed_limit_mps=11.0,
        outgoing=("ew_connector",),
    )
    ew_connector = _straight_lane(
        "ew_connector",
        x_center=0.0,
        length=10.0,
        width=3.5,
        speed_limit_mps=11.0,
        is_lane_connector=True,
        incoming=("ew_entry",),
        outgoing=("ew_exit",),
    )
    ew_exit = _straight_lane(
        "ew_exit",
        x_center=42.5,
        length=75.0,
        width=3.5,
        speed_limit_mps=11.0,
        incoming=("ew_connector",),
    )

    # North-south corridor through the same intersection box.
    ns_entry = _straight_lane(
        "ns_entry",
        x_center=0.0,
        y_center=-42.5,
        length=75.0,
        width=3.5,
        heading=math.pi / 2.0,
        speed_limit_mps=11.0,
        outgoing=("ns_connector",),
    )
    ns_connector = _straight_lane(
        "ns_connector",
        x_center=0.0,
        y_center=0.0,
        length=10.0,
        width=3.5,
        heading=math.pi / 2.0,
        speed_limit_mps=11.0,
        is_lane_connector=True,
        incoming=("ns_entry",),
        outgoing=("ns_exit",),
    )
    ns_exit = _straight_lane(
        "ns_exit",
        x_center=0.0,
        y_center=42.5,
        length=75.0,
        width=3.5,
        heading=math.pi / 2.0,
        speed_limit_mps=11.0,
        incoming=("ns_connector",),
    )

    intersection = IntersectionSnapshot(
        intersection_id="int_main",
        polygon=_rect(0.0, 0.0, 10.0, 10.0),
        is_signalized=True,
    )

    # `+`-shaped drivable: horizontal corridor ∪ vertical corridor.
    drivable_ew = DrivableAreaSnapshot(polygon=_rect(0.0, 0.0, 160.0, 7.0))
    drivable_ns = DrivableAreaSnapshot(polygon=_rect(0.0, 0.0, 7.0, 90.0))

    # Walkways: the four quadrants outside the drivable corridors.
    walkways = (
        WalkwaySnapshot("ww_nw", polygon=_rect(-40.0, 30.0, 60.0, 25.0)),
        WalkwaySnapshot("ww_ne", polygon=_rect(40.0, 30.0, 60.0, 25.0)),
        WalkwaySnapshot("ww_sw", polygon=_rect(-40.0, -30.0, 60.0, 25.0)),
        WalkwaySnapshot("ww_se", polygon=_rect(40.0, -30.0, 60.0, 25.0)),
    )

    # Two marked crosswalks sitting just outside the intersection box on the
    # approach side. The stop polylines (next block) are placed further
    # upstream so the ego stops *before* the crosswalk, never on it.
    crosswalks = (
        CrosswalkSnapshot("cw_west", polygon=_rect(-6.25, 0.0, 1.5, 8.0)),
        CrosswalkSnapshot("cw_south", polygon=_rect(0.0, -6.25, 8.0, 1.5)),
    )

    # Stop polylines (TRAFFIC_LIGHT type) — placed upstream of the crosswalks
    # so the planner's stop target sits behind the crosswalk in the driving
    # direction. 7r1 handles signal compliance violations.
    stop_lines = (
        StopLineSnapshot(
            stop_line_id="sl_ew",
            polyline=[(-9.0, -3.5), (-9.0, 3.5)],
            stop_type=StopType.TRAFFIC_LIGHT,
            associated_lane_id="ew_entry",
        ),
        StopLineSnapshot(
            stop_line_id="sl_ns",
            polyline=[(-3.5, -9.0), (3.5, -9.0)],
            stop_type=StopType.TRAFFIC_LIGHT,
            associated_lane_id="ns_entry",
        ),
    )

    # Traffic-light schedule: ew red at t=0 (ego must wait), ns green at t=0
    # (cross traffic passes through). Period 9 s; placing ew's phase
    # ``green_s + yellow_s`` into the cycle starts it in RED.
    green_s, yellow_s, red_s = 4.0, 1.0, 4.0
    ew_schedule = traffic_light_cycle(
        "ew_connector",
        green_s=green_s,
        yellow_s=yellow_s,
        red_s=red_s,
        phase_offset_s=green_s + yellow_s,  # t=0 lands at start of RED
    )
    ns_schedule = traffic_light_cycle(
        "ns_connector",
        green_s=green_s,
        yellow_s=yellow_s,
        red_s=red_s,
        # 180° out of phase: when ew is red, ns is green.
        phase_offset_s=green_s + yellow_s + (green_s + yellow_s + red_s) / 2.0,
    )

    # Cross traffic: two N-S vehicles, spaced so they enter the box on green.
    cross_v1 = AgentSnapshot(
        track_id="cross_v1",
        object_type=AgentType.VEHICLE,
        pose=Pose2D(x=0.0, y=-30.0, heading=math.pi / 2.0),
        vx=0.0,
        vy=6.0,
        length=4.5,
        width=1.8,
    )
    cross_v2 = AgentSnapshot(
        track_id="cross_v2",
        object_type=AgentType.VEHICLE,
        pose=Pose2D(x=0.0, y=-55.0, heading=math.pi / 2.0),
        vx=0.0,
        vy=6.0,
        length=4.5,
        width=1.8,
    )

    return World(
        lanes=(ew_entry, ew_exit, ns_entry, ns_exit),
        lane_connectors=(ew_connector, ns_connector),
        crosswalks=crosswalks,
        stop_lines=stop_lines,
        intersections=(intersection,),
        drivable_area=(drivable_ew, drivable_ns),
        walkways=walkways,
        traffic_lights=combine_traffic_light_schedules(ew_schedule, ns_schedule),
        scripted_agents=(
            constant_velocity_agent(cross_v1),
            constant_velocity_agent(cross_v2),
        ),
        route_lane_ids=("ew_entry", "ew_connector", "ew_exit"),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "08_simulated_signalized_intersection",
    )
    parser.add_argument("--ticks", type=int, default=150, help="Number of ticks at 10 Hz.")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--margin", type=float, default=55.0)
    args = parser.parse_args()

    world = _build_world()
    planner = UrbanDrivingPlanner(desired_speed_mps=11.0)
    snapshots = simulate(
        world=world,
        planner=planner,
        initial=initial_ego(x=-60.0, y=0.0, heading=0.0, speed=10.0),
        n_ticks=args.ticks,
    )

    engine = RuleEngine()
    artefacts = render_episode(
        engine=engine,
        snapshots=snapshots,
        scenario_name="signalized_intersection",
        output_dir=args.output_dir,
        fps=args.fps,
        map_margin_m=args.margin,
    )

    summary = engine.summary()
    print(
        f"\n== signalized_intersection ({len(snapshots)} ticks, {summary.duration_s:.1f}s) "
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
