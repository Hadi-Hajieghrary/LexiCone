# 02 — Near Long Vehicle

> **nuPlan scenario type.** `near_long_vehicle`
> **Behaviour class.** Overtake-style
> **Episode duration.** ~15 s (150 ticks)
> **Top observed violation.** `7r2` Opposing Lane (integrated $131.20$)
> **Total violating rules.** 12 of 25

## What happens

The ego is in close proximity to a long vehicle — a bus, truck, or articulated tractor — whose footprint dominates a substantial portion of the surrounding traffic envelope. nuPlan tags such instances as `near_long_vehicle` precisely because the geometry forces non-trivial lateral planning: the ego must give a wider berth than for a typical sedan, often crowding the lane edge or briefly cohabiting two lanes.

The dominant violation here is `7r2` Opposing Lane — the planner's MPC, in negotiating space around the long vehicle, briefly skirts or enters the polygon of an adjacent lane whose centreline heading is anti-parallel to ego's. The `OpposingLaneRule` encoder (Section V.E of [`../../../../References/comprehensive_report.md`](../../../../References/comprehensive_report.md)) tests for opposing-direction lane occupancy with oncoming traffic; the violation rate is the speed-weighted overlap area.

## Simulation playback

![02_near_long_vehicle](02_near_long_vehicle.gif)

> **How to watch.** Track the large green rectangle (the long vehicle) and the ego (red) around it. Note when the planned trajectory (dashed dark-blue) drifts toward the lane boundary — that's the moment the half-plane `LaneCorridorRule` and `OpposingLaneRule` constraints are competing for slack. The bottom strip's L7 (amber) row will pulse during the crowding manoeuvre.

Full resolution: [`02_near_long_vehicle.mp4`](02_near_long_vehicle.mp4). Summary: [`02_near_long_vehicle_summary.png`](02_near_long_vehicle_summary.png). Log: [`02_near_long_vehicle_log.csv`](02_near_long_vehicle_log.csv).

## What the LCP-WS-$L_1$ planner does

The MPC reads the long-vehicle footprint via the per-tick `obstacle_slot_count = 6` filter (closest 6 agents to the ego), inflated by `collision_buffer_m = 0.4`. To keep the per-agent circular keep-out constraint feasible, the planner shifts the trajectory laterally — but the `LaneCorridorRule`'s half-planes (the convex localisation of "stay on the route corridor") penalise the drift, while `OpposingLaneRule`'s half-planes penalise occupancy of the anti-parallel lane on the other side. The compromise the LCP $L_1$ weighted sum reaches is a brief, shallow lane-edge incursion that fires `7r2` at moderate rate over many ticks (hence the high integrated value).

## Top violations observed

| Rule | Level | Integrated | Why it fires |
|---|---|---:|---|
| `7r2` Opposing Lane | L7 | 131.20 | Ego footprint overlaps opposing-direction lane while skirting the long vehicle |
| `2r2` Route Adherence | L2 | ~45 | Observer-only; the lateral shift from the planned route corridor exceeds comfort tolerance |
| `3r3` Safe Headway | L3 | ~10 | Brief headway shortfalls when the long vehicle decelerates |
| `1r0` Yield Priority | L1 | ~25 | Observer-only; right-of-way ambiguity during the crowding |

## Files in this directory

- [`02_near_long_vehicle.mp4`](02_near_long_vehicle.mp4) — original MP4
- [`02_near_long_vehicle.gif`](02_near_long_vehicle.gif) — GIF embedded above
- [`02_near_long_vehicle_summary.png`](02_near_long_vehicle_summary.png) — episode summary plot
- [`02_near_long_vehicle_log.csv`](02_near_long_vehicle_log.csv) — per-tick CSV
- `README.md` — this file
