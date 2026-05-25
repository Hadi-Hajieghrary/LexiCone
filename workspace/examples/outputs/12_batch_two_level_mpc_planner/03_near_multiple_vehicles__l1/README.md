# 03 — Near Multiple Vehicles

> **nuPlan scenario type.** `near_multiple_vehicles`
> **Behaviour class.** Overtake-style (dense traffic)
> **Episode duration.** ~15 s (150 ticks)
> **Top observed violation.** `7r2` Opposing Lane (integrated $316.19$)
> **Total violating rules.** 10 of 25

## What happens

A dense-traffic moment: the ego is surrounded by several vehicles within its occupancy-map radius (`occupancy_map_radius_m = 40 m`). nuPlan tags these instances when the ego has three or more tracked agents close enough to interact with simultaneously. The geometry forces the MPC to manage multiple soft-collision constraints at once; each agent occupies a circular keep-out (`CollisionRule`'s 8 per-step slots, sized for the radius `0.5 · √(L² + W²) + collision_buffer_m`), and the level-1 (safety) slack matrix has to find feasible per-agent margins.

The dominant violation is again `7r2` — but here the magnitude (316) is more than double scenario 02's, indicating the ego spends a longer fraction of the episode skirting opposing lanes to keep the multi-agent collision constraints satisfied.

## Simulation playback

![03_near_multiple_vehicles](03_near_multiple_vehicles.gif)

> **How to watch.** Count the green rectangles around the ego — multiple lanes' worth of traffic. Notice the planned trajectory's lateral wobble as the MPC trades off between the per-agent collision slacks (L1) and the lane-occupancy slacks (L7). The bottom violation strip shows sustained amber (L7) and brief teal (L3 lateral clearance) episodes throughout.

Full resolution: [`03_near_multiple_vehicles.mp4`](03_near_multiple_vehicles.mp4). Summary: [`03_near_multiple_vehicles_summary.png`](03_near_multiple_vehicles_summary.png). Log: [`03_near_multiple_vehicles_log.csv`](03_near_multiple_vehicles_log.csv).

## What the LCP-WS-$L_1$ planner does

The LCP $L_1$ weighted-sum trades L1 (safety) against L3 (comfort / headway / lateral clearance) at the calibration weights $w_1 \gg w_2 \gg w_3$. With multiple agents in the occupancy map, the L1 slack vector is the binding cost component for most ticks; the trajectory shifts to keep every per-agent constraint feasible, accepting moderate L7 violations as the cost of safety. This is exactly the priority structure the LCP framework promises and the legacy flat-weight MPC does not provide — a key advantage the comparative protocol under [`../../13_protocol/`](../../13_protocol/) is designed to quantify.

## Top violations observed

| Rule | Level | Integrated | Why it fires |
|---|---|---:|---|
| `7r2` Opposing Lane | L7 | **316.19** | Multi-agent collision avoidance forces sustained opposing-lane skirting |
| `3r5` Lateral Clearance | L3 | ~30 | Adjacent agents come within the dynamic minimum lateral distance |
| `2r2` Route Adherence | L2 | ~45 | Observer-only; lateral drift from the route corridor |
| `1r0` Yield Priority | L1 | ~25 | Observer-only; right-of-way ambiguity in multi-agent traffic |

## Files in this directory

- [`03_near_multiple_vehicles.mp4`](03_near_multiple_vehicles.mp4) — original MP4
- [`03_near_multiple_vehicles.gif`](03_near_multiple_vehicles.gif) — GIF embedded above
- [`03_near_multiple_vehicles_summary.png`](03_near_multiple_vehicles_summary.png) — episode summary
- [`03_near_multiple_vehicles_log.csv`](03_near_multiple_vehicles_log.csv) — per-tick CSV
- `README.md` — this file
