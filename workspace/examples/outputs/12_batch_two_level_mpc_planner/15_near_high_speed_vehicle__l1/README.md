# 15 — Near High-Speed Vehicle

> **nuPlan scenario type.** `near_high_speed_vehicle`
> **Behaviour class.** Dynamic
> **Episode duration.** ~15 s (149 ticks)
> **Top observed violation.** `7r2` Opposing Lane (integrated $104.94$)
> **Total violating rules.** 9 of 25

## What happens

The ego is in close proximity to another vehicle that is itself travelling at high speed — a passing manoeuvre at speed, freeway following, or a high-relative-speed encounter on an arterial. nuPlan tags such instances when at least one tracked agent within the occupancy-map radius has speed above a high threshold. The instance is from `2021.05.12.22.00.38_veh-35_01008_01518` (same log as scenario 13).

The dominant `7r2` violation reflects that at high relative speed, lateral keep-out constraints get tight quickly — the planner's MPC briefly clips lane-edge polygons while keeping the per-agent collision slack feasible. The magnitude ($104.94$) sits in the middle of the L7 violation spectrum, well above the open-cruise scenarios (13, 14) but well below the multi-vehicle (03) or low-speed-turn (10) scenarios.

## Simulation playback

![15_near_high_speed_vehicle](15_near_high_speed_vehicle.gif)

> **How to watch.** Track the green rectangle moving at high speed near or alongside the ego. The planned trajectory's lateral response is the key signal — small but rapid corrections as the planner balances collision slack against lane-corridor slack. The bottom strip shows brief amber (L7) spikes correlated with the moments of closest passing.

Full resolution: [`15_near_high_speed_vehicle.mp4`](15_near_high_speed_vehicle.mp4). Summary: [`15_near_high_speed_vehicle_summary.png`](15_near_high_speed_vehicle_summary.png). Log: [`15_near_high_speed_vehicle_log.csv`](15_near_high_speed_vehicle_log.csv).

## What the LCP-WS-$L_1$ planner does

The high-speed agent enters the OCP's `obstacle_slot_count = 6` filter as a circular keep-out; the per-step collision slack constraint $\big( (x_k - o_{j,x})^2 + (y_k - o_{j,y})^2 + s_{j,k}^2 \big) \geq (r_j + r_{\mathrm{ego}})^2$ tightens as the relative velocity closes the gap quickly. The MPC's lateral adjustment fires `OpposingLaneRule` for brief windows as it accepts a small lane-corridor breach to maintain the safety margin. This is a textbook example of the LCP priority structure paying off — L1 safety strictly dominates L7 legal compliance at the calibrated weights.

## Top violations observed

| Rule | Level | Integrated | Why it fires |
|---|---|---:|---|
| `7r2` Opposing Lane | L7 | **104.94** | Lateral keep-out during the high-speed passing |
| `3r5` Lateral Clearance | L3 | ~15 | The high-speed agent briefly enters the dynamic lateral-clearance envelope |
| `2r2` Route Adherence | L2 | ~45 | Observer-only |
| `1r0` Yield Priority | L1 | ~25 | Observer-only |

## Files in this directory

- [`15_near_high_speed_vehicle.mp4`](15_near_high_speed_vehicle.mp4) · [`15_near_high_speed_vehicle.gif`](15_near_high_speed_vehicle.gif) · [`15_near_high_speed_vehicle_summary.png`](15_near_high_speed_vehicle_summary.png) · [`15_near_high_speed_vehicle_log.csv`](15_near_high_speed_vehicle_log.csv)
