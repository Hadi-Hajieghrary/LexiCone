# 13 — High-Magnitude Speed

> **nuPlan scenario type.** `high_magnitude_speed`
> **Behaviour class.** Dynamic
> **Episode duration.** ~15 s (149 ticks)
> **Top observed violation.** `1r0` Yield Priority (integrated $45.16$)
> **Total violating rules.** 5 of 25 — tied for cleanest with scenario 06

## What happens

The ego is travelling at the **upper end of its speed envelope** — typically $> 18\,\mathrm{m/s}$ — through a stretch where the recorded driver maintained high speed for an extended portion of the scenario. nuPlan tags these as `high_magnitude_speed` to capture freeway / arterial cruising behaviours. Within this batch the instance is from `2021.05.12.22.00.38_veh-35_01008_01518`.

The clean profile (5 rules) reflects that high-speed open-road cruising is *operationally easy* for a well-tracked MPC — the speed cap is not binding (cruise speed exceeds neither the posted limit nor `max_speed_mps = 25`), the lane corridor is straight, and there are no close-range agents demanding collision-slack activity. The dominant violation `1r0` is **observer-only** (yield-priority arbitration is a state-machine rule); the MPC variants are essentially indistinguishable on this scenario.

## Simulation playback

![13_high_magnitude_speed](13_high_magnitude_speed.gif)

> **How to watch.** Smooth, sustained high-speed cruising. The sidebar's context card shows speed well above $10\,\mathrm{m/s}$. The bottom strip is largely white except for the persistent purple (L1 yield priority) and blue (L2 route adherence) observer-only bands. A useful visual baseline for what the LCP planner looks like when it's *not* under stress.

Full resolution: [`13_high_magnitude_speed.mp4`](13_high_magnitude_speed.mp4). Summary: [`13_high_magnitude_speed_summary.png`](13_high_magnitude_speed_summary.png). Log: [`13_high_magnitude_speed_log.csv`](13_high_magnitude_speed_log.csv).

## What the LCP-WS-$L_1$ planner does

Pure reference tracking. The route-corridor half-planes, speed-limit constraint, and headway constraint are all comfortably satisfied; the LCP slack vector stays near zero across all priority levels. This scenario is the planner's baseline behaviour — it shows what the trajectory looks like when no rule encoders are firing.

## Top violations observed

| Rule | Level | Integrated | Why it fires |
|---|---|---:|---|
| `1r0` Yield Priority | L1 | **45.16** | Observer-only; sustained right-of-way pattern in the high-speed stretch |
| `2r2` Route Adherence | L2 | ~45 | Observer-only |
| `0r2` Longitudinal Comfort | L0 | small | Minor speed adjustments |

## Files in this directory

- [`13_high_magnitude_speed.mp4`](13_high_magnitude_speed.mp4) · [`13_high_magnitude_speed.gif`](13_high_magnitude_speed.gif) · [`13_high_magnitude_speed_summary.png`](13_high_magnitude_speed_summary.png) · [`13_high_magnitude_speed_log.csv`](13_high_magnitude_speed_log.csv)
