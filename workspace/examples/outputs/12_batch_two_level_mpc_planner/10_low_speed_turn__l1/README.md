# 10 — Low-Speed Turn

> **nuPlan scenario type.** `starting_low_speed_turn`
> **Behaviour class.** Turn (low-speed)
> **Episode duration.** ~15 s (149 ticks)
> **Top observed violation.** `7r2` Opposing Lane (integrated $369.72$) — the largest single-rule violation in the batch
> **Total violating rules.** 8 of 25

## What happens

The ego negotiates a low-speed turn — typically a parking-lot exit, residential-street corner, or queue-into-intersection moment where the recorded driver slows substantially before initiating yaw. nuPlan tags such instances when speed drops below the turn-speed threshold for an extended portion of the manoeuvre.

The remarkable feature of this scenario is the **integrated `7r2` violation of $369.72$ — the largest single-rule magnitude in the entire 16-scenario batch**. The mechanism: at low speed, the kinematic-bicycle's tan(δ) term dominates the dynamics; the planner can execute a tight turn but the trajectory's footprint sweeps a wide arc through opposing-lane polygons. The speed-weighted overlap accumulates over many ticks (the violation is *low rate × many ticks*, not a single sharp spike). The planner is *correctly* prioritising the lane-corridor constraint while accepting steady opposing-lane occupancy as the price of completing the turn — exactly the kind of trade-off the LCP framework's priority structure makes explicit.

## Simulation playback

![10_low_speed_turn](10_low_speed_turn.gif)

> **How to watch.** The ego turns slowly through an arc whose geometry forces a multi-tick opposing-lane occupancy. Track the bottom strip — you'll see the amber (L7) band sustained for a long horizontal stretch, unlike the brief spikes in scenarios 02 or 11. The context card shows low speed throughout.

Full resolution: [`10_low_speed_turn.mp4`](10_low_speed_turn.mp4). Summary: [`10_low_speed_turn_summary.png`](10_low_speed_turn_summary.png). Log: [`10_low_speed_turn_log.csv`](10_low_speed_turn_log.csv).

## What the LCP-WS-$L_1$ planner does

The LCP $L_1$ weighted-sum's calibrated weight prioritises lane-corridor staying-in-route (L2 / L7 surface constraints) over the opposing-lane constraint when the route requires the turn. At low speed the trade-off is permissive: the per-step lateral excursion is small, but the cumulative count is high. The cascade variant (`C4` in the comparative protocol) would tighten this further at the cost of one Algorithm 1A calibration per scenario class.

## Top violations observed

| Rule | Level | Integrated | Why it fires |
|---|---|---:|---|
| `7r2` Opposing Lane | L7 | **369.72** | Sustained low-speed swing through opposing-direction lane polygon |
| `2r2` Route Adherence | L2 | ~45 | Observer-only |
| `1r0` Yield Priority | L1 | ~25 | Observer-only |
| `0r2` Longitudinal Comfort | L0 | small | Brake / launch ramps at turn entry / exit |

## Files in this directory

- [`10_low_speed_turn.mp4`](10_low_speed_turn.mp4) · [`10_low_speed_turn.gif`](10_low_speed_turn.gif) · [`10_low_speed_turn_summary.png`](10_low_speed_turn_summary.png) · [`10_low_speed_turn_log.csv`](10_low_speed_turn_log.csv)
