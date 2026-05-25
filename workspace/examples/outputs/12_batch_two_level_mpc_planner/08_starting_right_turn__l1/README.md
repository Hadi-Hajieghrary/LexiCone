# 08 — Starting Right Turn

> **nuPlan scenario type.** `starting_right_turn`
> **Behaviour class.** Turn
> **Episode duration.** ~15 s (150 ticks)
> **Top observed violation.** `2r2` Route Adherence (integrated $44.99$)
> **Total violating rules.** 6 of 25 — cleaner than scenario 07 (left turn)

## What happens

The ego initiates a right turn — typically a less-constrained manoeuvre than a left turn in right-hand-drive traffic since there is no opposing-lane yield obligation. nuPlan tags such instances when the recorded driver begins a right-yaw transient with stop-and-launch profile. Within this batch the right-turn instance is taken from `2021.06.08.14.35.24_veh-26_02555_03004` — a different log than scenario 07's left turn.

The much cleaner violation profile (6 rules vs 14 for the left turn) reflects the right-turn geometry: no opposing-lane interaction, simpler traffic-light constraint structure (only the right-turn-arrow vs through-traffic distinction at most signalised intersections), and tighter routing options that keep the trajectory inside the route corridor for more of the manoeuvre.

## Simulation playback

![08_starting_right_turn](08_starting_right_turn.gif)

> **How to watch.** The ego swings right, threading the right-turn lane-connector geometry. Compare to scenario 07 (left turn) — the sidebar's active-rules panel is noticeably quieter here, and the bottom strip is dominated by the L2 (blue) route-adherence band without the L7 (amber) traffic-light or opposing-lane spikes.

Full resolution: [`08_starting_right_turn.mp4`](08_starting_right_turn.mp4). Summary: [`08_starting_right_turn_summary.png`](08_starting_right_turn_summary.png). Log: [`08_starting_right_turn_log.csv`](08_starting_right_turn_log.csv).

## What the LCP-WS-$L_1$ planner does

Same machinery as scenario 07 (global re-extraction onto the post-turn lane corridor, MPC tracks; `TrafficLightRule` enforces stop-line when applicable). The simpler geometry means fewer rule encoders fire — the LCP slack vector is dominated by the `LaneCorridorRule`'s two route-corridor half-planes, which the trajectory just barely respects through the swing.

## Top violations observed

| Rule | Level | Integrated | Why it fires |
|---|---|---:|---|
| `2r2` Route Adherence | L2 | **44.99** | Observer-only; the swing's lateral offset exceeds the route corridor during the turn |
| `1r0` Yield Priority | L1 | ~25 | Observer-only |
| `0r2` Longitudinal Comfort | L0 | small | Brake / launch ramps |

## Files in this directory

- [`08_starting_right_turn.mp4`](08_starting_right_turn.mp4) · [`08_starting_right_turn.gif`](08_starting_right_turn.gif) · [`08_starting_right_turn_summary.png`](08_starting_right_turn_summary.png) · [`08_starting_right_turn_log.csv`](08_starting_right_turn_log.csv)
