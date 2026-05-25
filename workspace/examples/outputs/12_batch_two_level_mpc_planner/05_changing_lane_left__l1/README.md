# 05 — Changing Lane (To Left)

> **nuPlan scenario type.** `changing_lane_to_left`
> **Behaviour class.** Lane change
> **Episode duration.** ~15 s (149 ticks)
> **Top observed violation.** `2r2` Route Adherence (integrated $44.70$)
> **Total violating rules.** 11 of 25

## What happens

The directionally-isolated companion to scenario [04](../04_changing_lane__l1/README.md): the ego transitions from its current lane to the lane on its **left**. nuPlan tags such instances when the recorded ego heading delta crosses a left-merge threshold mid-scenario. In US right-hand-drive traffic this is typically a passing manoeuvre or a left-exit preparation.

The episode duration, violating-rule count, and dominant violation magnitude are essentially identical to scenario 04 — they share the same recorded log instance (`2021.06.07.18.53.26_veh-26_00005_00427`) but differ in the scenario-type filter. The observer-only `2r2` value of $44.70$ is the structural signature of a single ~3 m lateral merge: the route-corridor predicate trips for a fixed number of consecutive ticks, integrated over the same duration.

## Simulation playback

![05_changing_lane_left](05_changing_lane_left.gif)

> **How to watch.** The ego (red) shifts leftward through the episode. Notice how the dashed planned trajectory anticipates the merge a few ticks before the actual lateral motion begins — that's the global planner's reference reprojection onto the target lane's centreline triggering before the MPC's velocity profile catches up.

Full resolution: [`05_changing_lane_left.mp4`](05_changing_lane_left.mp4). Summary: [`05_changing_lane_left_summary.png`](05_changing_lane_left_summary.png). Log: [`05_changing_lane_left_log.csv`](05_changing_lane_left_log.csv).

## What the LCP-WS-$L_1$ planner does

Identical mechanism to scenario 04 — no explicit lane-change decision module; the global planner's periodic re-extraction (every `replan_period_s = 8 s` or on lateral drift > 5 m) shifts the reference to the target lane and the MPC tracks. Lateral-clearance and comfort terms fire briefly during the merge; the route-adherence flag fires for the duration of the lateral offset.

## Top violations observed

| Rule | Level | Integrated | Why it fires |
|---|---|---:|---|
| `2r2` Route Adherence | L2 | **44.70** | Observer-only; left-merge offset exceeds the route corridor |
| `3r5` Lateral Clearance | L3 | ~10 | Adjacent agents in the target lane during the merge |
| `0r2` / `0r3` Comfort | L0 | small | Brief brake / steering ramps |
| `1r0` Yield Priority | L1 | ~25 | Observer-only |

## Files in this directory

- [`05_changing_lane_left.mp4`](05_changing_lane_left.mp4) · [`05_changing_lane_left.gif`](05_changing_lane_left.gif) · [`05_changing_lane_left_summary.png`](05_changing_lane_left_summary.png) · [`05_changing_lane_left_log.csv`](05_changing_lane_left_log.csv)
