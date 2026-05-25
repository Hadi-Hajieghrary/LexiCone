# 04 — Changing Lane

> **nuPlan scenario type.** `changing_lane`
> **Behaviour class.** Lane change
> **Episode duration.** ~15 s (149 ticks)
> **Top observed violation.** `2r2` Route Adherence (integrated $44.70$)
> **Total violating rules.** 11 of 25

## What happens

The ego is performing a lane change — direction unspecified (the `changing_lane` scenario type subsumes both left and right manoeuvres; scenarios 05 and 06 isolate each direction). The recorded human driver makes a lateral transition between two adjacent travel lanes; the simulator's reactive agents respond to the ego's planned motion.

The dominant violation, `2r2` Route Adherence, is **observer-only** — it has no LCP encoder because route adherence is a state-machine property the per-step convex template cannot express. The 25-rule observer set is partitioned into $\mathcal{R}_\mathrm{MPC}$ (the 16 rules with an active LCP encoder) and $\mathcal{R}_\mathrm{inv}$ (the 9 observer-only state-machine rules plus 3 stub placeholders); no MPC variant can affect rules in $\mathcal{R}_\mathrm{inv}$, so `2r2` is reported as an *invariant negative control* in the comparative protocol. The integrated value of $44.70$ is essentially constant across many lane-change scenarios; it reflects the fact that the global planner's route corridor and the simulator's reactive-agent-driven recorded path diverge during the manoeuvre, which the observer flags every tick the lateral offset exceeds the corridor.

## Simulation playback

![04_changing_lane](04_changing_lane.gif)

> **How to watch.** Track the red ego making a lateral transition between adjacent lanes. The dashed planned trajectory will bend into the target lane; the solid trail behind the ego records the actual world-frame path. The bottom strip shows a sustained blue (L2 route adherence) band — that's the observer-only signal that no MPC variant can affect.

Full resolution: [`04_changing_lane.mp4`](04_changing_lane.mp4). Summary: [`04_changing_lane_summary.png`](04_changing_lane_summary.png). Log: [`04_changing_lane_log.csv`](04_changing_lane_log.csv).

## What the LCP-WS-$L_1$ planner does

The planner has no explicit lane-change *decision* logic — it tracks the global planner's reference path, which is recomputed every `replan_period_s = 8 s` and on lateral-drift triggers (`abs(lat) > 5 m`). When the recorded scenario's drift exceeds 5 m, the global planner re-projects onto the target lane's centreline, and the MPC tracks the new reference. The lane-change therefore appears in the trajectory as a smooth lateral transition rather than a discrete state-machine event.

The 11 violating rules include comfort terms (`0r2` brake / accel ramps, `0r3` lateral steering excess) that fire briefly during the merge, plus observer-only state-machine rules (`2r2`, `1r0`, `3r6` lane intrusion) that any planner variant would trigger on the same scenario instance.

## Top violations observed

| Rule | Level | Integrated | Why it fires |
|---|---|---:|---|
| `2r2` Route Adherence | L2 | **44.70** | Observer-only; the merge's lateral offset exceeds the route corridor |
| `0r2` Longitudinal Comfort | L0 | ~5 | Brief brake / accel ramps during the merge |
| `3r5` Lateral Clearance | L3 | ~10 | Adjacent agents in the target lane briefly enter the lateral-clearance envelope |
| `1r0` Yield Priority | L1 | ~25 | Observer-only |

## Files in this directory

- [`04_changing_lane.mp4`](04_changing_lane.mp4) — original MP4
- [`04_changing_lane.gif`](04_changing_lane.gif) — GIF embedded above
- [`04_changing_lane_summary.png`](04_changing_lane_summary.png) — episode summary
- [`04_changing_lane_log.csv`](04_changing_lane_log.csv) — per-tick CSV
- `README.md` — this file
