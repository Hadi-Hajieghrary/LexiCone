# 11 — Protected Cross Turn

> **nuPlan scenario type.** `starting_protected_cross_turn`
> **Behaviour class.** Turn (protected intersection)
> **Episode duration.** ~15 s (149 ticks)
> **Top observed violation.** `7r2` Opposing Lane (integrated $174.24$)
> **Total violating rules.** 11 of 25

## What happens

The ego initiates a turn at a *protected* intersection — meaning the controlling traffic signal provides a dedicated turn-green phase that excludes conflicting cross-traffic. The recorded driver waits for the protected phase, then executes the turn through the intersection polygon. nuPlan distinguishes this from the unprotected variant ([scenario 12](../12_unprotected_cross__l1/README.md)) by the absence of yield-to-oncoming behaviour in the recorded ego trace.

The protected designation is reflected in the violation profile: the magnitude of `7r2` ($174$) is roughly **half** of scenario 12's unprotected counterpart ($366$), because the protected phase removes the need to creep into the intersection to read oncoming gaps — the ego simply waits for green and goes.

## Simulation playback

![11_protected_cross](11_protected_cross.gif)

> **How to watch.** A clean stop-and-go pattern at the intersection entrance, followed by a swing through the lane-connector geometry. The bottom strip's amber (L7) band is concentrated in the turn-execution phase, not the wait phase — contrast with scenario 12 where you'll see L7 activity throughout the creeping approach.

Full resolution: [`11_protected_cross.mp4`](11_protected_cross.mp4). Summary: [`11_protected_cross_summary.png`](11_protected_cross_summary.png). Log: [`11_protected_cross_log.csv`](11_protected_cross_log.csv).

## What the LCP-WS-$L_1$ planner does

Same mechanism as scenarios 07 and 12: `TrafficLightRule` caps longitudinal progress when the controlling TL is RED/YELLOW; the planner brakes to the stop line and waits. Once green, the route corridor through the intersection's lane connectors is tracked; `OpposingLaneRule` fires briefly as the trajectory clips opposing-direction connector polygons mid-swing.

## Top violations observed

| Rule | Level | Integrated | Why it fires |
|---|---|---:|---|
| `7r2` Opposing Lane | L7 | **174.24** | Trajectory clips opposing-direction connectors during the swing |
| `2r2` Route Adherence | L2 | ~45 | Observer-only |
| `7r1` Traffic Light | L7 | small | Brief overlap with stop-line constraint at the wait |
| `1r0` Yield Priority | L1 | ~25 | Observer-only |
| `0r2` / `0r3` Comfort | L0 | small | Brake / launch ramps |

## Files in this directory

- [`11_protected_cross.mp4`](11_protected_cross.mp4) · [`11_protected_cross.gif`](11_protected_cross.gif) · [`11_protected_cross_summary.png`](11_protected_cross_summary.png) · [`11_protected_cross_log.csv`](11_protected_cross_log.csv)
