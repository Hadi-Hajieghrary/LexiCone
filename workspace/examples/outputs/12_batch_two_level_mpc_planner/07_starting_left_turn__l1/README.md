# 07 — Starting Left Turn

> **nuPlan scenario type.** `starting_left_turn`
> **Behaviour class.** Turn (sharp)
> **Episode duration.** ~15 s (150 ticks)
> **Top observed violation.** `7r1` Traffic-Light Compliance (integrated $47.99$)
> **Total violating rules.** 14 of 25 — the highest count in the batch

## What happens

The ego begins a sharp left turn — typically at a signalised intersection, since nuPlan tags such manoeuvres when the recorded driver initiates a left-yaw transient with a stop / wait / launch profile characteristic of signalised left-turn negotiation. Within the curated 16-scenario list this serves as the proxy for U-turn / hard-left geometry; the scenario's value is exercising the planner's interaction with traffic-light state, the opposing-lane encoder during the swing, and the lane-corridor encoder as the route transitions across lane connectors inside the intersection polygon.

This scenario is notable for **two simultaneous high-priority firings**: `7r1` (the *new* TrafficLightRule wired in this project's Phase 2 work — Section V.E) and `7r2` (OpposingLane) both dominate, plus the usual route-adherence and yield-priority observer activity. The **14-rule violating count** is the highest in the batch, reflecting how stressful left turns are for a rulebook MPC: virtually every traffic-rule encoder fires at some point during the manoeuvre.

## Simulation playback

![07_starting_left_turn](07_starting_left_turn.gif)

> **How to watch.** The ego sits at or approaches a left-turn position; watch the traffic-light marker (filled circle) over the controlling lane connector cycle through colours. The MPC's `TrafficLightRule` enforces `x_ego ≤ x_stop - buffer` while the light is RED/YELLOW; when green, the ego launches into the intersection along the lane-connector centreline. The bottom strip lights up across multiple priority bands during the swing.

Full resolution: [`07_starting_left_turn.mp4`](07_starting_left_turn.mp4). Summary: [`07_starting_left_turn_summary.png`](07_starting_left_turn_summary.png). Log: [`07_starting_left_turn_log.csv`](07_starting_left_turn_log.csv).

## What the LCP-WS-$L_1$ planner does

The `TrafficLightRule` enforces a longitudinal stop constraint $x_{\mathrm{ego},k} - (x_{\mathrm{stop}} - \mathrm{buffer}) \leq 0$ when the controlling lane connector's TL is RED or YELLOW, active across every horizon step. The planner brakes to the stop line, holds, then launches when the light goes GREEN. During the swing through the intersection the `OpposingLaneRule` fires briefly as the ego's trajectory clips opposing-direction lane connectors mid-turn — an artefact of the route corridor passing through intersection geometry where lane polygons overlap. The total $V_7$ contribution from `7r1` and `7r2` is the dominant cost component for this episode.

## Top violations observed

| Rule | Level | Integrated | Why it fires |
|---|---|---:|---|
| `7r1` Traffic Light | L7 | **47.99** | Ego approaches a RED/YELLOW light; the stop-line constraint is the binding longitudinal cap during the wait |
| `7r2` Opposing Lane | L7 | ~30 | Trajectory clips opposing-direction lane connectors mid-swing |
| `2r2` Route Adherence | L2 | ~45 | Observer-only |
| `1r0` Yield Priority | L1 | ~25 | Observer-only; left-turn yield-to-oncoming pattern |
| `0r2` / `0r3` Comfort | L0 | small | Brake-and-launch ramps |

The combined L7 activity (TrafficLight + OpposingLane) is one of the most distinctive signatures of the LCP planner's behaviour — both encoders are recent (Phase 2 wiring) and would not be present in the legacy baseline.

## Files in this directory

- [`07_starting_left_turn.mp4`](07_starting_left_turn.mp4) · [`07_starting_left_turn.gif`](07_starting_left_turn.gif) · [`07_starting_left_turn_summary.png`](07_starting_left_turn_summary.png) · [`07_starting_left_turn_log.csv`](07_starting_left_turn_log.csv)
