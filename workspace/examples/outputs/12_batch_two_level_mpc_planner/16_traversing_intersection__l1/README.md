# 16 — Traversing Intersection

> **nuPlan scenario type.** `traversing_intersection`
> **Behaviour class.** Dynamic (full intersection traversal)
> **Episode duration.** ~15 s (149 ticks)
> **Top observed violation.** `7r2` Opposing Lane (integrated $186.39$)
> **Total violating rules.** 9 of 25

## What happens

The ego executes a *through-pass* of an intersection — entering on one approach lane, crossing the intersection polygon along a lane-connector, and exiting on the corresponding exit lane. nuPlan tags such instances when the recorded ego trajectory passes through an intersection without turning (distinguishing it from the protected/unprotected cross-turn variants in scenarios 11 and 12).

This is one of the **most visually striking scenarios in the batch**: the intersection polygon is large, multiple lane connectors are visible, traffic-light markers cycle through colours, and the planner must coordinate the route corridor through the intersection geometry while respecting the signal. The dominant violation `7r2` ($186.39$) reflects the inevitable opposing-direction lane-connector clipping during the cross-traversal — at most intersection geometries the route corridor passes through connector polygons whose centreline headings are anti-parallel to the approach direction.

## Simulation playback

![16_traversing_intersection](16_traversing_intersection.gif)

> **How to watch.** This is the showcase scenario for the project's visualisation pipeline. Watch the traffic-light markers cycle, the planned trajectory thread through the intersection, the multiple lane connectors light up in the legend, and the sidebar's active-rules panel pulse with multi-priority activity. The bottom strip shows distinct phases: approach (steady L7), within-intersection (peak L7 + brief other firings), post-intersection (decay).

Full resolution: [`16_traversing_intersection.mp4`](16_traversing_intersection.mp4). Summary: [`16_traversing_intersection_summary.png`](16_traversing_intersection_summary.png). Log: [`16_traversing_intersection_log.csv`](16_traversing_intersection_log.csv).

## What the LCP-WS-$L_1$ planner does

The intersection traversal is the most rule-encoder-active phase of the LCP planner's operating envelope. Active during this scenario: `TrafficLightRule` (longitudinal stop when controlling TL is RED/YELLOW; satisfied here since the ego catches a green), `LaneCorridorRule` (route corridor through lane connectors), `OpposingLaneRule` (sustained clipping of opposing-direction connectors mid-cross), `CollisionRule` (per-agent keep-outs for any cross-traffic during the traversal), and the always-on comfort rules (`0r2`, `1r11`, `0r3`). The LCP slack vector has activity at every priority level for at least a portion of the episode — a rare combination.

This scenario is also the canonical demonstration of the LCP framework's *structural* value: a flat-weight planner cannot guarantee priority semantics in such a multi-rule-active situation, while the LCP's per-level epigraph slacks make the trade-off explicit. The comparative-protocol cells under [`../../13_protocol/`](../../13_protocol/) for this scenario should show the LCP cascade (`C4`) lex-Pareto-dominating the legacy baseline (`C0`) by the largest margin in the benchmark.

## Top violations observed

| Rule | Level | Integrated | Why it fires |
|---|---|---:|---|
| `7r2` Opposing Lane | L7 | **186.39** | Sustained lane-connector clipping during the cross-traversal |
| `2r2` Route Adherence | L2 | ~45 | Observer-only |
| `1r0` Yield Priority | L1 | ~25 | Observer-only |
| `3r5` Lateral Clearance | L3 | ~10 | Brief proximity to cross-traffic mid-intersection |
| `0r2` / `0r3` Comfort | L0 | small | Brake / launch ramps at intersection entry / exit |

## Featured in

This scenario is the source of the headline frame in [`../artifacts/violations/16_traversing_intersection__gallery.png`](../artifacts/violations/16_traversing_intersection__gallery.png) and individual snapshots at [`../artifacts/violations/16_traversing_intersection/`](../artifacts/violations/16_traversing_intersection/). It's referenced from the visualiser-redesign section of the [comprehensive report](../../../../References/comprehensive_report.md) and the formal LCP paper's §14.9 implementation lessons.

## Files in this directory

- [`16_traversing_intersection.mp4`](16_traversing_intersection.mp4) · [`16_traversing_intersection.gif`](16_traversing_intersection.gif) · [`16_traversing_intersection_summary.png`](16_traversing_intersection_summary.png) · [`16_traversing_intersection_log.csv`](16_traversing_intersection_log.csv)
