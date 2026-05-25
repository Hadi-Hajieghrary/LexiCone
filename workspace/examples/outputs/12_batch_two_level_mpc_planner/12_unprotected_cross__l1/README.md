# 12 — Unprotected Cross Turn

> **nuPlan scenario type.** `starting_unprotected_cross_turn`
> **Behaviour class.** Turn (unprotected intersection)
> **Episode duration.** ~15 s (149 ticks)
> **Top observed violation.** `7r2` Opposing Lane (integrated $366.05$)
> **Total violating rules.** 8 of 25

## What happens

The ego initiates a turn at an *unprotected* intersection — meaning the controlling traffic signal does not provide a dedicated turn-protected phase, so the recorded driver must yield to oncoming through-traffic before executing the manoeuvre. nuPlan distinguishes this from the protected variant ([scenario 11](../11_protected_cross__l1/README.md)) by the *creep-and-wait* pattern in the recorded ego trace as the driver edges into the intersection to read gaps in oncoming traffic.

The integrated `7r2` violation ($366.05$) is approximately **double** the protected variant's, reflecting how much longer the trajectory occupies opposing-direction lane geometry during the creep / wait / commit pattern. This is one of the most informative scenarios for the LCP framework: it shows the planner correctly handling the multi-step priority trade-off — wait at the intersection edge (L1 safety > L7 legal), then commit through the swing once the gap is read, accepting L7 opposing-lane cost for the duration of the swing.

## Simulation playback

![12_unprotected_cross](12_unprotected_cross.gif)

> **How to watch.** The ego edges forward into the intersection, pauses to read oncoming traffic, then commits to the turn. The bottom strip's amber (L7) band is sustained throughout — both during the creep (clipping opposing lane while waiting) and the swing (clipping it during execution). Compare to scenario 11 where L7 is more concentrated.

Full resolution: [`12_unprotected_cross.mp4`](12_unprotected_cross.mp4). Summary: [`12_unprotected_cross_summary.png`](12_unprotected_cross_summary.png). Log: [`12_unprotected_cross_log.csv`](12_unprotected_cross_log.csv).

## What the LCP-WS-$L_1$ planner does

Without an explicit gap-detection state machine, the LCP planner relies on the MPC's collision constraints (`CollisionRule`'s 8 slots) to handle oncoming traffic — each oncoming vehicle enters the OCP as a circular keep-out within the `occupancy_map_radius_m = 40 m`. The trajectory naturally pauses when the keep-outs forbid forward progress, then commits as the oncoming queue clears. The `OpposingLaneRule`'s L7 slack absorbs the geometry penalty for the duration; the priority structure (L1 safety > L7 legal) ensures the ego never trades collision safety for legal compliance.

## Top violations observed

| Rule | Level | Integrated | Why it fires |
|---|---|---:|---|
| `7r2` Opposing Lane | L7 | **366.05** | Sustained occupation of opposing lane during creep-and-commit |
| `2r2` Route Adherence | L2 | ~45 | Observer-only |
| `1r0` Yield Priority | L1 | ~25 | Observer-only; unprotected-turn yield is the rule's archetypal trigger |
| `0r2` Longitudinal Comfort | L0 | small | Multiple brake / launch ramps during the creep |

## Files in this directory

- [`12_unprotected_cross.mp4`](12_unprotected_cross.mp4) · [`12_unprotected_cross.gif`](12_unprotected_cross.gif) · [`12_unprotected_cross_summary.png`](12_unprotected_cross_summary.png) · [`12_unprotected_cross_log.csv`](12_unprotected_cross_log.csv)
