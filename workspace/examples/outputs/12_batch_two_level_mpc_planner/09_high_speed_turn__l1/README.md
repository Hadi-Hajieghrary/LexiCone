# 09 — High-Speed Turn

> **nuPlan scenario type.** `starting_high_speed_turn`
> **Behaviour class.** Turn (high-speed) — proxy for ramp exit
> **Episode duration.** ~15 s (149 ticks)
> **Top observed violation.** `7r2` Opposing Lane (integrated $90.79$)
> **Total violating rules.** 8 of 25

## What happens

The ego negotiates a curving lane segment at high cruise speed — the canonical *ramp exit* or *high-speed sweeper* moment. nuPlan tags such instances when the recorded driver maintains $> 8\,\mathrm{m/s}$ through a turn whose curvature exceeds a threshold. Within the curated batch this is the proxy for the freeway-ramp / cloverleaf-exit family of behaviours; the value is exercising the planner's lateral-acceleration encoder (`1r11` LateralAccelerationRule) against the speed constraint at high $v$.

The dominant violation is again `7r2` Opposing Lane — at high speed, the kinematic-bicycle's turning geometry forces the trajectory to occasionally clip lane-edge polygons; the LCP $L_1$ slack opens to absorb this. The magnitude ($90.79$) is well above scenarios 08 (right turn at lower speed) and 10 (low-speed turn) because the speed-weighted overlap accumulates faster.

## Simulation playback

![09_high_speed_turn](09_high_speed_turn.gif)

> **How to watch.** The ego sweeps through a high-curvature lane segment maintaining cruise speed. Note the planned trajectory's lateral excursion at the apex — that's where the lateral-acceleration constraint (`1r11`, $|a_y| \leq a_{y,\max}^{\mathrm{comf}} = 2.0\,\mathrm{m/s^2}$) trades off against the lane-corridor constraint. The sidebar's context card will show speed staying near or just below the posted limit.

Full resolution: [`09_high_speed_turn.mp4`](09_high_speed_turn.mp4). Summary: [`09_high_speed_turn_summary.png`](09_high_speed_turn_summary.png). Log: [`09_high_speed_turn_log.csv`](09_high_speed_turn_log.csv).

## What the LCP-WS-$L_1$ planner does

The kinematic-bicycle MPC linearises around the warm-start at each tick; the `BicycleLinearisation`'s $\dot\psi = (v / L) \tan \delta$ row is the source of the lateral-acceleration coupling that `LateralAccelerationRule` constrains. At cruise speed, even modest curvature produces high $a_y$ — the LCP slack at L3 opens to keep the OCP feasible. The `OpposingLaneRule` simultaneously activates as the curved trajectory clips lane-edge half-planes. The combination of L3 and L7 slack activity is the signature pattern of high-speed cornering.

## Top violations observed

| Rule | Level | Integrated | Why it fires |
|---|---|---:|---|
| `7r2` Opposing Lane | L7 | **90.79** | Curved trajectory clips opposing-direction lane polygon at the apex |
| `1r11` Lateral Acceleration | L1 | ~5 | $\|a_y\|$ briefly exceeds the comfort threshold during the sweep |
| `2r2` Route Adherence | L2 | ~45 | Observer-only |
| `1r0` Yield Priority | L1 | ~25 | Observer-only |

## Files in this directory

- [`09_high_speed_turn.mp4`](09_high_speed_turn.mp4) · [`09_high_speed_turn.gif`](09_high_speed_turn.gif) · [`09_high_speed_turn_summary.png`](09_high_speed_turn_summary.png) · [`09_high_speed_turn_log.csv`](09_high_speed_turn_log.csv)
