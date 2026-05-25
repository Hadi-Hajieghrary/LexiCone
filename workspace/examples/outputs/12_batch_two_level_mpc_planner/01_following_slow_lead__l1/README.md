# 01 — Following Slow Lead

> **nuPlan scenario type.** `following_lane_with_slow_lead`
> **Behaviour class.** Overtake-style
> **Episode duration.** ~15 s (150 ticks)
> **Top observed violation.** `3r3` Safe Headway (integrated $68.29$)
> **Total violating rules.** 10 of 25

## What happens

The ego is approaching a vehicle moving substantially slower in the same lane — the canonical *follow-or-pass* moment that triggers headway-driven decision-making. nuPlan's mini split picks an instance from `2021.06.08.12.54.54_veh-26_04262_04732`. The lead vehicle's speed sits below the ego's desired cruise speed of $12\,\mathrm{m/s}$, so the ego's MPC must trade off route progression against opening up safe space.

The scenario stress-tests the planner's `3r3` safe-headway rule: the headway encoder (Section V.E of [`../../../../References/comprehensive_report.md`](../../../../References/comprehensive_report.md)) constrains $t_{\mathrm{hw}} \cdot v_k + d_{\min} - \mathrm{gap}_k \leq 0$ at every horizon step, where $\mathrm{gap}_k$ is the longitudinal distance to the in-lane lead. When the gap closes faster than the ego can decelerate within its `max_decel_mps2 = 3.5` cap, the slack term in the LCP $L_1$ epigraph (eq. 9 in the report) opens and the level-3 violation $V_3$ grows.

## Simulation playback

![01_following_slow_lead](01_following_slow_lead.gif)

> **How to watch.** The red rectangle is the ego; the green rectangle directly ahead in the same lane is the slow lead. The dashed dark-blue line is the MPC's 3-second forward prediction (note how it stays in lane and decelerates). The bottom strip lights up amber when the headway constraint fires. The sidebar's *active rules* panel sorts the violations by severity each tick.

Full-resolution playback: [`01_following_slow_lead.mp4`](01_following_slow_lead.mp4). Static episode summary: [`01_following_slow_lead_summary.png`](01_following_slow_lead_summary.png). Per-tick CSV: [`01_following_slow_lead_log.csv`](01_following_slow_lead_log.csv).

## What the LCP-WS-$L_1$ planner does

The planner detects the slow lead via the rule encoder's `_find_lead` heuristic (closest agent ahead within ±1.6 m lateral tolerance), inflates the headway constraint at every horizon step, and decelerates from cruise toward the lead's speed. Because the lead is also slowing intermittently, the MPC continually re-adjusts; the headway slack opens repeatedly as the gap closes, producing the sustained `3r3` violation. The ego does **not** overtake in this episode — the planner's global route stays on the current lane and there is no lane-change decision module driving a merge.

## Top violations observed

| Rule | Level | Integrated | Why it fires |
|---|---|---:|---|
| `3r3` Safe Headway | L3 | 68.29 | Lead repeatedly closes inside the safe-headway envelope $t_\mathrm{hw} \cdot v + d_\min$ |
| `2r2` Route Adherence | L2 | ~45 | Observer-only state-machine rule; fires whenever the global-planner reference and the simulator's recorded route diverge by more than the comfort tolerance |
| `1r0` Yield Priority | L1 | ~25 | Observer-only; fires when an agent has right-of-way the ego is encroaching on |
| `0r2` Longitudinal Comfort | L0 | small | Brake / launch ramps exceed $a_{x,\max}^{\mathrm{comf}} = 1.8\,\mathrm{m/s^2}$ |

Of the 10 violating rules, only `3r3` and `0r2` are MPC-controlled; the rest are observer-only state machines (Section VII.A of the comprehensive report) and would not change under any planner variant in the comparative protocol.

## Files in this directory

- [`01_following_slow_lead.mp4`](01_following_slow_lead.mp4) — original IEEE-styled MP4 (10 fps, 130 dpi)
- [`01_following_slow_lead.gif`](01_following_slow_lead.gif) — GIF embedded above (20 fps, 960 px wide)
- [`01_following_slow_lead_summary.png`](01_following_slow_lead_summary.png) — static episode summary plot
- [`01_following_slow_lead_log.csv`](01_following_slow_lead_log.csv) — per-tick × per-rule evaluations
- `README.md` — this file
