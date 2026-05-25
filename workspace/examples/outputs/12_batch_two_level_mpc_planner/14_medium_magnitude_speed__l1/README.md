# 14 — Medium-Magnitude Speed

> **nuPlan scenario type.** `medium_magnitude_speed`
> **Behaviour class.** Dynamic
> **Episode duration.** ~15 s (150 ticks)
> **Top observed violation.** `3r0` Speed Limit (integrated $48.80$)
> **Total violating rules.** 6 of 25

## What happens

The ego cruises in the **mid-speed regime** — typically $8$–$14\,\mathrm{m/s}$, characteristic of arterial roads with posted limits around $25$–$35\,\mathrm{mph}$. nuPlan tags such instances when the recorded driver maintains medium speed for the scenario duration. The instance here is from `2021.06.08.14.35.24_veh-26_02555_03004`.

The distinctive feature: this is the **only scenario in the batch where the top violation is `3r0` Speed Limit**. The planner's `desired_speed_mps = 12.0` plus the per-step velocity cap (`_build_v_cap`) interact with the lane's posted limit (read from `LaneSnapshot.speed_limit_mps`), and on this particular log instance the cruise speed briefly exceeds the limit by the 1 m/s tolerance the `SpeedLimitRule` allows before firing. The integrated value of $48.80$ is small in magnitude (it's a transient over-shoot) but is the dominant signal because nothing else fires significantly.

## Simulation playback

![14_medium_magnitude_speed](14_medium_magnitude_speed.gif)

> **How to watch.** Steady mid-speed cruising. Track the sidebar context card's `v_lim` vs ego speed columns — the brief moments where the speed exceeds the posted limit will correlate with the L3 (teal) blips in the bottom strip. The trajectory is otherwise straight and unperturbed.

Full resolution: [`14_medium_magnitude_speed.mp4`](14_medium_magnitude_speed.mp4). Summary: [`14_medium_magnitude_speed_summary.png`](14_medium_magnitude_speed_summary.png). Log: [`14_medium_magnitude_speed_log.csv`](14_medium_magnitude_speed_log.csv).

## What the LCP-WS-$L_1$ planner does

The `SpeedLimitRule` encoder generates the linear constraint $v_k - v_{\lim}(s_k) \leq 0$ at every horizon step where $v_{\lim}(s_k)$ is the per-vertex speed limit read from the global planner's reference. When the ego briefly exceeds the limit, the slack opens with low rate but accumulates over the over-shoot duration. The planner is otherwise in pure tracking mode.

## Top violations observed

| Rule | Level | Integrated | Why it fires |
|---|---|---:|---|
| `3r0` Speed Limit | L3 | **48.80** | Transient over-shoot of posted lane speed limit |
| `2r2` Route Adherence | L2 | ~45 | Observer-only |
| `1r0` Yield Priority | L1 | ~25 | Observer-only |

## Files in this directory

- [`14_medium_magnitude_speed.mp4`](14_medium_magnitude_speed.mp4) · [`14_medium_magnitude_speed.gif`](14_medium_magnitude_speed.gif) · [`14_medium_magnitude_speed_summary.png`](14_medium_magnitude_speed_summary.png) · [`14_medium_magnitude_speed_log.csv`](14_medium_magnitude_speed_log.csv)
