"""Planners for the closed-loop simulation demos.

All planners implement the :class:`~examples.simulation.Planner` protocol:
``reset()`` and ``plan(ctx) -> PlannerCommand``. They read whatever derived
context they need from the :class:`SceneContext` the simulator gives them —
``ego_lane``, ``ego_speed_limit_mps``, ``lead_agent()``, etc.

Six planners ship today:

- :class:`ConstantSpeedPlanner` — hold initial speed, no steering. Baseline.
- :class:`IDMPlanner` — Intelligent Driver Model: maintain a desired speed
  and a safe headway behind an in-lane lead.
- :class:`AggressivePlanner` — push toward a high target speed regardless of
  in-lane lead. Mostly used to make rules fire.
- :class:`OvertakePlanner` — five-phase state machine
  ``approach → merge_left → pass → merge_right → cruise`` for a single
  overtake against a slow leader in an adjacent lane.
- :class:`UrbanDrivingPlanner` — composite IDM extended with traffic-light,
  stop-sign, and pedestrian-yield handling; picks the most restrictive of
  several longitudinal hazards each tick.
- :class:`LaneChangePlanner` — six-phase state machine
  ``follow → prepare_lane_change → merge → pass → return → follow`` with
  explicit target-lane clearance verification inside ``merge_corridor_m``
  before committing to lateral motion.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

from lexicone.observer import SceneContext
from lexicone.observer.context import VEHICLE_LIKE_TYPES, VRU_TYPES, _type_value
from lexicone.observer.geometry import polygon_from_points
from lexicone.observer.types import AgentType, TrafficLightState

from .simulation import PlannerCommand, project_forward


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _wrap_pi(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


class ConstantSpeedPlanner:
    """Hold the current speed; never steer.

    Used as a baseline ("naive autopilot") — most rules should stay quiet
    in a clean scene but it will gladly drive into a leader.
    """

    name = "constant_speed"

    def reset(self) -> None:
        pass

    def plan(self, ctx: SceneContext) -> PlannerCommand:
        cmd = PlannerCommand(ax_mps2=0.0, yaw_rate_radps=0.0)
        cmd.planned_trajectory = project_forward(ctx.ego, cmd)
        return cmd


class IDMPlanner:
    """Treiber's Intelligent Driver Model along the lane centerline.

    Free-flow term pulls toward ``desired_speed``; interaction term backs off
    based on bumper-to-bumper gap and closing speed to an in-lane lead.
    Optionally caps at the posted speed limit.
    """

    name = "idm"

    def __init__(
        self,
        desired_speed_mps: float = 11.0,
        time_headway_s: float = 1.5,
        min_gap_m: float = 2.0,
        max_accel_mps2: float = 1.5,
        comfort_decel_mps2: float = 2.0,
        respect_speed_limit: bool = True,
    ):
        self.v0 = desired_speed_mps
        self.T = time_headway_s
        self.s0 = min_gap_m
        self.a = max_accel_mps2
        self.b = comfort_decel_mps2
        self.respect_speed_limit = respect_speed_limit

    def reset(self) -> None:
        pass

    def plan(self, ctx: SceneContext) -> PlannerCommand:
        v = ctx.ego.speed
        v0 = self.v0
        if self.respect_speed_limit and ctx.ego_speed_limit_mps is not None:
            v0 = min(v0, ctx.ego_speed_limit_mps)

        free_term = 1.0 - (v / max(v0, 1e-3)) ** 4
        lead = ctx.lead_agent()
        if lead is not None:
            gap = max(0.1, lead.longitudinal_distance_m - (ctx.ego.length + lead.agent.length) / 2.0)
            dv = v - lead.agent.speed
            s_star = self.s0 + max(
                0.0, v * self.T + (v * dv) / (2.0 * math.sqrt(self.a * self.b))
            )
            interaction_term = (s_star / gap) ** 2
            ax = self.a * (free_term - interaction_term)
        else:
            ax = self.a * free_term

        ax = max(-4.0, min(self.a, ax))
        cmd = PlannerCommand(ax_mps2=ax, yaw_rate_radps=0.0)
        cmd.planned_trajectory = project_forward(ctx.ego, cmd)
        return cmd


class AggressivePlanner:
    """Push hard toward a high target speed, ignore lead-vehicle gap, hold
    a fixed lateral drift.

    Designed to exercise rule violations: the lateral drift drives encroachment
    into adjacent lanes / bike lanes / walkways depending on the scene; the
    high target speed routinely overshoots the posted limit.
    """

    name = "aggressive"

    def __init__(
        self,
        target_speed_mps: float = 20.0,
        max_accel_mps2: float = 2.5,
        lateral_drift_radps: float = 0.0,
        ignore_lead_below_m: float = 3.0,
    ):
        self.v_target = target_speed_mps
        self.a_max = max_accel_mps2
        self.yaw_rate = lateral_drift_radps
        self.ignore_lead_below_m = ignore_lead_below_m

    def reset(self) -> None:
        pass

    def plan(self, ctx: SceneContext) -> PlannerCommand:
        v = ctx.ego.speed
        # Only brake if we'd basically hit the leader within a car length.
        lead = ctx.lead_agent()
        if lead is not None and lead.longitudinal_distance_m < self.ignore_lead_below_m + ctx.ego.length / 2.0:
            ax = -3.5
        elif v < self.v_target:
            ax = self.a_max
        else:
            ax = 0.0
        cmd = PlannerCommand(ax_mps2=ax, yaw_rate_radps=self.yaw_rate)
        cmd.planned_trajectory = project_forward(ctx.ego, cmd)
        return cmd


class OvertakePlanner:
    """Approach a slow lead, change lanes left, pass it, return to lane.

    Internal state machine::

        approach   →  merge_left  →  pass  →  merge_right  →  cruise

    Transitions:

    - ``approach → merge_left`` when an in-lane lead is closer than
      ``trigger_gap_m``.
    - ``merge_left → pass`` once the ego's lateral error to the target
      offset is below ``lateral_tol_m``.
    - ``pass → merge_right`` once the leader has fallen behind by at least
      ``clear_gap_m`` longitudinally in the ego frame.
    - ``merge_right → cruise`` once back near the original lateral position.

    Steering uses a small P-on-position / P-on-heading cascade: lateral
    error → target heading (capped) → yaw-rate command. Speed is governed
    by simple P-control toward a phase-specific target.
    """

    name = "overtake"

    PHASES = ("approach", "merge_left", "pass", "merge_right", "cruise")

    def __init__(
        self,
        cruise_speed_mps: float = 11.0,
        overtake_speed_mps: float = 13.0,
        lateral_offset_m: float = 3.5,
        trigger_gap_m: float = 30.0,
        clear_gap_m: float = 8.0,
        max_accel_mps2: float = 1.5,
        # Yaw rate cap chosen so that |ay| = v * yaw stays inside the comfort
        # threshold at typical speeds: 0.08 rad/s × 12 m/s ≈ 0.96 m/s² < 1.5.
        max_yaw_rate_radps: float = 0.08,
        lateral_tol_m: float = 0.25,
        # Heading cap chosen so the lateral velocity component (~v·sin(h)) is
        # ~1.4 m/s at cruise speed — gentle, but enough to cross a lane in a
        # few seconds.
        heading_cap_rad: float = 0.12,
        respect_speed_limit: bool = True,
    ):
        self.cruise_v = cruise_speed_mps
        self.overtake_v = overtake_speed_mps
        self.lateral_offset = lateral_offset_m
        self.trigger_gap = trigger_gap_m
        self.clear_gap = clear_gap_m
        self.max_a = max_accel_mps2
        self.max_yaw = max_yaw_rate_radps
        self.lat_tol = lateral_tol_m
        self.heading_cap = heading_cap_rad
        self.respect_speed_limit = respect_speed_limit

        self._phase: str = "approach"
        self._anchor_y: Optional[float] = None
        self._anchor_heading: Optional[float] = None

    def reset(self) -> None:
        self._phase = "approach"
        self._anchor_y = None
        self._anchor_heading = None

    # ----- low-level controllers -----

    def _ax_to(self, v_current: float, v_target: float, v_limit: Optional[float]) -> float:
        if self.respect_speed_limit and v_limit is not None:
            v_target = min(v_target, v_limit)
        err = v_target - v_current
        return _clamp(err * 1.5, -3.0, self.max_a)

    def _yaw_for_lateral(
        self, ego_y: float, ego_heading: float, target_y: float, ref_heading: float
    ) -> float:
        # Lateral error → desired heading (saturated), then heading error → yaw rate.
        lat_err = target_y - ego_y
        target_heading = ref_heading + _clamp(lat_err * 0.6, -self.heading_cap, self.heading_cap)
        head_err = _wrap_pi(target_heading - ego_heading)
        return _clamp(head_err * 2.5, -self.max_yaw, self.max_yaw)

    # ----- planning -----

    def plan(self, ctx: SceneContext) -> PlannerCommand:
        ego = ctx.ego
        if self._anchor_y is None:
            self._anchor_y = ego.pose.y
            self._anchor_heading = ego.pose.heading

        assert self._anchor_y is not None and self._anchor_heading is not None
        anchor_y = self._anchor_y
        anchor_h = self._anchor_heading
        target_y_left = anchor_y + self.lateral_offset

        v_limit = ctx.ego_speed_limit_mps

        if self._phase == "approach":
            lead = ctx.lead_agent()
            if lead is not None and lead.longitudinal_distance_m < self.trigger_gap:
                self._phase = "merge_left"
            ax = self._ax_to(ego.speed, self.cruise_v, v_limit)
            yaw = self._yaw_for_lateral(ego.pose.y, ego.pose.heading, anchor_y, anchor_h)

        elif self._phase == "merge_left":
            ax = self._ax_to(ego.speed, self.overtake_v, v_limit)
            yaw = self._yaw_for_lateral(ego.pose.y, ego.pose.heading, target_y_left, anchor_h)
            if abs(ego.pose.y - target_y_left) < self.lat_tol:
                self._phase = "pass"

        elif self._phase == "pass":
            ax = self._ax_to(ego.speed, self.overtake_v, v_limit)
            yaw = self._yaw_for_lateral(ego.pose.y, ego.pose.heading, target_y_left, anchor_h)
            # Leader passed = any agent now behind ego in the ego frame by clear_gap.
            cos_h = math.cos(ego.pose.heading)
            sin_h = math.sin(ego.pose.heading)
            for a in ctx.snapshot.agents:
                dx = a.pose.x - ego.pose.x
                dy = a.pose.y - ego.pose.y
                lon = dx * cos_h + dy * sin_h
                if lon < -self.clear_gap:
                    self._phase = "merge_right"
                    break

        elif self._phase == "merge_right":
            ax = self._ax_to(ego.speed, self.cruise_v, v_limit)
            yaw = self._yaw_for_lateral(ego.pose.y, ego.pose.heading, anchor_y, anchor_h)
            if abs(ego.pose.y - anchor_y) < self.lat_tol:
                self._phase = "cruise"

        else:  # "cruise"
            ax = self._ax_to(ego.speed, self.cruise_v, v_limit)
            yaw = self._yaw_for_lateral(ego.pose.y, ego.pose.heading, anchor_y, anchor_h)

        cmd = PlannerCommand(ax_mps2=ax, yaw_rate_radps=yaw)
        cmd.planned_trajectory = project_forward(ego, cmd)
        return cmd

    @property
    def phase(self) -> str:
        return self._phase


class UrbanDrivingPlanner:
    """Composite urban planner.

    Layers:

    1. **Free-flow / car-following** — Treiber's IDM term with the desired
       speed capped at the posted limit.
    2. **Traffic-light awareness** — looks ahead along the ego heading for any
       lane connector showing RED (and YELLOW with safe stopping); computes
       the deceleration required to stop *before* the connector entry. The
       planner commits to that deceleration when it exceeds IDM's.
    3. **Stop-sign awareness** — if a stop polyline sits ahead within the
       lookahead window, commit to a hard stop just before it. Releases the
       brake once the ego has been below ``stop_release_speed_mps`` for one
       tick beyond the line (so 8r0 doesn't fire after the legal stop).
    4. **Pedestrian yield** — if a pedestrian is on or within ``ped_buffer_m``
       of a crosswalk in the ego's forward sweep, command a stop *before*
       the crosswalk. Releases once the pedestrian has cleared the corridor.

    The four hazards are evaluated independently each tick and the planner
    obeys whichever requires the largest deceleration. The trajectory
    forward-projection used by the visualiser is the same kinematic roll-out
    the other planners use.
    """

    name = "urban"

    def __init__(
        self,
        desired_speed_mps: float = 11.0,
        time_headway_s: float = 1.5,
        min_gap_m: float = 2.0,
        max_accel_mps2: float = 1.5,
        comfort_decel_mps2: float = 2.5,
        emergency_decel_mps2: float = 4.0,
        tl_lookahead_m: float = 35.0,
        stop_sign_lookahead_m: float = 30.0,
        ped_lookahead_m: float = 25.0,
        ped_buffer_m: float = 1.5,
        stop_buffer_m: float = 2.0,
        stop_release_speed_mps: float = 0.2,
        respect_speed_limit: bool = True,
    ):
        self._idm = IDMPlanner(
            desired_speed_mps=desired_speed_mps,
            time_headway_s=time_headway_s,
            min_gap_m=min_gap_m,
            max_accel_mps2=max_accel_mps2,
            comfort_decel_mps2=comfort_decel_mps2,
            respect_speed_limit=respect_speed_limit,
        )
        self.emergency_decel_mps2 = emergency_decel_mps2
        self.tl_lookahead_m = tl_lookahead_m
        self.stop_sign_lookahead_m = stop_sign_lookahead_m
        self.ped_lookahead_m = ped_lookahead_m
        self.ped_buffer_m = ped_buffer_m
        self.stop_buffer_m = stop_buffer_m
        self.stop_release_speed_mps = stop_release_speed_mps
        self._cleared_stop_lines: set[str] = set()

    def reset(self) -> None:
        self._idm.reset()
        self._cleared_stop_lines.clear()

    # ----- hazard distance helpers (return distance ahead, or None) -----

    def _distance_along_heading(self, ctx: SceneContext, x: float, y: float) -> Optional[float]:
        ego = ctx.ego
        dx = x - ego.pose.x
        dy = y - ego.pose.y
        lon = dx * math.cos(ego.pose.heading) + dy * math.sin(ego.pose.heading)
        return lon if lon > 0 else None

    def _near_edge_distance(self, ctx: SceneContext, polygon) -> Optional[float]:
        """Longitudinal distance from ego center to the nearest polygon
        vertex in front of the ego (along the ego heading)."""
        ego = ctx.ego
        cos_h = math.cos(ego.pose.heading)
        sin_h = math.sin(ego.pose.heading)
        best: Optional[float] = None
        for px, py in polygon:
            dx = px - ego.pose.x
            dy = py - ego.pose.y
            lon = dx * cos_h + dy * sin_h
            if lon <= 0:
                continue
            if best is None or lon < best:
                best = lon
        return best

    def _distance_to_red_tl(self, ctx: SceneContext) -> Optional[float]:
        """Distance ahead to the controlling stop polyline of any RED/YELLOW
        connector, falling back to the connector centerline start if no stop
        polyline is associated.
        """
        best: Optional[float] = None
        # Build a (lane_id → stop_polyline_midpoint) map so we can prefer the
        # stop polyline as the stopping target.
        sl_for_lane: dict[str, tuple[float, float]] = {}
        for sl in ctx.snapshot.map.stop_lines:
            if sl.associated_lane_id and sl.polyline and len(sl.polyline) >= 2:
                mx = sum(p[0] for p in sl.polyline) / len(sl.polyline)
                my = sum(p[1] for p in sl.polyline) / len(sl.polyline)
                sl_for_lane[sl.associated_lane_id] = (mx, my)
        for tl in ctx.snapshot.traffic_lights:
            state = tl.state.value if hasattr(tl.state, "value") else str(tl.state)
            if state not in (TrafficLightState.RED.value, TrafficLightState.YELLOW.value):
                continue
            lc = next(
                (c for c in ctx.snapshot.map.lane_connectors if c.lane_id == tl.lane_connector_id),
                None,
            )
            if lc is None:
                continue
            # Prefer any incoming-lane stop polyline; else use the connector
            # centerline start.
            target_xy: Optional[tuple[float, float]] = None
            for inc_id in lc.incoming_lane_ids:
                if inc_id in sl_for_lane:
                    target_xy = sl_for_lane[inc_id]
                    break
            if target_xy is None and lc.centerline:
                target_xy = (lc.centerline[0][0], lc.centerline[0][1])
            if target_xy is None:
                continue
            d = self._distance_along_heading(ctx, target_xy[0], target_xy[1])
            if d is None or d > self.tl_lookahead_m:
                continue
            if best is None or d < best:
                best = d
        return best

    def _distance_to_stop_sign(self, ctx: SceneContext) -> Tuple[Optional[float], Optional[str]]:
        best: Optional[float] = None
        best_id: Optional[str] = None
        for sl in ctx.snapshot.map.stop_lines:
            if not sl.polyline or len(sl.polyline) < 2:
                continue
            mx = sum(p[0] for p in sl.polyline) / len(sl.polyline)
            my = sum(p[1] for p in sl.polyline) / len(sl.polyline)
            d = self._distance_along_heading(ctx, mx, my)
            if d is None or d > self.stop_sign_lookahead_m:
                continue
            if sl.stop_line_id in self._cleared_stop_lines:
                continue
            if best is None or d < best:
                best = d
                best_id = sl.stop_line_id
        return best, best_id

    def _distance_to_crosswalk_with_ped(self, ctx: SceneContext) -> Optional[float]:
        """Distance from ego center to the near edge of any crosswalk in
        front of the ego that has a VRU on or within ``ped_buffer_m`` of it."""
        from shapely.geometry import Point

        best: Optional[float] = None
        for cw in ctx.snapshot.map.crosswalks:
            poly = polygon_from_points(cw.polygon)
            if poly is None:
                continue
            d = self._near_edge_distance(ctx, cw.polygon)
            if d is None or d > self.ped_lookahead_m:
                continue
            inflated = poly.buffer(self.ped_buffer_m)
            has_ped = False
            for a in ctx.snapshot.agents:
                if _type_value(a.object_type) not in VRU_TYPES:
                    continue
                if inflated.contains(Point(a.pose.x, a.pose.y)):
                    has_ped = True
                    break
            if not has_ped:
                continue
            if best is None or d < best:
                best = d
        return best

    @staticmethod
    def _decel_to_stop(v: float, distance_m: float) -> float:
        """Required (positive) deceleration to stop in ``distance_m``."""
        d = max(0.3, distance_m)
        return (v * v) / (2.0 * d)

    # ----- planning -----

    def plan(self, ctx: SceneContext) -> PlannerCommand:
        ego = ctx.ego
        v = ego.speed
        half_len = ego.length / 2.0

        idm_cmd = self._idm.plan(ctx)
        ax_idm = idm_cmd.ax_mps2

        # Find the most-restrictive (closest) active hazard.
        hazard_d: Optional[float] = None

        d_tl = self._distance_to_red_tl(ctx)
        if d_tl is not None:
            hazard_d = d_tl

        d_ss, ss_id = self._distance_to_stop_sign(ctx)
        if d_ss is not None and ss_id is not None:
            hazard_d = d_ss if hazard_d is None else min(hazard_d, d_ss)

        d_xwalk = self._distance_to_crosswalk_with_ped(ctx)
        if d_xwalk is not None:
            hazard_d = d_xwalk if hazard_d is None else min(hazard_d, d_xwalk)

        if hazard_d is None:
            ax = ax_idm
        else:
            # Convert hazard distance (ego center → hazard near edge) into a
            # stopping distance for the ego center: subtract half the ego
            # length so the front bumper aligns with the line, plus a safety
            # buffer.
            stopping_d = hazard_d - half_len - self.stop_buffer_m

            if v < self.stop_release_speed_mps and stopping_d < 5.0:
                # Hold stationary against the hazard until it clears.
                ax = -0.5
            else:
                # Brake at the larger of (minimum required, comfort decel)
                # so the planner commits firmly to the stop instead of
                # gliding toward the line at the bare-minimum rate.
                required = self._decel_to_stop(v, max(0.05, stopping_d))
                effective = min(
                    max(required, self._idm.b),  # at least comfort decel
                    self.emergency_decel_mps2,
                )
                ax = -effective

        # Mark the stop sign cleared once the ego has come to rest near it,
        # so we don't re-trigger 8r0 after the legal stop.
        if (
            d_ss is not None
            and ss_id is not None
            and v <= self.stop_release_speed_mps
            and d_ss - half_len < self.stop_buffer_m + 1.0
        ):
            self._cleared_stop_lines.add(ss_id)

        ax = max(-self.emergency_decel_mps2, min(self._idm.a, ax))
        cmd = PlannerCommand(ax_mps2=ax, yaw_rate_radps=0.0)
        cmd.planned_trajectory = project_forward(ego, cmd)
        return cmd


class LaneChangePlanner:
    """Approach a slow leader, then change lanes only when the target lane is
    *verified clear*. Otherwise wait behind the leader (IDM).

    State machine::

        follow  →  prepare_lane_change  →  merge  →  pass  →  return  →  follow

    Differs from :class:`OvertakePlanner` in two ways:

    - the merge waits until ``_target_lane_clear()`` is true (no agent in the
      target lane within the merge corridor), rather than firing on a
      simple gap threshold against the leader;
    - the longitudinal behaviour falls back to IDM during ``follow`` and
      ``prepare_lane_change`` so the planner doesn't run into the leader if
      it can't find a gap.
    """

    name = "lane_change"

    def __init__(
        self,
        cruise_speed_mps: float = 12.0,
        overtake_speed_mps: float = 14.0,
        lateral_offset_m: float = 3.5,
        trigger_thw_s: float = 2.0,
        merge_corridor_m: float = 30.0,
        clear_gap_m: float = 8.0,
        max_yaw_rate_radps: float = 0.08,
        heading_cap_rad: float = 0.12,
        lateral_tol_m: float = 0.25,
        respect_speed_limit: bool = True,
    ):
        self._idm = IDMPlanner(
            desired_speed_mps=cruise_speed_mps,
            time_headway_s=1.5,
            respect_speed_limit=respect_speed_limit,
        )
        self.cruise_v = cruise_speed_mps
        self.overtake_v = overtake_speed_mps
        self.lateral_offset = lateral_offset_m
        self.trigger_thw_s = trigger_thw_s
        self.merge_corridor_m = merge_corridor_m
        self.clear_gap_m = clear_gap_m
        self.max_yaw = max_yaw_rate_radps
        self.heading_cap = heading_cap_rad
        self.lat_tol = lateral_tol_m
        self.respect_speed_limit = respect_speed_limit

        self._phase: str = "follow"
        self._anchor_y: Optional[float] = None
        self._anchor_heading: Optional[float] = None

    def reset(self) -> None:
        self._idm.reset()
        self._phase = "follow"
        self._anchor_y = None
        self._anchor_heading = None

    @property
    def phase(self) -> str:
        return self._phase

    def _target_lane_clear(self, ctx: SceneContext, side: float) -> bool:
        """Is the target lane clear within ±``merge_corridor_m`` of ego?

        ``side`` is the signed target lateral offset (positive = left).
        """
        ego = ctx.ego
        cos_h = math.cos(ego.pose.heading)
        sin_h = math.sin(ego.pose.heading)
        for a in ctx.snapshot.agents:
            ot = _type_value(a.object_type)
            if ot not in VEHICLE_LIKE_TYPES:
                continue
            dx = a.pose.x - ego.pose.x
            dy = a.pose.y - ego.pose.y
            lon = dx * cos_h + dy * sin_h
            lat = -dx * sin_h + dy * cos_h
            # Project lat relative to the target lane offset.
            lat_rel = lat - side
            if abs(lat_rel) > 1.8:
                continue
            if -self.merge_corridor_m <= lon <= self.merge_corridor_m:
                return False
        return True

    def _yaw_for_lateral(self, ego_y, ego_h, target_y, ref_h):
        lat_err = target_y - ego_y
        target_h = ref_h + _clamp(lat_err * 0.6, -self.heading_cap, self.heading_cap)
        head_err = _wrap_pi(target_h - ego_h)
        return _clamp(head_err * 2.5, -self.max_yaw, self.max_yaw)

    def plan(self, ctx: SceneContext) -> PlannerCommand:
        ego = ctx.ego
        if self._anchor_y is None:
            self._anchor_y = ego.pose.y
            self._anchor_heading = ego.pose.heading
        anchor_y = self._anchor_y
        anchor_h = self._anchor_heading
        target_y_left = anchor_y + self.lateral_offset

        v_limit = ctx.ego_speed_limit_mps
        lead = ctx.lead_agent()

        # Always have an IDM fallback for ax during follow / prepare phases.
        idm_ax = self._idm.plan(ctx).ax_mps2

        if self._phase == "follow":
            ax = idm_ax
            yaw = self._yaw_for_lateral(ego.pose.y, ego.pose.heading, anchor_y, anchor_h)
            # Trigger when THW to leader < trigger_thw_s.
            if lead is not None:
                v = max(ego.speed, 1e-3)
                gap = max(0.1, lead.longitudinal_distance_m - (ego.length + lead.agent.length) / 2.0)
                thw = gap / v
                if thw < self.trigger_thw_s:
                    self._phase = "prepare_lane_change"

        elif self._phase == "prepare_lane_change":
            ax = idm_ax
            yaw = self._yaw_for_lateral(ego.pose.y, ego.pose.heading, anchor_y, anchor_h)
            if self._target_lane_clear(ctx, side=self.lateral_offset):
                self._phase = "merge"

        elif self._phase == "merge":
            ax = max(
                idm_ax,
                _clamp((self.overtake_v - ego.speed) * 1.0, -2.0, 1.8),
            )
            yaw = self._yaw_for_lateral(ego.pose.y, ego.pose.heading, target_y_left, anchor_h)
            if abs(ego.pose.y - target_y_left) < self.lat_tol:
                self._phase = "pass"

        elif self._phase == "pass":
            target_v = self.overtake_v
            if self.respect_speed_limit and v_limit is not None:
                target_v = min(target_v, v_limit)
            ax = _clamp((target_v - ego.speed) * 1.0, -2.0, 1.8)
            yaw = self._yaw_for_lateral(ego.pose.y, ego.pose.heading, target_y_left, anchor_h)
            cos_h = math.cos(ego.pose.heading)
            sin_h = math.sin(ego.pose.heading)
            for a in ctx.snapshot.agents:
                dx = a.pose.x - ego.pose.x
                dy = a.pose.y - ego.pose.y
                lon = dx * cos_h + dy * sin_h
                if lon < -self.clear_gap_m:
                    if self._target_lane_clear(ctx, side=0.0):
                        self._phase = "return"
                    break

        elif self._phase == "return":
            ax = _clamp((self.cruise_v - ego.speed) * 1.0, -2.0, 1.8)
            yaw = self._yaw_for_lateral(ego.pose.y, ego.pose.heading, anchor_y, anchor_h)
            if abs(ego.pose.y - anchor_y) < self.lat_tol:
                self._phase = "follow"

        else:
            ax = idm_ax
            yaw = self._yaw_for_lateral(ego.pose.y, ego.pose.heading, anchor_y, anchor_h)

        cmd = PlannerCommand(ax_mps2=ax, yaw_rate_radps=yaw)
        cmd.planned_trajectory = project_forward(ego, cmd)
        return cmd
