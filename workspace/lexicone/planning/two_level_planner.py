"""Top-level two-level motion planner exposed to the nuPlan simulator.

A thin :class:`~nuplan.planning.simulation.planner.abstract_planner.AbstractPlanner`
subclass that orchestrates:

* a :class:`~lexicone.planning.global_planner.GlobalRoutePlanner` that re-extracts
  a lane-graph route every ``replan_period_s`` seconds and turns it into a
  reference path, and
* a per-tick MPC layer that solves a CasADi/IPOPT optimal-control problem
  to track that reference subject to kinematic and obstacle constraints.

The MPC layer has two flavours, selected by the ``penalty_form`` parameter:

- ``penalty_form: null`` (default) — legacy
  :class:`~lexicone.planning.trajectory_planner.MPCTrajectoryPlanner`: a single
  flat-weight nonlinear MPC. Backward-compatible with the original YAML keys.
- ``penalty_form: "l1"`` or ``"l2"`` — the new LCP-structured
  :class:`~lexicone.planning.lcp_mpc.LCPTrajectoryPlanner` with per-level
  epigraph slacks, applicability-masked rule constraints, and sequential
  linearisation. Wires together :mod:`~lexicone.planning.map_lifter`,
  :mod:`~lexicone.planning.rule_encoder`, :mod:`~lexicone.planning.calibration_cache`,
  and :mod:`~lexicone.planning.compliance_checker`.

The returned ``InterpolatedTrajectory`` is a list of ``EgoState`` samples spaced
``mpc_dt_s`` apart; the simulator picks the state at the next iteration time
via the InterpolatedTrajectory's linear interpolation.
"""

from __future__ import annotations

import logging
import math
from typing import List, Optional, Sequence, Tuple, Type, Union

import numpy as np

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.common.actor_state.vehicle_parameters import VehicleParameters, get_pacifica_parameters
from nuplan.common.geometry.compute import principal_value
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks, Observation
from nuplan.planning.simulation.planner.abstract_planner import (
    AbstractPlanner,
    PlannerInitialization,
    PlannerInput,
)
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory

from .bicycle_model import NU, NX
from .calibration_cache import CalibrationCache
from .compliance_checker import ComplianceChecker
from .global_planner import GlobalRoutePlanner
from .lcp_mpc import (
    LCPLevelSpec,
    LCPLimits,
    LCPParameters,
    LCPTrajectoryPlanner,
)
from .map_lifter import MapLifter
from .reference_path import ReferencePath, straight_reference
from .rule_encoder import AgentSlot, EncoderContext, RuleSet, make_default_ruleset
from .trajectory_planner import (
    MPCLimits,
    MPCParameters,
    MPCTrajectoryPlanner,
    MPCWeights,
    ObstacleSnapshot,
)

logger = logging.getLogger(__name__)


class TwoLevelMPCPlanner(AbstractPlanner):
    """Global route + MPC trajectory pipeline for the nuPlan closed-loop simulator."""

    requires_scenario: bool = False

    def __init__(
        self,
        mpc_horizon_s: float = 2.0,
        mpc_dt_s: float = 0.1,
        replan_period_s: float = 5.0,
        desired_speed_mps: float = 12.0,
        occupancy_map_radius_m: float = 40.0,
        global_lookahead_m: float = 200.0,
        obstacle_slot_count: int = 8,
        collision_buffer_m: float = 0.5,
        max_accel_mps2: float = 2.5,
        max_decel_mps2: float = 3.0,
        max_speed_mps: float = 25.0,
        max_steer_rad: float = 0.5,
        max_steer_rate_radps: float = 0.5,
        max_jerk_mps3: float = 4.0,
        weight_pos: float = 1.0,
        weight_heading: float = 5.0,
        weight_speed: float = 0.5,
        weight_control: float = 0.1,
        weight_control_rate: float = 0.5,
        weight_slack: float = 1000.0,
        # ------------------------------------------------------------------
        # LCP-mode (Lexicographic Constraint Programming) — opt-in.
        # When ``penalty_form`` is ``None`` (default), the planner uses the
        # legacy single-tier ``MPCTrajectoryPlanner`` and the constructor
        # parameters above. When ``penalty_form`` is ``"l1"`` or ``"l2"`` the
        # planner builds the new :class:`LCPTrajectoryPlanner` with the rule
        # encoders, the per-scenario-class calibration cache, and the
        # compliance checker.
        # ------------------------------------------------------------------
        penalty_form: Optional[str] = None,
        weights_per_level: Optional[Sequence[Union[float, str]]] = None,
        epsilon_per_level: Optional[Sequence[float]] = None,
        scenario_class_hint: str = "default",
        lcp_map_radius_m: float = 80.0,
        # Runtime knobs — see ``LCPParameters`` for semantics.
        runtime_mode: str = "ws",
        slp_max_iterations: int = 1,
        slp_residual_tol_m: float = 0.05,
        vehicle_parameters: Optional[VehicleParameters] = None,
    ):
        self._mpc_horizon_s = mpc_horizon_s
        self._mpc_dt_s = mpc_dt_s
        self._replan_period_s = replan_period_s
        self._desired_speed_mps = desired_speed_mps
        self._occupancy_map_radius_m = occupancy_map_radius_m
        self._global_lookahead_m = global_lookahead_m
        self._collision_buffer_m = collision_buffer_m
        self._vehicle_parameters = vehicle_parameters or get_pacifica_parameters()

        self._mpc_params = MPCParameters(
            horizon_s=mpc_horizon_s,
            dt_s=mpc_dt_s,
            desired_speed_mps=desired_speed_mps,
            obstacle_slot_count=obstacle_slot_count,
            collision_buffer_m=collision_buffer_m,
            weights=MPCWeights(
                pos=weight_pos,
                heading=weight_heading,
                speed=weight_speed,
                control=weight_control,
                control_rate=weight_control_rate,
                slack=weight_slack,
            ),
            limits=MPCLimits(
                v_max=max_speed_mps,
                a_max=max_accel_mps2,
                a_min=-abs(max_decel_mps2),
                steer_max=max_steer_rad,
                steer_rate_max=max_steer_rate_radps,
                jerk_max=max_jerk_mps3,
            ),
        )

        # Lazy state initialised in ``initialize`` / per tick.
        self._global: Optional[GlobalRoutePlanner] = None
        self._mpc: Optional[MPCTrajectoryPlanner] = None
        self._reference: Optional[ReferencePath] = None
        self._last_replan_s: Optional[float] = None
        self._fallback_reference_length_m = max(50.0, mpc_horizon_s * desired_speed_mps * 1.5)

        # ------------------------------------------------------------------
        # LCP-mode plumbing — only populated when ``penalty_form`` is set.
        # ------------------------------------------------------------------
        if penalty_form is not None and penalty_form not in ("l1", "l2"):
            raise ValueError(f"penalty_form must be one of 'l1', 'l2', or None; got {penalty_form!r}")
        self._penalty_form: Optional[str] = penalty_form
        self._scenario_class_hint = scenario_class_hint
        self._lcp_map_radius_m = lcp_map_radius_m
        self._lcp_weights_spec: Optional[List[Union[float, str]]] = (
            list(weights_per_level) if weights_per_level is not None else None
        )
        self._lcp_epsilon_per_level: Optional[List[float]] = (
            [float(e) for e in epsilon_per_level] if epsilon_per_level is not None else None
        )
        if runtime_mode not in ("ws", "cascade"):
            raise ValueError(f"runtime_mode must be 'ws' or 'cascade'; got {runtime_mode!r}")
        self._lcp_runtime_mode: str = runtime_mode
        self._lcp_slp_max_iter: int = int(slp_max_iterations)
        self._lcp_slp_residual_tol_m: float = float(slp_residual_tol_m)
        # Per-instance LCP state, populated in ``initialize`` when in LCP mode.
        self._lcp_planner: Optional[LCPTrajectoryPlanner] = None
        self._lcp_ruleset: Optional[RuleSet] = None
        self._lcp_map_lifter: Optional[MapLifter] = None
        self._lcp_cache: Optional[CalibrationCache] = None
        self._lcp_compliance: Optional[ComplianceChecker] = None
        self._lcp_prev_X_local: Optional[np.ndarray] = None
        self._lcp_prev_U: Optional[np.ndarray] = None
        self._lcp_b_eps_lex: Optional[List[bool]] = None

    # ------------------------------------------------------------------
    # AbstractPlanner contract
    # ------------------------------------------------------------------

    def name(self) -> str:
        return self.__class__.__name__

    def observation_type(self) -> Type[Observation]:
        return DetectionsTracks  # type: ignore[return-value]

    def initialize(self, initialization: PlannerInitialization) -> None:
        self._global = GlobalRoutePlanner(
            map_api=initialization.map_api,
            route_roadblock_ids=list(initialization.route_roadblock_ids),
            mission_goal=initialization.mission_goal,
            default_speed_limit_mps=self._desired_speed_mps,
            lookahead_m=self._global_lookahead_m,
        )
        # Legacy MPC is always built (it's the fallback path and is cheap to
        # construct). The LCP-mode planner is only built when penalty_form is set.
        self._mpc = MPCTrajectoryPlanner(
            vehicle_parameters=self._vehicle_parameters, params=self._mpc_params
        )
        self._reference = None
        self._last_replan_s = None

        if self._penalty_form is not None:
            self._initialize_lcp(initialization)

    def _initialize_lcp(self, initialization: PlannerInitialization) -> None:
        """Build the LCP-mode plumbing — runs only when ``penalty_form`` is set."""
        # Rule set determines per-level slot counts.
        ruleset = make_default_ruleset()
        slots_per_level = ruleset.slots_per_step_per_level()
        epsilon = self._lcp_epsilon_per_level or [1e-4, 4e-2, 5e-1]
        level_specs = tuple(
            LCPLevelSpec(name=name, slots_per_step=slots, epsilon_tolerance=eps)
            for name, slots, eps in zip(("safety", "legal", "comfort"), slots_per_level, epsilon)
        )

        # Calibration cache — resolves "auto" weight sentinels to per-scenario
        # cached calibrations (or to heuristic defaults on cache miss).
        cache = CalibrationCache()
        weights_spec = self._lcp_weights_spec or ["auto", "auto", "auto"]
        weights_resolved, cache_entry = cache.resolve_weights(
            weights_spec=weights_spec,
            scenario_class=self._scenario_class_hint,
            penalty_form=self._penalty_form,
            epsilon_per_level=epsilon if self._penalty_form == "l2" else None,
        )
        if cache_entry is None:
            logger.info(
                "LCP: no calibrated weights for scenario_class=%r (penalty=%s); "
                "using heuristic defaults %s",
                self._scenario_class_hint, self._penalty_form, weights_resolved,
            )
            self._lcp_b_eps_lex = None
        else:
            self._lcp_b_eps_lex = cache_entry.b_eps_lex
            logger.info(
                "LCP: loaded calibration for %r from cache (%s)",
                self._scenario_class_hint, cache_entry.computed_at,
            )

        params = LCPParameters(
            horizon_s=self._mpc_horizon_s,
            dt_s=self._mpc_dt_s,
            desired_speed_mps=self._desired_speed_mps,
            penalty_form=self._penalty_form,
            runtime_mode=self._lcp_runtime_mode,
            slp_max_iterations=self._lcp_slp_max_iter,
            slp_residual_tol_m=self._lcp_slp_residual_tol_m,
            level_specs=level_specs,
            weights_per_level=tuple(weights_resolved[: len(level_specs)]),
            limits=LCPLimits(
                v_max=self._mpc_params.limits.v_max,
                a_max=self._mpc_params.limits.a_max,
                a_min=self._mpc_params.limits.a_min,
                steer_max=self._mpc_params.limits.steer_max,
                steer_rate_max=self._mpc_params.limits.steer_rate_max,
                jerk_max=self._mpc_params.limits.jerk_max,
            ),
        )

        self._lcp_planner = LCPTrajectoryPlanner(self._vehicle_parameters, params)
        self._lcp_ruleset = ruleset
        try:
            self._lcp_map_lifter = MapLifter(
                map_api=initialization.map_api,
                route_roadblock_ids=list(initialization.route_roadblock_ids),
                radius_m=self._lcp_map_radius_m,
            )
        except Exception as exc:  # pragma: no cover - nuplan-devkit env-specific
            logger.warning("LCP: could not build MapLifter (%s); proceeding without map view.", exc)
            self._lcp_map_lifter = None
        self._lcp_cache = cache
        self._lcp_compliance = ComplianceChecker(epsilon_per_level=epsilon)
        self._lcp_prev_X_local = None
        self._lcp_prev_U = None

    def compute_planner_trajectory(self, current_input: PlannerInput) -> AbstractTrajectory:
        assert self._global is not None and self._mpc is not None, (
            "TwoLevelMPCPlanner.initialize() must be called before compute_planner_trajectory()"
        )
        ego_state, observation = current_input.history.current_state
        now_s = float(current_input.iteration.time_us) * 1e-6

        self._maybe_replan(ego_state, now_s)
        reference = self._reference or straight_reference(
            origin=ego_state.rear_axle,
            length_m=self._fallback_reference_length_m,
            speed_limit_mps=self._desired_speed_mps,
        )
        if self._reference is None:
            logger.warning(
                "TwoLevelMPCPlanner: no reference from global planner yet; using straight fallback."
            )

        # Branch on LCP-mode availability. Falling back to the legacy MPC on
        # exceptions keeps the demo robust to LCP integration glitches; in
        # practice this should never happen once the encoders are stable.
        if self._penalty_form is not None and self._lcp_planner is not None:
            try:
                states = self._solve_lcp_tick(ego_state, observation, reference, current_input)
                return InterpolatedTrajectory(states)
            except Exception as exc:  # pragma: no cover - defensive fallback
                logger.warning(
                    "LCP tick failed (%s); falling back to legacy MPC for this tick.", exc
                )

        obstacles = self._extract_obstacles(ego_state, observation)
        states = self._mpc.solve(ego_state, reference, obstacles)
        return InterpolatedTrajectory(states)

    # ------------------------------------------------------------------
    # LCP-mode per-tick path
    # ------------------------------------------------------------------

    def _solve_lcp_tick(
        self,
        ego_state: EgoState,
        observation: Observation,
        reference: ReferencePath,
        current_input: PlannerInput,
    ) -> List[EgoState]:
        """One LCP MPC tick.

        Mirrors the legacy MPC's solve flow but populates the LCP MPC's
        parameter slots via the rule encoders + map lifter. Operates entirely
        in the ego-local frame (origin at the rear axle, x-axis aligned with
        the ego heading).
        """
        assert self._lcp_planner is not None
        assert self._lcp_ruleset is not None
        N = self._lcp_planner.horizon_steps
        dt = self._lcp_planner.dt

        # Anchor: ego rear axle in world frame.
        rear = ego_state.rear_axle
        anchor_xy = np.array([rear.x, rear.y], dtype=np.float64)
        anchor_psi = float(rear.heading)

        # x0 in local frame is (0, 0, 0, v).
        v_now = float(ego_state.dynamic_car_state.rear_axle_velocity_2d.magnitude())
        x0_local = np.array([0.0, 0.0, 0.0, v_now])
        u_prev = np.array([
            float(ego_state.dynamic_car_state.rear_axle_acceleration_2d.x),
            float(ego_state.tire_steering_angle),
        ])

        # Warm-start: shift previous solution or roll forward with zero control.
        if self._lcp_prev_X_local is not None and self._lcp_prev_U is not None:
            X_bar = np.concatenate(
                [self._lcp_prev_X_local[:, 1:], self._lcp_prev_X_local[:, -1:]], axis=1
            )
            X_bar[:, 0] = x0_local
            U_bar = np.concatenate(
                [self._lcp_prev_U[:, 1:], self._lcp_prev_U[:, -1:]], axis=1
            )
        else:
            X_bar = self._roll_forward(x0_local, N, dt)
            U_bar = np.zeros((NU, N))

        # Linearise the dynamics around the warm-start (the SLP step).
        affine_steps = self._lcp_planner.linearisation().linearise_trajectory(X_bar, U_bar)

        # Reference path in local frame and per-step velocity cap.
        Xref_local = self._build_local_xref(reference, anchor_xy, anchor_psi, N, dt, v_now)
        v_cap = self._build_v_cap(v_now, Xref_local[3, :])

        # Lift map data into the ego-local frame.
        map_view = None
        if self._lcp_map_lifter is not None:
            try:
                map_view = self._lcp_map_lifter.view(
                    anchor_xy_world=anchor_xy,
                    anchor_heading_world=anchor_psi,
                    traffic_light_data=getattr(current_input, "traffic_light_data", None),
                )
            except Exception as exc:
                logger.debug("LCP: map view unavailable (%s); proceeding without map data.", exc)

        # Extract agents in ego-local frame.
        agents_local = self._extract_agents_local(observation, anchor_xy, anchor_psi)

        # Build the encoder context and encode every rule.
        ctx = EncoderContext(
            horizon_steps=N,
            dt_s=dt,
            warm_start_X=X_bar,
            warm_start_U=U_bar,
            Xref_local=Xref_local,
            agents_local=agents_local,
            map_view=map_view,
            desired_speed_mps=self._desired_speed_mps,
            ego_radius_m=self._lcp_planner._ego_radius,
            wheel_base_m=self._vehicle_parameters.wheel_base,
        )
        rule_pack = self._lcp_ruleset.encode_all(ctx)

        # SLP outer iteration: re-linearise the dynamics around the new
        # solution after each solve until the residual to the nonlinear
        # ground-truth dynamics is below ``slp_residual_tol_m``, or we hit
        # the iteration cap.
        X_local = X_bar.copy()
        U_sol = U_bar.copy()
        T_sol = [np.zeros((spec.slots_per_step, N)) for spec in self._lcp_planner._params.level_specs]
        nonlinear_step = self._lcp_planner.step_function()
        for outer_iter in range(max(1, self._lcp_slp_max_iter)):
            try:
                if self._lcp_runtime_mode == "cascade":
                    X_local, U_sol, T_sol = self._solve_cascade_one_iter(
                        affine_steps=affine_steps,
                        x0_local=x0_local,
                        u_prev=u_prev,
                        v_cap=v_cap,
                        Xref_local=Xref_local,
                        rule_pack=rule_pack,
                        X_init=X_local,
                        U_init=U_sol,
                    )
                else:
                    # WS mode: single solve at the calibrated weights.
                    self._lcp_planner.push_parameters(
                        affine_steps=affine_steps,
                        x0_local=x0_local,
                        u_prev=u_prev,
                        v_cap=v_cap,
                        Xref_local=Xref_local,
                        rule_pack=rule_pack,
                    )
                    self._lcp_planner.warm_start(X_local, U_sol)
                    X_local, U_sol, T_sol = self._lcp_planner.solve_once()
            except RuntimeError as exc:
                logger.warning(
                    "LCP %s solve failed at SLP iter %d (%s); coasting instead.",
                    self._lcp_runtime_mode, outer_iter, exc,
                )
                X_local = self._roll_forward(x0_local, N, dt)
                U_sol = np.zeros((NU, N))
                T_sol = [np.zeros((spec.slots_per_step, N)) for spec in self._lcp_planner._params.level_specs]
                break

            # Check SLP convergence; if good enough, stop iterating.
            from .slp_linearisation import sqp_convergence_metric
            max_res, _max_step = sqp_convergence_metric(
                affine_steps, X_local, U_sol, X_bar, U_bar, nonlinear_step
            )
            if max_res < self._lcp_slp_residual_tol_m:
                break
            # Re-linearise around the new solution for the next outer pass.
            affine_steps = self._lcp_planner.linearisation().linearise_trajectory(X_local, U_sol)
            X_bar = X_local.copy()
            U_bar = U_sol.copy()

        self._lcp_prev_X_local = X_local
        self._lcp_prev_U = U_sol

        # Transform back to world frame and build EgoStates.
        X_world = self._local_to_world(X_local, anchor_xy, anchor_psi)
        return self._states_from_solution(X_world, U_sol, ego_state, dt)

    def _solve_cascade_one_iter(
        self,
        affine_steps,
        x0_local,
        u_prev,
        v_cap,
        Xref_local,
        rule_pack,
        X_init,
        U_init,
    ):
        """One SLP outer iteration in cascade mode — runs the full
        L+1-stage lex cascade at the current linearisation."""
        from .lex_cascade import run_cascade

        result = run_cascade(
            base_params=self._lcp_planner._params,
            vehicle_parameters=self._vehicle_parameters,
            affine_steps=affine_steps,
            x0_local=x0_local,
            u_prev=u_prev,
            v_cap=v_cap,
            Xref_local=Xref_local,
            rule_pack=rule_pack,
        )
        return result.z_lex_X, result.z_lex_U, result.T_lex

    def _roll_forward(self, x0_local: np.ndarray, N: int, dt: float) -> np.ndarray:
        """Constant-control forward roll for the SLP warm start."""
        X = np.zeros((NX, N + 1))
        X[:, 0] = x0_local
        psi = x0_local[2]
        v = x0_local[3]
        for k in range(N):
            X[0, k + 1] = X[0, k] + v * math.cos(psi) * dt
            X[1, k + 1] = X[1, k] + v * math.sin(psi) * dt
            X[2, k + 1] = psi
            X[3, k + 1] = v
        return X

    def _build_local_xref(
        self,
        reference: ReferencePath,
        anchor_xy: np.ndarray,
        anchor_psi: float,
        N: int,
        dt: float,
        v_now: float,
    ) -> np.ndarray:
        """Sample the reference in arc length, then rotate/translate into local frame."""
        v_des = self._desired_speed_mps
        rear_point = type(  # build a lightweight Point2D without importing it explicitly
            "Point2D", (), {"x": float(anchor_xy[0]), "y": float(anchor_xy[1])}
        )()
        s0, _ = reference.project(rear_point)
        sample0 = reference.sample(s0)
        v_target = min(v_des, sample0.v_limit) if sample0.v_limit > 0 else v_des
        v_seed = max(min(v_now, v_target), 0.5 * v_target, 1.0)

        cos_h, sin_h = math.cos(-anchor_psi), math.sin(-anchor_psi)
        rot = np.array([[cos_h, -sin_h], [sin_h, cos_h]])

        xref_local = np.zeros((NX, N + 1))
        prev_unwrap = 0.0
        for k in range(N + 1):
            s_k = s0 + v_seed * dt * k
            sample = reference.sample(s_k)
            xy_local = rot @ (np.array([sample.x, sample.y]) - anchor_xy)
            xref_local[0, k] = xy_local[0]
            xref_local[1, k] = xy_local[1]
            psi_delta = principal_value(sample.psi - anchor_psi)
            unwrapped = prev_unwrap + principal_value(psi_delta - prev_unwrap)
            xref_local[2, k] = unwrapped
            prev_unwrap = unwrapped
            v_target_k = min(v_des, sample.v_limit) if sample.v_limit > 0 else v_des
            xref_local[3, k] = v_target_k
        return xref_local

    def _build_v_cap(self, v0: float, v_target_per_step: np.ndarray) -> np.ndarray:
        """Per-step velocity cap that decays from v0 to v_target at max_decel."""
        N1 = v_target_per_step.shape[0]
        dt = self._mpc_dt_s
        max_decel = self._mpc_params.limits.a_min
        caps = np.empty(N1)
        cap = max(v0, 0.0)
        for k in range(N1):
            target = float(max(v_target_per_step[k], 0.0))
            cap = max(cap + max_decel * dt, target)
            caps[k] = cap if k > 0 else max(v0, target)
        return caps

    def _extract_agents_local(
        self,
        observation: Observation,
        anchor_xy: np.ndarray,
        anchor_psi: float,
    ) -> Tuple[AgentSlot, ...]:
        """World → ego-local frame for every detected agent."""
        if not isinstance(observation, DetectionsTracks):
            return ()
        cos_h, sin_h = math.cos(-anchor_psi), math.sin(-anchor_psi)
        rot = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
        agents: List[AgentSlot] = []
        for det in observation.tracked_objects.tracked_objects:
            cx = float(det.center.x)
            cy = float(det.center.y)
            xy_local = rot @ (np.array([cx, cy]) - anchor_xy)
            # Velocity rotation (no translation).
            v = getattr(det, "velocity", None)
            vx_world = float(v.x) if v is not None else 0.0
            vy_world = float(v.y) if v is not None else 0.0
            v_local = rot @ np.array([vx_world, vy_world])
            box = det.box
            type_name = getattr(getattr(det, "tracked_object_type", None), "name", "")
            is_vru = type_name.upper() in {"PEDESTRIAN", "BICYCLE"}
            agents.append(
                AgentSlot(
                    track_id=str(getattr(det, "track_token", "agent")),
                    x=float(xy_local[0]),
                    y=float(xy_local[1]),
                    vx=float(v_local[0]),
                    vy=float(v_local[1]),
                    length=float(box.length),
                    width=float(box.width),
                    is_vru=is_vru,
                )
            )
        return tuple(agents)

    def _local_to_world(
        self, X_local: np.ndarray, anchor_xy: np.ndarray, anchor_psi: float
    ) -> np.ndarray:
        cos_h, sin_h = math.cos(anchor_psi), math.sin(anchor_psi)
        rot = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
        xy_world = rot @ X_local[:2, :] + anchor_xy[:, None]
        psi_world = np.array(
            [principal_value(anchor_psi + X_local[2, k]) for k in range(X_local.shape[1])]
        )
        out = np.zeros_like(X_local)
        out[:2, :] = xy_world
        out[2, :] = psi_world
        out[3, :] = X_local[3, :]
        return out

    def _states_from_solution(
        self,
        X_world: np.ndarray,
        U_sol: np.ndarray,
        ego_state: EgoState,
        dt: float,
    ) -> List[EgoState]:
        """Build the list-of-EgoStates the simulator expects."""
        N = U_sol.shape[1]
        dt_us = int(dt * 1e6)
        t0_us = int(ego_state.time_point.time_us)
        states: List[EgoState] = []
        for k in range(N + 1):
            px = float(X_world[0, k])
            py = float(X_world[1, k])
            psi = float(principal_value(X_world[2, k]))
            v = float(X_world[3, k])
            if k < N:
                a_k = float(U_sol[0, k])
                d_k = float(U_sol[1, k])
            else:
                a_k = float(U_sol[0, -1])
                d_k = float(U_sol[1, -1])
            states.append(
                EgoState.build_from_rear_axle(
                    rear_axle_pose=StateSE2(px, py, psi),
                    rear_axle_velocity_2d=StateVector2D(v, 0.0),
                    rear_axle_acceleration_2d=StateVector2D(a_k, 0.0),
                    tire_steering_angle=d_k,
                    time_point=TimePoint(t0_us + k * dt_us),
                    vehicle_parameters=self._vehicle_parameters,
                )
            )
        return states

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _maybe_replan(self, ego_state: EgoState, now_s: float) -> None:
        assert self._global is not None

        replan_reason: Optional[str] = None
        if self._reference is None or self._last_replan_s is None:
            replan_reason = "first plan"
        elif (now_s - self._last_replan_s) >= self._replan_period_s:
            replan_reason = "timer"
        else:
            # Replan early if the ego has drifted significantly off the current reference
            # (e.g. ego missed an exit or was pushed sideways by collision-avoidance).
            try:
                _, lat = self._reference.project(ego_state.rear_axle.point)
                if abs(lat) > 5.0:
                    replan_reason = f"off-reference lateral={lat:.2f} m"
            except Exception:  # pragma: no cover - defensive: numerical edge cases
                replan_reason = "projection failure"

        if replan_reason is None:
            return
        new_reference = self._global.plan(ego_state)
        if new_reference is None:
            if self._reference is None:
                logger.warning("TwoLevelMPCPlanner: initial global plan failed; will retry.")
            return
        logger.debug("TwoLevelMPCPlanner: replanned (%s).", replan_reason)
        self._reference = new_reference
        self._last_replan_s = now_s

    def _extract_obstacles(
        self, ego_state: EgoState, observation: Observation
    ) -> List[ObstacleSnapshot]:
        if not isinstance(observation, DetectionsTracks):
            return []
        ego_xy = np.array([ego_state.center.x, ego_state.center.y])
        snapshots: List[ObstacleSnapshot] = []
        for det in observation.tracked_objects.tracked_objects:
            dx = float(det.center.x) - ego_xy[0]
            dy = float(det.center.y) - ego_xy[1]
            dist = math.hypot(dx, dy)
            if dist > self._occupancy_map_radius_m:
                continue
            box = det.box
            radius = 0.5 * math.hypot(float(box.length), float(box.width)) + self._collision_buffer_m
            snapshots.append(
                ObstacleSnapshot(x=float(det.center.x), y=float(det.center.y), radius=radius)
            )
        # Keep the closest K (the MPC has a fixed obstacle-slot count; far ones add no value).
        snapshots.sort(key=lambda s: (s.x - ego_xy[0]) ** 2 + (s.y - ego_xy[1]) ** 2)
        return snapshots[: self._mpc_params.obstacle_slot_count]
