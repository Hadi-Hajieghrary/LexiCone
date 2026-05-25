"""CasADi/IPOPT nonlinear MPC trajectory planner.

The OCP is built **once** at construction time as a parametric :class:`casadi.Opti`
problem so per-tick solves only push fresh parameter values and (warm-)start the
decision variables from the previous solution shifted by one step.

State :math:`x = [p_x, p_y, \\psi, v]` at the rear axle; control
:math:`u = [a, \\delta]`. Constraints:

- discrete kinematic bicycle dynamics (RK4 from :mod:`.bicycle_model`),
- box bounds on speed, acceleration, steering angle,
- rate bounds on acceleration and steering,
- soft circular obstacle avoidance for the top-K nearest agents.

The cost penalises lateral/longitudinal tracking error against a reference path,
heading error (via :math:`1 - \\cos(\\Delta\\psi)`), speed deviation, control effort,
control rate, and the squared obstacle slacks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import casadi as ca
import numpy as np

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.common.actor_state.vehicle_parameters import VehicleParameters
from nuplan.common.geometry.compute import principal_value

from .bicycle_model import NU, NX, discrete_dynamics
from .reference_path import ReferencePath


logger = logging.getLogger(__name__)


@dataclass
class MPCWeights:
    pos: float = 1.0
    heading: float = 5.0
    speed: float = 0.5
    control: float = 0.1
    control_rate: float = 0.5
    slack: float = 1000.0


@dataclass
class MPCLimits:
    v_max: float = 25.0
    a_max: float = 2.5
    a_min: float = -3.0
    steer_max: float = 0.5
    steer_rate_max: float = 0.5
    jerk_max: float = 4.0


@dataclass
class ObstacleSnapshot:
    x: float
    y: float
    radius: float


@dataclass
class MPCParameters:
    """Tunable parameters for :class:`MPCTrajectoryPlanner`."""

    horizon_s: float = 2.0
    dt_s: float = 0.1
    desired_speed_mps: float = 12.0
    obstacle_slot_count: int = 8
    collision_buffer_m: float = 0.5
    weights: MPCWeights = field(default_factory=MPCWeights)
    limits: MPCLimits = field(default_factory=MPCLimits)
    solver_options: Dict[str, object] = field(default_factory=dict)


class MPCTrajectoryPlanner:
    """Nonlinear MPC built on CasADi Opti + IPOPT."""

    def __init__(self, vehicle_parameters: VehicleParameters, params: MPCParameters):
        self._params = params
        self._vehicle = vehicle_parameters
        self._wheel_base = vehicle_parameters.wheel_base
        self._ego_radius = 0.5 * float(
            np.hypot(vehicle_parameters.length, vehicle_parameters.width)
        )

        if params.dt_s <= 0:
            raise ValueError("dt_s must be > 0")
        self._horizon = int(round(params.horizon_s / params.dt_s))
        if self._horizon < 2:
            raise ValueError(f"horizon must be >= 2 steps, got {self._horizon}")

        self._step_fn = discrete_dynamics(self._wheel_base, params.dt_s)
        self._build_problem()

        self._prev_X_local: Optional[np.ndarray] = None
        self._prev_U: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Pickling — CasADi Opti/SX/Function are SwigPyObjects (not picklable);
    # drop them and rebuild on load.
    # ------------------------------------------------------------------

    _CASADI_ATTRS = (
        "_opti",
        "_X",
        "_U",
        "_slack",
        "_p_x0",
        "_p_Xref",
        "_p_u_prev",
        "_p_obstacles",
        "_p_v_cap",
        "_step_fn",
    )

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        for key in self._CASADI_ATTRS:
            state.pop(key, None)
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._step_fn = discrete_dynamics(self._wheel_base, self._params.dt_s)
        self._build_problem()

    @property
    def horizon_steps(self) -> int:
        return self._horizon

    @property
    def dt(self) -> float:
        return self._params.dt_s

    # ------------------------------------------------------------------
    # Problem construction (one-shot)
    # ------------------------------------------------------------------

    def _build_problem(self) -> None:
        N = self._horizon
        K = self._params.obstacle_slot_count
        w = self._params.weights
        lim = self._params.limits

        opti = ca.Opti()
        X = opti.variable(NX, N + 1)
        U = opti.variable(NU, N)
        slack = opti.variable(K, N) if K > 0 else None

        x0 = opti.parameter(NX)
        Xref = opti.parameter(NX, N + 1)
        u_prev = opti.parameter(NU)
        obstacles = opti.parameter(3, K) if K > 0 else None
        # Per-step velocity cap. We compute this each tick so that even when the ego
        # arrives at v > v_target the cap decays smoothly toward v_target at max_decel,
        # forcing the trajectory poses to imply a velocity the downstream LQR tracker
        # will actually respect.
        v_cap = opti.parameter(N + 1)

        # Initial condition.
        opti.subject_to(X[:, 0] == x0)

        # Dynamics.
        for k in range(N):
            x_next = self._step_fn(X[:, k], U[:, k])
            opti.subject_to(X[:, k + 1] == x_next)

        # State / control box constraints (element-wise: CasADi treats matrix-vs-scalar
        # comparisons element-wise but ``opti.bounded`` on a slice triggers a matrix inequality).
        for k in range(N + 1):
            opti.subject_to(X[3, k] >= 0.0)
            opti.subject_to(X[3, k] <= v_cap[k])
        for k in range(N):
            opti.subject_to(U[0, k] >= lim.a_min)
            opti.subject_to(U[0, k] <= lim.a_max)
            opti.subject_to(U[1, k] >= -lim.steer_max)
            opti.subject_to(U[1, k] <= lim.steer_max)

        # Rate constraints (first step uses u_prev as the reference).
        steer_rate_step = lim.steer_rate_max * self._params.dt_s
        jerk_step = lim.jerk_max * self._params.dt_s
        d_steer = U[1, 0] - u_prev[1]
        d_accel = U[0, 0] - u_prev[0]
        opti.subject_to(d_steer >= -steer_rate_step)
        opti.subject_to(d_steer <= steer_rate_step)
        opti.subject_to(d_accel >= -jerk_step)
        opti.subject_to(d_accel <= jerk_step)
        for k in range(1, N):
            d_steer_k = U[1, k] - U[1, k - 1]
            d_accel_k = U[0, k] - U[0, k - 1]
            opti.subject_to(d_steer_k >= -steer_rate_step)
            opti.subject_to(d_steer_k <= steer_rate_step)
            opti.subject_to(d_accel_k >= -jerk_step)
            opti.subject_to(d_accel_k <= jerk_step)

        # Obstacle avoidance (soft, circular).
        if K > 0:
            for j in range(K):
                for k in range(N):
                    opti.subject_to(slack[j, k] >= 0)
            ego_r = self._ego_radius
            for j in range(K):
                ox = obstacles[0, j]
                oy = obstacles[1, j]
                rj = obstacles[2, j]
                min_d = rj + ego_r
                for k in range(1, N + 1):
                    dx = X[0, k] - ox
                    dy = X[1, k] - oy
                    # Squared formulation; slack relaxes infeasibility when an agent overlaps an aggressive prediction.
                    opti.subject_to(dx * dx + dy * dy + slack[j, k - 1] >= min_d * min_d)

        # Stage cost.
        cost = 0
        for k in range(N + 1):
            ex = X[0, k] - Xref[0, k]
            ey = X[1, k] - Xref[1, k]
            ev = X[3, k] - Xref[3, k]
            d_psi = X[2, k] - Xref[2, k]
            cost += w.pos * (ex * ex + ey * ey)
            cost += w.heading * (1.0 - ca.cos(d_psi))
            cost += w.speed * ev * ev
        for k in range(N):
            cost += w.control * (U[0, k] ** 2 + U[1, k] ** 2)
            if k > 0:
                d_a = U[0, k] - U[0, k - 1]
                d_d = U[1, k] - U[1, k - 1]
                cost += w.control_rate * (d_a * d_a + d_d * d_d)
        if K > 0 and slack is not None:
            cost += w.slack * ca.sumsqr(slack)

        opti.minimize(cost)

        ipopt_opts = {
            "print_level": 0,
            "max_iter": 300,
            "tol": 1e-3,
            "acceptable_tol": 1e-2,
            "acceptable_iter": 5,
            "sb": "yes",
        }
        ipopt_opts.update(self._params.solver_options.get("ipopt", {}))  # type: ignore[arg-type]
        plugin_opts = {"print_time": 0, "expand": True}
        plugin_opts.update({k: v for k, v in self._params.solver_options.items() if k != "ipopt"})
        opti.solver("ipopt", plugin_opts, ipopt_opts)

        self._opti = opti
        self._X = X
        self._U = U
        self._slack = slack
        self._p_x0 = x0
        self._p_Xref = Xref
        self._p_u_prev = u_prev
        self._p_obstacles = obstacles
        self._p_v_cap = v_cap

    # ------------------------------------------------------------------
    # Per-tick solve
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self._prev_X_local = None
        self._prev_U = None

    def solve(
        self,
        ego_state: EgoState,
        reference: ReferencePath,
        obstacles: Sequence[ObstacleSnapshot],
    ) -> List[EgoState]:
        # nuPlan maps use UTM coordinates in the 1e5-1e6 range. Squared-distance penalties at that scale
        # destroy IPOPT's internal scaling, so the OCP is built once in a world frame but solved each tick
        # in an ego-local frame (origin at the rear axle at t=0, x-axis aligned with the ego heading).
        # Solutions are transformed back to world coordinates before being handed to nuPlan.
        N = self._horizon
        anchor_xy, anchor_psi = self._build_anchor(ego_state)

        x0_world = self._ego_state_to_vector(ego_state)
        x0_local = self._world_to_local_state(x0_world, anchor_xy, anchor_psi)
        xref_world = self._build_xref(ego_state, reference)
        xref_local = self._world_to_local_xref(xref_world, anchor_xy, anchor_psi)
        u_prev = self._initial_control(ego_state)
        obs_param_local = self._pack_obstacles_local(obstacles, anchor_xy, anchor_psi)

        self._opti.set_value(self._p_x0, x0_local)
        self._opti.set_value(self._p_Xref, xref_local)
        self._opti.set_value(self._p_u_prev, u_prev)
        if self._p_obstacles is not None:
            self._opti.set_value(self._p_obstacles, obs_param_local)
        self._opti.set_value(self._p_v_cap, self._build_v_cap(x0_local[3], xref_local[3, :]))

        if self._prev_X_local is not None and self._prev_U is not None:
            X_init = np.concatenate([self._prev_X_local[:, 1:], self._prev_X_local[:, -1:]], axis=1)
            X_init[:, 0] = x0_local
            U_init = np.concatenate([self._prev_U[:, 1:], self._prev_U[:, -1:]], axis=1)
        else:
            X_init = self._rollout(x0_local)
            U_init = np.zeros((NU, N))
        self._opti.set_initial(self._X, X_init)
        self._opti.set_initial(self._U, U_init)
        if self._slack is not None:
            self._opti.set_initial(self._slack, np.zeros((self._params.obstacle_slot_count, N)))

        try:
            sol = self._opti.solve()
            X_local = np.asarray(sol.value(self._X))
            U_sol = np.asarray(sol.value(self._U))
        except RuntimeError as exc:
            logger.warning("MPC IPOPT solve failed (%s); coasting instead.", exc)
            X_local, U_sol = self._coast_fallback(x0_local)

        self._prev_X_local = X_local
        self._prev_U = U_sol
        X_world = self._local_to_world_states(X_local, anchor_xy, anchor_psi)
        return self._states_from_solution(X_world, U_sol, ego_state)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _ego_state_to_vector(self, ego_state: EgoState) -> np.ndarray:
        rear = ego_state.rear_axle
        v = float(ego_state.dynamic_car_state.rear_axle_velocity_2d.magnitude())
        return np.array([rear.x, rear.y, rear.heading, v], dtype=np.float64)

    def _initial_control(self, ego_state: EgoState) -> np.ndarray:
        a0 = float(ego_state.dynamic_car_state.rear_axle_acceleration_2d.x)
        d0 = float(ego_state.tire_steering_angle)
        return np.array([a0, d0], dtype=np.float64)

    def _build_xref(self, ego_state: EgoState, reference: ReferencePath) -> np.ndarray:
        N = self._horizon
        dt = self._params.dt_s
        v_des = self._params.desired_speed_mps
        v_now = float(ego_state.dynamic_car_state.rear_axle_velocity_2d.magnitude())
        rear = ego_state.rear_axle
        s0, _ = reference.project(rear.point)

        xref = np.zeros((NX, N + 1))
        # Determine a target velocity capped by the reference's local speed limit.
        # The reference grid then advances at *this* velocity so the LQR tracker (which
        # estimates reference speed from pose displacement) does not see an over-speed
        # signal when the ego itself is currently faster than v_des.
        sample0 = reference.sample(s0)
        v_target = min(v_des, sample0.v_limit) if sample0.v_limit > 0 else v_des
        # Keep some forward progress on the first tick even when stopped, so the LQR
        # does not lock the ego in place near a red light.
        v_seed = max(min(v_now, v_target), 0.5 * v_target, 1.0)

        for k in range(N + 1):
            s_k = s0 + v_seed * dt * k
            sample = reference.sample(s_k)
            v_target_k = min(v_des, sample.v_limit) if sample.v_limit > 0 else v_des
            xref[0, k] = sample.x
            xref[1, k] = sample.y
            # Unwrap reference heading relative to ego so the cost ``1 - cos(dpsi)`` works smoothly.
            psi_ref = sample.psi
            if k == 0:
                psi_unwrapped = rear.heading + principal_value(psi_ref - rear.heading)
            else:
                psi_unwrapped = xref[2, k - 1] + principal_value(psi_ref - xref[2, k - 1])
            xref[2, k] = psi_unwrapped
            xref[3, k] = v_target_k
        return xref

    def _build_v_cap(self, v0: float, v_target_per_step: np.ndarray) -> np.ndarray:
        """Per-step upper bound on velocity.

        The cap starts at ``v0`` (so the initial-state constraint stays feasible) and
        decays at ``max_decel`` toward each step's reference velocity. This guarantees
        the planned trajectory's pose displacement never implies a velocity above the
        target, which is what the simulator's LQR tracker actually consumes.
        """
        N1 = v_target_per_step.shape[0]
        dt = self._params.dt_s
        max_decel = self._params.limits.a_min  # negative
        caps = np.empty(N1)
        cap = max(v0, 0.0)
        for k in range(N1):
            target = float(max(v_target_per_step[k], 0.0))
            cap = max(cap + max_decel * dt, target)
            caps[k] = cap if k > 0 else max(v0, target)
        return caps

    def _build_anchor(self, ego_state: EgoState) -> Tuple[np.ndarray, float]:
        rear = ego_state.rear_axle
        return np.array([rear.x, rear.y], dtype=np.float64), float(rear.heading)

    @staticmethod
    def _world_to_local_xy(xy_world: np.ndarray, anchor_xy: np.ndarray, anchor_psi: float) -> np.ndarray:
        c, s = np.cos(-anchor_psi), np.sin(-anchor_psi)
        rot = np.array([[c, -s], [s, c]])
        delta = xy_world - anchor_xy[:, None] if xy_world.ndim == 2 else xy_world - anchor_xy
        return rot @ delta if xy_world.ndim == 2 else rot @ delta

    def _world_to_local_state(self, x_world: np.ndarray, anchor_xy: np.ndarray, anchor_psi: float) -> np.ndarray:
        xy_local = self._world_to_local_xy(x_world[:2], anchor_xy, anchor_psi)
        psi_local = principal_value(x_world[2] - anchor_psi)
        return np.array([xy_local[0], xy_local[1], psi_local, x_world[3]], dtype=np.float64)

    def _world_to_local_xref(self, xref_world: np.ndarray, anchor_xy: np.ndarray, anchor_psi: float) -> np.ndarray:
        out = np.zeros_like(xref_world)
        out[:2, :] = self._world_to_local_xy(xref_world[:2, :], anchor_xy, anchor_psi)
        # Unwrap heading deltas to keep the cost ``1 - cos(dpsi)`` smooth across the horizon.
        prev = 0.0  # anchor heading in local frame is 0
        for k in range(xref_world.shape[1]):
            delta = principal_value(xref_world[2, k] - anchor_psi)
            unwrapped = prev + principal_value(delta - prev)
            out[2, k] = unwrapped
            prev = unwrapped
        out[3, :] = xref_world[3, :]
        return out

    def _local_to_world_states(self, X_local: np.ndarray, anchor_xy: np.ndarray, anchor_psi: float) -> np.ndarray:
        c, s = np.cos(anchor_psi), np.sin(anchor_psi)
        rot = np.array([[c, -s], [s, c]])
        xy_world = rot @ X_local[:2, :] + anchor_xy[:, None]
        psi_world = np.array([principal_value(anchor_psi + X_local[2, k]) for k in range(X_local.shape[1])])
        out = np.zeros_like(X_local)
        out[:2, :] = xy_world
        out[2, :] = psi_world
        out[3, :] = X_local[3, :]
        return out

    def _pack_obstacles_local(
        self, obstacles: Sequence[ObstacleSnapshot], anchor_xy: np.ndarray, anchor_psi: float
    ) -> np.ndarray:
        K = self._params.obstacle_slot_count
        arr = np.zeros((3, K))
        # Park unused slots far away so the corresponding constraint is trivially satisfied.
        arr[0, :] = 1e6
        arr[1, :] = 1e6
        arr[2, :] = 0.0
        for j, ob in enumerate(obstacles[:K]):
            local = self._world_to_local_xy(np.array([ob.x, ob.y]), anchor_xy, anchor_psi)
            arr[0, j] = float(local[0])
            arr[1, j] = float(local[1])
            arr[2, j] = max(ob.radius, 0.1)
        return arr

    def _rollout(self, x0: np.ndarray) -> np.ndarray:
        """Forward-roll x0 with zero controls for use as a dynamically consistent warm start."""
        N = self._horizon
        X = np.zeros((NX, N + 1))
        X[:, 0] = x0
        x_curr = x0.copy()
        u_zero = np.zeros(NU)
        for k in range(N):
            x_curr = np.asarray(self._step_fn(x_curr, u_zero)).flatten()
            X[:, k + 1] = x_curr
        return X

    def _coast_fallback(self, x0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        N = self._horizon
        dt = self._params.dt_s
        X = np.zeros((NX, N + 1))
        X[:, 0] = x0
        psi = x0[2]
        v = x0[3]
        for k in range(N):
            X[0, k + 1] = X[0, k] + v * np.cos(psi) * dt
            X[1, k + 1] = X[1, k] + v * np.sin(psi) * dt
            X[2, k + 1] = psi
            X[3, k + 1] = v
        U = np.zeros((NU, N))
        return X, U

    def _states_from_solution(
        self, X_sol: np.ndarray, U_sol: np.ndarray, ego_state: EgoState
    ) -> List[EgoState]:
        N = self._horizon
        dt_us = int(self._params.dt_s * 1e6)
        t0_us = int(ego_state.time_point.time_us)
        states: List[EgoState] = []
        for k in range(N + 1):
            px = float(X_sol[0, k])
            py = float(X_sol[1, k])
            psi = float(principal_value(X_sol[2, k]))
            v = float(X_sol[3, k])
            if k < N:
                a_k = float(U_sol[0, k])
                d_k = float(U_sol[1, k])
            else:
                a_k = float(U_sol[0, -1])
                d_k = float(U_sol[1, -1])
            t_us = t0_us + k * dt_us
            states.append(
                EgoState.build_from_rear_axle(
                    rear_axle_pose=StateSE2(px, py, psi),
                    rear_axle_velocity_2d=StateVector2D(v, 0.0),
                    rear_axle_acceleration_2d=StateVector2D(a_k, 0.0),
                    tire_steering_angle=d_k,
                    time_point=TimePoint(t_us),
                    vehicle_parameters=self._vehicle,
                )
            )
        return states
