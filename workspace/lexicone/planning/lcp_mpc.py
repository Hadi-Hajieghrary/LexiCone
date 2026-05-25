"""LCP-structured convex MPC for the lexicographic-priority rulebook.

This module implements the per-tick convex optimal-control problem (OCP) that
realises Lexicographic Constraint Programming (LCP) — see
[References/lex_constraint_programming_report_v10_2.md](References/lex_constraint_programming_report_v10_2.md).

The OCP structure is:

.. math::

    \\min_{X, U, T}\\  J(X, U) + \\sum_{i=1}^L w_i V_i(T_i)

subject to:

- **Linearised dynamics** (Setting (P)/(Q) of the paper)
  ``X[:, k+1] = A_k X[:, k] + B_k U[:, k] + c_k``
  with ``(A_k, B_k, c_k)`` produced by :mod:`.slp_linearisation` at each
  outer SQP iteration. Linear in the decision variables, preserving convexity.
- **Hard actuator/state box constraints** on velocity, acceleration, steering,
  jerk, steering rate, and the per-step velocity cap (decaying to v_target).
- **Per-level rule constraints with epigraph slacks**:
  ``g_{i,j,k}^T [X[:, k]; U[:, k]] + e_{i,j,k} <= t_{i,j,k}``,
  ``t_{i,j,k} >= 0``,
  for every applicable ``(i, j, k)`` triple. ``i`` is the priority level
  (1..L=3), ``j`` indexes the constraint within the level, ``k`` is the step.
  Linearised by the rule encoder at the warm-start trajectory.
- **Per-level violation functional**:
  ``V_i = sum_k sum_j t_{i,j,k}`` (L₁) or ``V_i = sum_k sum_j t_{i,j,k}^2`` (L₂).
- **Tikhonov stabiliser** under L₁: ``+ ε_T sum_{i,j,k} t_{i,j,k}^2`` with
  ``ε_T = 1e-6`` — fully smooth, doesn't bias the L₁ semantics to first order.
- **Performance objective** ``J`` (level L+1=4): existing reference-tracking
  position/heading/velocity cost plus control regularisation.

Per-tick applicability masking: every rule constraint is multiplied by a
binary ``mask_{i,j,k}`` parameter set at runtime. When a rule does not apply
at step ``k`` (e.g., the relevant traffic light is GREEN), the encoder sets
``mask = 0`` and ``e = -BIG`` so the constraint is trivially satisfied with
``t = 0`` — no contribution to the cost.

Convexity. The dynamics are linear in decision variables; the rule
constraints are affine in decision variables (the encoder linearises them
around the warm-start); the cost is quadratic (L₂) or linear (L₁) +
Tikhonov. Therefore the OCP is a strictly convex QP, satisfying Theorem 4.1's
hypothesis (A3) at the linearised problem. Each SQP outer iteration solves
this convex QP and updates the warm-start; under standard SQP regularity
(Nocedal & Wright §18) the iterates converge to a Karush-Kuhn-Tucker point
of the original nonlinear problem.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import casadi as ca
import numpy as np

from .bicycle_model import NU, NX, discrete_dynamics
from .slp_linearisation import AffineDynamics, BicycleLinearisation

logger = logging.getLogger(__name__)


PenaltyForm = Literal["l1", "l2"]
RuntimeMode = Literal["ws", "cascade"]


@dataclass
class LCPLimits:
    """Hard actuator/state limits (physical, not rule-based)."""

    v_max: float = 25.0
    a_max: float = 2.5
    a_min: float = -3.5
    steer_max: float = 0.5
    steer_rate_max: float = 0.7
    jerk_max: float = 12.0


@dataclass
class LCPLevelSpec:
    """How many constraint slots level ``i`` has in the OCP.

    The MPC builds the OCP once with a fixed number of constraint slots per
    level (and per step). At runtime the rule encoder fills those slots with
    the active constraints; unused slots have ``mask = 0``.
    """

    name: str                    # human-readable, e.g. "safety"
    slots_per_step: int          # max number of (a, e) constraint rows per step at this level
    epsilon_tolerance: float     # ε_i — operator-supplied compliance tolerance per Section 7.2


@dataclass
class LCPParameters:
    """Tunable parameters for :class:`LCPTrajectoryPlanner`."""

    horizon_s: float = 3.0
    dt_s: float = 0.1
    desired_speed_mps: float = 12.0
    # Penalty form. L₁ → exact equivalence (Theorem 5.1). L₂ → tolerance
    # compliance (Proposition 7.2).
    penalty_form: PenaltyForm = "l1"
    # Runtime mode selects how each per-tick OCP is solved:
    # - ``ws``: single weighted-sum solve at the calibrated w† (paper §4-7).
    #          Fast (1 NLP per tick) but requires correct weight calibration
    #          to match lex semantics.
    # - ``cascade``: run the L+1-stage lex cascade per tick (paper §3, eq. 2).
    #               Slower (L+1 NLPs per tick) but formally lex-optimal by
    #               construction with no calibration required. Recommended
    #               for offline simulation where per-tick budget is relaxed.
    runtime_mode: RuntimeMode = "ws"
    # Sequential linearisation outer-iteration budget per tick. The orchestrator
    # iterates: solve → re-linearise around the new solution → re-solve, until
    # either ``slp_max_iterations`` is reached or the SLP residual (max
    # dynamics deviation between linearised and true nonlinear step) falls
    # below ``slp_residual_tol_m``. For real-time deployment use 1; for
    # offline cascade use 3-5 for convergence.
    slp_max_iterations: int = 1
    slp_residual_tol_m: float = 0.05
    # One spec per priority level. Priority is the *index*: level_specs[0] is
    # the highest priority (level 1 in the paper), level_specs[-1] is the
    # lowest constrained level (level L). The performance objective J is
    # implicit at level L+1 and is not represented here.
    level_specs: Tuple[LCPLevelSpec, ...] = ()
    # Calibrated weights per level (one per LCPLevelSpec). Use "auto" sentinels
    # in the YAML to defer to the calibration cache; resolved at construction.
    weights_per_level: Tuple[float, ...] = ()
    # Tikhonov regulariser on epigraph slacks (L₁ smoothing). 0 disables.
    tikhonov_slack: float = 1e-6
    # Hard actuator/state limits.
    limits: LCPLimits = field(default_factory=LCPLimits)
    # IPOPT solver options.
    solver_options: Dict[str, object] = field(default_factory=dict)
    # Performance-objective (efficiency level) cost weights.
    weight_pos: float = 4.0
    weight_heading: float = 10.0
    weight_speed: float = 1.0
    weight_control: float = 0.05
    weight_control_rate: float = 0.2


@dataclass
class LinearisedRuleConstraint:
    """Affine constraint produced by the rule encoder for one (i, j, k) triple.

    Encodes ``a^T x_k + b^T u_k + e <= t`` where ``t >= 0`` is the epigraph
    slack. When ``mask = 0`` the constraint is interpreted as inactive: the
    encoder sets ``e = -BIG`` so the constraint is trivially satisfied with
    ``t = 0``.
    """

    a: np.ndarray         # (NX,) coefficient on x_k
    b: np.ndarray         # (NU,) coefficient on u_k
    e: float              # scalar offset; ``a^T x + b^T u + e <= t``
    mask: float           # 1.0 if active, 0.0 if inactive


@dataclass
class LCPRulePack:
    """Per-tick collection of all linearised rule constraints, grouped by level.

    ``constraints_by_level[i][k]`` is a list of length ``level_specs[i].slots_per_step``
    of :class:`LinearisedRuleConstraint`. The MPC enforces these as
    ``a^T X[:, k] + b^T U[:, k] + e <= t_{i,j,k}`` for every ``(i, j, k)``.

    The encoder is responsible for filling EXACTLY ``slots_per_step`` rows
    per step per level (padding with inactive slots if fewer constraints
    actually apply).
    """

    constraints_by_level: List[List[List[LinearisedRuleConstraint]]]


# Sentinel value used in the offset of an inactive constraint so it is trivially
# satisfied with t = 0 regardless of the (x, u) values the MPC chooses.
_INACTIVE_OFFSET = -1e6


def make_inactive_constraint() -> LinearisedRuleConstraint:
    """A trivially-satisfied placeholder used to pad unused slots."""
    return LinearisedRuleConstraint(
        a=np.zeros(NX),
        b=np.zeros(NU),
        e=_INACTIVE_OFFSET,
        mask=0.0,
    )


class LCPTrajectoryPlanner:
    """Convex linearised MPC realising lexicographic constraint programming.

    Construction builds the parametric :class:`casadi.Opti` problem once
    with:
    - Decision variables: ``X`` (NX × (N+1)), ``U`` (NU × N),
      per-level slack matrices ``T_i`` of shape ``(slots_i, N)``.
    - Parameters: linearised dynamics ``(A_k, B_k, c_k)``, initial state
      ``x0``, previous control ``u_prev``, per-step velocity cap ``v_cap``,
      reference trajectory ``Xref`` for the efficiency objective, and a
      parameter block per (level, step, slot) of shape ``(NX + NU + 2,)``
      packing ``(a, b, e, mask)``.
    - Hard constraints: linearised dynamics, actuator/state boxes,
      jerk/steering-rate boxes, slack non-negativity, and the rule-constraint
      inequalities.
    - Cost: weighted L₁ or L₂ violations + Tikhonov + reference-tracking.

    The per-tick :meth:`solve` method:
    1. Builds the warm-start trajectory (shifted from previous solution).
    2. Linearises the bicycle dynamics around the warm-start (SLP).
    3. Asks the rule encoder for the per-tick rule constraints.
    4. Pushes everything into the parameter slots.
    5. Solves the convex QP via IPOPT.
    6. Optionally iterates the SLP linearisation (2-3 outer iterations).
    7. Returns the trajectory as a list of EgoStates.
    """

    def __init__(
        self,
        vehicle_parameters,
        params: LCPParameters,
    ) -> None:
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
        if not params.level_specs:
            raise ValueError("LCPParameters must specify at least one level_spec")
        if len(params.weights_per_level) != len(params.level_specs):
            raise ValueError(
                f"weights_per_level (len={len(params.weights_per_level)}) must match "
                f"level_specs (len={len(params.level_specs)})"
            )

        self._n_levels = len(params.level_specs)
        self._linearisation = BicycleLinearisation(self._wheel_base, params.dt_s)
        self._step_fn = discrete_dynamics(self._wheel_base, params.dt_s)
        self._build_problem()

        self._prev_X: Optional[np.ndarray] = None
        self._prev_U: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Pickling (mirrors MPCTrajectoryPlanner)
    # ------------------------------------------------------------------

    _CASADI_ATTRS = (
        "_opti", "_X", "_U", "_T", "_p_A", "_p_B", "_p_c", "_p_x0",
        "_p_u_prev", "_p_v_cap", "_p_Xref", "_p_rules", "_step_fn",
        "_linearisation",
    )

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        for key in self._CASADI_ATTRS:
            state.pop(key, None)
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._linearisation = BicycleLinearisation(self._wheel_base, self._params.dt_s)
        self._step_fn = discrete_dynamics(self._wheel_base, self._params.dt_s)
        self._build_problem()

    @property
    def horizon_steps(self) -> int:
        return self._horizon

    @property
    def dt(self) -> float:
        return self._params.dt_s

    @property
    def n_levels(self) -> int:
        return self._n_levels

    # ------------------------------------------------------------------
    # Problem construction
    # ------------------------------------------------------------------

    def _build_problem(self) -> None:
        N = self._horizon
        L = self._n_levels
        lim = self._params.limits

        opti = ca.Opti()

        # Decision variables.
        X = opti.variable(NX, N + 1)
        U = opti.variable(NU, N)
        T: List[ca.MX] = []
        for spec in self._params.level_specs:
            T.append(opti.variable(spec.slots_per_step, N))

        # Parameters.
        # Linearised dynamics, one per step.
        p_A = [opti.parameter(NX, NX) for _ in range(N)]
        p_B = [opti.parameter(NX, NU) for _ in range(N)]
        p_c = [opti.parameter(NX) for _ in range(N)]
        # Initial state, previous control.
        x0 = opti.parameter(NX)
        u_prev = opti.parameter(NU)
        # Per-step velocity cap (same idiom as the legacy planner).
        v_cap = opti.parameter(N + 1)
        # Reference trajectory for the efficiency-level performance objective.
        Xref = opti.parameter(NX, N + 1)
        # Per-rule constraint parameters: for each (level, step, slot) we
        # store (a, b, e, mask) packed as NX + NU + 2 entries in a single
        # parameter block. This keeps the parameter count manageable.
        p_rules: List[List[ca.MX]] = []
        for spec in self._params.level_specs:
            # Shape (NX + NU + 2, slots_per_step) per step.
            p_rules.append([opti.parameter(NX + NU + 2, spec.slots_per_step) for _ in range(N)])

        # ------------------------------------------------------------------
        # Hard constraints
        # ------------------------------------------------------------------

        opti.subject_to(X[:, 0] == x0)

        # Linearised dynamics — affine in decision variables.
        for k in range(N):
            x_next = p_A[k] @ X[:, k] + p_B[k] @ U[:, k] + p_c[k]
            opti.subject_to(X[:, k + 1] == x_next)

        # State / control box constraints.
        for k in range(N + 1):
            opti.subject_to(X[3, k] >= 0.0)
            opti.subject_to(X[3, k] <= v_cap[k])
        for k in range(N):
            opti.subject_to(U[0, k] >= lim.a_min)
            opti.subject_to(U[0, k] <= lim.a_max)
            opti.subject_to(U[1, k] >= -lim.steer_max)
            opti.subject_to(U[1, k] <= lim.steer_max)

        # Rate constraints.
        steer_rate_step = lim.steer_rate_max * self._params.dt_s
        jerk_step = lim.jerk_max * self._params.dt_s
        d_steer0 = U[1, 0] - u_prev[1]
        d_accel0 = U[0, 0] - u_prev[0]
        opti.subject_to(d_steer0 >= -steer_rate_step)
        opti.subject_to(d_steer0 <= steer_rate_step)
        opti.subject_to(d_accel0 >= -jerk_step)
        opti.subject_to(d_accel0 <= jerk_step)
        for k in range(1, N):
            d_steer_k = U[1, k] - U[1, k - 1]
            d_accel_k = U[0, k] - U[0, k - 1]
            opti.subject_to(d_steer_k >= -steer_rate_step)
            opti.subject_to(d_steer_k <= steer_rate_step)
            opti.subject_to(d_accel_k >= -jerk_step)
            opti.subject_to(d_accel_k <= jerk_step)

        # Per-rule epigraph constraints. For each (level, step, slot):
        #   a^T X[:, k] + b^T U[:, k] + e <= t_{i,j,k}   when mask = 1
        #   trivially satisfied                          when mask = 0
        # We encode both cases by multiplying (a, b, e) by the mask: when
        # mask = 0 the constraint reduces to 0 <= t, satisfied with t = 0.
        for i, spec in enumerate(self._params.level_specs):
            for k in range(N):
                block = p_rules[i][k]  # shape (NX + NU + 2, slots_per_step)
                for j in range(spec.slots_per_step):
                    a = block[0:NX, j]
                    b = block[NX:NX + NU, j]
                    e = block[NX + NU, j]
                    mask = block[NX + NU + 1, j]
                    lhs = mask * (a.T @ X[:, k] + b.T @ U[:, k] + e)
                    opti.subject_to(lhs <= T[i][j, k])
                    opti.subject_to(T[i][j, k] >= 0)

        # ------------------------------------------------------------------
        # Cost
        # ------------------------------------------------------------------

        cost = ca.MX(0)

        # Per-level violation functional + Tikhonov stabiliser (L₁ only).
        for i, spec in enumerate(self._params.level_specs):
            w_i = self._params.weights_per_level[i]
            if self._params.penalty_form == "l1":
                cost += w_i * ca.sum1(ca.sum2(T[i]))
                if self._params.tikhonov_slack > 0:
                    cost += self._params.tikhonov_slack * ca.sumsqr(T[i])
            elif self._params.penalty_form == "l2":
                cost += w_i * ca.sumsqr(T[i])
            else:
                raise ValueError(f"Unknown penalty_form: {self._params.penalty_form}")

        # Performance-objective (efficiency-level) reference tracking.
        for k in range(N + 1):
            ex = X[0, k] - Xref[0, k]
            ey = X[1, k] - Xref[1, k]
            ev = X[3, k] - Xref[3, k]
            d_psi = X[2, k] - Xref[2, k]
            cost += self._params.weight_pos * (ex * ex + ey * ey)
            cost += self._params.weight_heading * (1.0 - ca.cos(d_psi))
            cost += self._params.weight_speed * ev * ev
        for k in range(N):
            cost += self._params.weight_control * (U[0, k] ** 2 + U[1, k] ** 2)
            if k > 0:
                d_a = U[0, k] - U[0, k - 1]
                d_d = U[1, k] - U[1, k - 1]
                cost += self._params.weight_control_rate * (d_a * d_a + d_d * d_d)

        opti.minimize(cost)

        # IPOPT setup.
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

        # Stash everything.
        self._opti = opti
        self._X = X
        self._U = U
        self._T = T
        self._p_A = p_A
        self._p_B = p_B
        self._p_c = p_c
        self._p_x0 = x0
        self._p_u_prev = u_prev
        self._p_v_cap = v_cap
        self._p_Xref = Xref
        self._p_rules = p_rules

    # ------------------------------------------------------------------
    # Per-tick solve API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self._prev_X = None
        self._prev_U = None

    def push_parameters(
        self,
        affine_steps: Sequence[AffineDynamics],
        x0_local: np.ndarray,
        u_prev: np.ndarray,
        v_cap: np.ndarray,
        Xref_local: np.ndarray,
        rule_pack: LCPRulePack,
    ) -> None:
        """Populate every parameter slot of the OCP for one tick.

        After calling this the caller can invoke :meth:`solve_once` or
        :meth:`solve_sqp` to obtain a trajectory.
        """
        N = self._horizon
        if len(affine_steps) != N:
            raise ValueError(f"need {N} AffineDynamics, got {len(affine_steps)}")

        for k, step in enumerate(affine_steps):
            self._opti.set_value(self._p_A[k], step.A)
            self._opti.set_value(self._p_B[k], step.B)
            self._opti.set_value(self._p_c[k], step.c)
        self._opti.set_value(self._p_x0, x0_local)
        self._opti.set_value(self._p_u_prev, u_prev)
        self._opti.set_value(self._p_v_cap, v_cap)
        self._opti.set_value(self._p_Xref, Xref_local)

        if len(rule_pack.constraints_by_level) != self._n_levels:
            raise ValueError(
                f"rule_pack has {len(rule_pack.constraints_by_level)} levels, "
                f"expected {self._n_levels}"
            )
        for i, spec in enumerate(self._params.level_specs):
            level_constraints = rule_pack.constraints_by_level[i]
            if len(level_constraints) != N:
                raise ValueError(
                    f"level {i} ({spec.name}) has {len(level_constraints)} step entries, "
                    f"expected {N}"
                )
            for k in range(N):
                step_constraints = level_constraints[k]
                if len(step_constraints) != spec.slots_per_step:
                    raise ValueError(
                        f"level {i} step {k} has {len(step_constraints)} constraints, "
                        f"expected slots_per_step={spec.slots_per_step}"
                    )
                block = np.zeros((NX + NU + 2, spec.slots_per_step))
                for j, c in enumerate(step_constraints):
                    block[0:NX, j] = c.a
                    block[NX:NX + NU, j] = c.b
                    block[NX + NU, j] = c.e
                    block[NX + NU + 1, j] = c.mask
                self._opti.set_value(self._p_rules[i][k], block)

    def warm_start(
        self,
        X_init: np.ndarray,
        U_init: np.ndarray,
        T_init: Optional[Sequence[np.ndarray]] = None,
    ) -> None:
        """Set the initial guess for the decision variables."""
        self._opti.set_initial(self._X, X_init)
        self._opti.set_initial(self._U, U_init)
        if T_init is not None:
            for i, T_i in enumerate(T_init):
                self._opti.set_initial(self._T[i], T_i)

    def solve_once(self) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
        """Solve the QP at the currently-pushed parameters.

        Returns
        -------
        ``(X_sol, U_sol, T_sol)`` where ``T_sol`` is a list of per-level
        slack matrices.
        """
        sol = self._opti.solve()
        X_sol = np.asarray(sol.value(self._X))
        U_sol = np.asarray(sol.value(self._U))
        T_sol = [np.asarray(sol.value(self._T[i])) for i in range(self._n_levels)]
        return X_sol, U_sol, T_sol

    def per_level_violation(self, T_sol: Sequence[np.ndarray]) -> np.ndarray:
        """Compute ``V_i`` from the slack matrices for the active penalty form.

        For L₁: ``V_i = sum_{j, k} t_{i,j,k}``.
        For L₂: ``V_i = sum_{j, k} t_{i,j,k}^2``.
        Returns a length-L array of non-negative scalars.
        """
        V = np.zeros(self._n_levels)
        for i, T_i in enumerate(T_sol):
            if self._params.penalty_form == "l1":
                V[i] = float(np.sum(np.maximum(T_i, 0.0)))
            elif self._params.penalty_form == "l2":
                V[i] = float(np.sum(np.maximum(T_i, 0.0) ** 2))
        return V

    # ------------------------------------------------------------------
    # Internal access (used by lex_cascade)
    # ------------------------------------------------------------------

    def opti(self) -> ca.Opti:
        """Expose the underlying Opti for the cascade's per-stage modifications."""
        return self._opti

    def step_function(self):
        """Expose the nonlinear RK4 step function (for SQP residual checks)."""
        return self._step_fn

    def linearisation(self) -> BicycleLinearisation:
        return self._linearisation
