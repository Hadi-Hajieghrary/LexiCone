"""Lexicographic cascade for offline weight calibration.

This module realises the L+1-stage cascade of equation (2) in the LCP paper:

.. math::

    V_i^\\star := \\min_{z \\in \\mathcal{F}_{i-1}} V_i(z), \\qquad
    \\mathcal{F}_i := \\{z \\in \\mathcal{F}_{i-1} : V_i(z) = V_i^\\star\\}

ending with the performance objective ``J`` minimised on :math:`\\mathcal{F}_L`.

The cascade is the **gold-standard reference solution** the paper's Algorithm
1A and 1B calibrate the WS weights against. It is too slow to run at 10 Hz
during normal MPC operation, but it is exactly what we need *offline* — once
per scenario class — to extract the lex active set and the lex achievement
point :math:`p^\\star = (V_1^\\star, V_2^\\star, V_3^\\star, J^\\star)`.

Concrete implementation
-----------------------

The cascade reuses the :class:`~.lcp_mpc.LCPTrajectoryPlanner` skeleton (so the
constraints, dynamics linearisation, actuator bounds, and rule parameters are
all shared). The trick is that at each stage we *change the objective* of the
underlying CasADi :class:`~casadi.Opti` problem to minimise only :math:`V_i`,
and we *add a constraint* :math:`V_{i'}(T) \\leq V_{i'}^\\star + \\delta_{\\text{lex}}`
for every previously-resolved level :math:`i' < i`. The constraint includes a
small numerical slack :math:`\\delta_{\\text{lex}} = 10^{-6}` to keep the
non-convex active manifold tractable under SQP linearisation (Section 7.2 of
the paper recommends this).

Algorithm sketch
----------------

1. Build the LCP MPC with all rules in place.
2. For each stage ``i = 1, ..., L``:
   a. Replace the objective with :math:`V_i(T)` only (no performance term,
      no other-level cost).
   b. Add inequality constraints
      :math:`\\sum_{j, k} t_{i', j, k} \\leq V_{i'}^\\star + \\delta_{\\text{lex}}`
      (L₁) or analogous L₂ form, for every :math:`i' < i`.
   c. Solve. Record :math:`V_i^\\star`.
3. Final stage ``L + 1``: restore the original performance objective :math:`J`,
   with all violation budgets :math:`V_{i'}^\\star + \\delta` constraints in
   place. Solve. Record :math:`J^\\star` and the active set.

The full solution :math:`p^\\star = (V_1^\\star, \\ldots, V_L^\\star, J^\\star)`
is returned along with the trajectory and active-set metadata that Algorithm
1A / 1B consume.

Because the cascade modifies the LCP MPC's Opti object in place, we operate
on a *snapshot* — each :func:`run_cascade` call builds a fresh
``LCPTrajectoryPlanner`` instance from the same construction parameters, so
no cross-call contamination.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import casadi as ca
import numpy as np

from .lcp_mpc import LCPParameters, LCPRulePack, LCPTrajectoryPlanner
from .slp_linearisation import AffineDynamics

logger = logging.getLogger(__name__)


DELTA_LEX: float = 1e-6
"""Numerical slack on the V_i ≤ V_i* + δ_lex cascade constraint.

The paper (Section 7.2 / Algorithm 1A Step A1) recommends a small positive
slack here so that the second-stage feasibility set is open in the L_2 regime
where the boundary is a measure-zero manifold; for L_1 the slack is
algorithmically irrelevant but kept for symmetry."""


@dataclass
class CascadeStageResult:
    """Outcome of one stage of the cascade.

    For stages 1..L, ``V_star_stage`` is :math:`V_i^\\star` for the *current*
    stage's level; ``V_star_all`` carries the cumulative budget vector
    :math:`(V_1^\\star, \\ldots, V_i^\\star)` recorded so far.
    """

    stage_index: int                # 1..L for violation stages, L+1 for performance
    is_performance_stage: bool      # True iff this is the J-minimisation final stage
    V_star_stage: float             # V_i* (or J* for performance stage)
    X_sol: np.ndarray
    U_sol: np.ndarray
    T_sol: List[np.ndarray]
    V_star_all: List[float]         # cumulative V_i* values for i <= stage_index


@dataclass
class CascadeResult:
    """Full output of :func:`run_cascade`."""

    p_star: np.ndarray              # (V_1*, ..., V_L*, J*)
    z_lex_X: np.ndarray             # state trajectory at lex optimum
    z_lex_U: np.ndarray             # control trajectory at lex optimum
    T_lex: List[np.ndarray]         # per-level slack matrices at lex optimum
    stage_results: List[CascadeStageResult] = field(default_factory=list)
    active_set: List[Tuple[int, int, int]] = field(default_factory=list)
    """List of (level, slot_index, step_index) tuples where the slack ``t > δ_lex``.

    The active set identifies which rule constraints are *boundary-binding* or
    *violated* at the lex optimum — the data Algorithm 1A's KKT system needs
    to construct the equivalence region polyhedron."""


def run_cascade(
    base_params: LCPParameters,
    vehicle_parameters,
    affine_steps: Sequence[AffineDynamics],
    x0_local: np.ndarray,
    u_prev: np.ndarray,
    v_cap: np.ndarray,
    Xref_local: np.ndarray,
    rule_pack: LCPRulePack,
    delta_lex: float = DELTA_LEX,
    active_threshold: float = 1e-4,
) -> CascadeResult:
    """Execute the L+1-stage lexicographic cascade.

    Parameters
    ----------
    base_params:
        The LCPParameters describing the OCP structure (levels, slot counts,
        Tikhonov, penalty form, performance-objective weights).
    vehicle_parameters:
        nuPlan VehicleParameters (passed straight to
        :class:`LCPTrajectoryPlanner`).
    affine_steps, x0_local, u_prev, v_cap, Xref_local, rule_pack:
        Same per-tick parameters as :meth:`LCPTrajectoryPlanner.push_parameters`.
        These define the OCP instance the cascade solves.
    delta_lex:
        Numerical slack added to each cascade-stage budget constraint.
    active_threshold:
        Slack value above which a rule constraint is considered "active" in
        the returned ``active_set``.

    Returns
    -------
    A :class:`CascadeResult` carrying the lex achievement point
    ``p_star = (V_1*, ..., V_L*, J*)``, the lex trajectory, the per-level
    slack matrices, and the active-set classification used by Algorithm 1A.
    """
    n_levels = len(base_params.level_specs)

    # Build a fresh planner with stage-1 weights = (1.0, 0.0, ..., 0.0) so
    # the violation objective dominates. We rebuild between stages because
    # CasADi's Opti object can't have its objective changed in-place once
    # the solver is bound; we rebuild with a different cost expression.
    stage_results: List[CascadeStageResult] = []
    V_star_running: List[float] = []

    # Stage 1..L: minimise V_i with V_{i'} <= V_{i'}* + delta for i' < i.
    for i in range(n_levels):
        planner, V_after_stage = _solve_violation_stage(
            base_params=base_params,
            vehicle_parameters=vehicle_parameters,
            stage_level=i,
            V_star_prev=V_star_running,
            delta_lex=delta_lex,
            affine_steps=affine_steps,
            x0_local=x0_local,
            u_prev=u_prev,
            v_cap=v_cap,
            Xref_local=Xref_local,
            rule_pack=rule_pack,
        )
        V_star_running.append(V_after_stage[i])
        X_sol, U_sol, T_sol = planner._last_solution
        stage_results.append(
            CascadeStageResult(
                stage_index=i + 1,
                is_performance_stage=False,
                V_star_stage=V_star_running[-1],
                X_sol=X_sol,
                U_sol=U_sol,
                T_sol=T_sol,
                V_star_all=list(V_star_running),
            )
        )

    # Stage L+1: minimise J with all V_i budgets in place.
    planner, V_final = _solve_performance_stage(
        base_params=base_params,
        vehicle_parameters=vehicle_parameters,
        V_star_all=V_star_running,
        delta_lex=delta_lex,
        affine_steps=affine_steps,
        x0_local=x0_local,
        u_prev=u_prev,
        v_cap=v_cap,
        Xref_local=Xref_local,
        rule_pack=rule_pack,
    )
    X_lex, U_lex, T_lex = planner._last_solution

    # J* is the performance-objective value at the lex point. We recompute it
    # outside the OCP because the OCP only stores the *cost*, which includes
    # the violation-budget multipliers when the cascade is active.
    J_star = _evaluate_performance_objective(base_params, X_lex, U_lex, Xref_local)

    stage_results.append(
        CascadeStageResult(
            stage_index=n_levels + 1,
            is_performance_stage=True,
            V_star_stage=J_star,
            X_sol=X_lex,
            U_sol=U_lex,
            T_sol=T_lex,
            V_star_all=list(V_star_running),
        )
    )

    p_star = np.array(V_star_running + [J_star], dtype=np.float64)

    # Active-set classification at the lex point.
    active_set: List[Tuple[int, int, int]] = []
    for level_idx, T_level in enumerate(T_lex):
        T_arr = np.atleast_2d(np.asarray(T_level))
        if T_arr.shape[0] != base_params.level_specs[level_idx].slots_per_step:
            T_arr = T_arr.T
        rows, cols = T_arr.shape
        for j in range(rows):
            for k in range(cols):
                if T_arr[j, k] > active_threshold:
                    active_set.append((level_idx, j, k))

    return CascadeResult(
        p_star=p_star,
        z_lex_X=X_lex,
        z_lex_U=U_lex,
        T_lex=T_lex,
        stage_results=stage_results,
        active_set=active_set,
    )


def _solve_violation_stage(
    base_params: LCPParameters,
    vehicle_parameters,
    stage_level: int,
    V_star_prev: Sequence[float],
    delta_lex: float,
    affine_steps,
    x0_local,
    u_prev,
    v_cap,
    Xref_local,
    rule_pack,
) -> Tuple[LCPTrajectoryPlanner, List[float]]:
    """Solve the stage that minimises ``V_{stage_level}`` with previous-stage
    budgets in place.

    We construct an LCP planner whose weights are zero everywhere except at
    the current stage (where the weight is 1.0). The previous-stage budgets
    are enforced by additional opti.subject_to constraints injected after the
    standard build.
    """
    # Per-level weights for this stage: 1 on the current level, 0 elsewhere.
    n_levels = len(base_params.level_specs)
    stage_weights = tuple(1.0 if i == stage_level else 0.0 for i in range(n_levels))
    # Zero out the performance-objective weights so the cost is purely V_i.
    stage_params = LCPParameters(
        horizon_s=base_params.horizon_s,
        dt_s=base_params.dt_s,
        desired_speed_mps=base_params.desired_speed_mps,
        penalty_form=base_params.penalty_form,
        level_specs=base_params.level_specs,
        weights_per_level=stage_weights,
        tikhonov_slack=base_params.tikhonov_slack,
        limits=base_params.limits,
        solver_options=base_params.solver_options,
        weight_pos=0.0,
        weight_heading=0.0,
        weight_speed=0.0,
        weight_control=0.0,
        weight_control_rate=0.0,
    )
    planner = LCPTrajectoryPlanner(vehicle_parameters, stage_params)

    # Inject previous-stage budgets as additional inequality constraints on the slack matrices.
    opti = planner.opti()
    for i_prev, V_prev in enumerate(V_star_prev):
        if base_params.penalty_form == "l1":
            opti.subject_to(ca.sum1(ca.sum2(planner._T[i_prev])) <= V_prev + delta_lex)
        elif base_params.penalty_form == "l2":
            opti.subject_to(ca.sumsqr(planner._T[i_prev]) <= V_prev + delta_lex)

    planner.push_parameters(
        affine_steps=affine_steps,
        x0_local=x0_local,
        u_prev=u_prev,
        v_cap=v_cap,
        Xref_local=Xref_local,
        rule_pack=rule_pack,
    )

    try:
        X_sol, U_sol, T_sol = planner.solve_once()
    except RuntimeError as exc:
        logger.warning("Cascade stage %d (violation) failed: %s; using warm-start fallback", stage_level + 1, exc)
        N = planner.horizon_steps
        X_sol = np.zeros((4, N + 1))
        X_sol[:, 0] = x0_local
        U_sol = np.zeros((2, N))
        T_sol = [
            np.zeros((spec.slots_per_step, N)) for spec in base_params.level_specs
        ]
    planner._last_solution = (X_sol, U_sol, T_sol)

    V_after = planner.per_level_violation(T_sol)
    return planner, V_after.tolist()


def _solve_performance_stage(
    base_params: LCPParameters,
    vehicle_parameters,
    V_star_all: Sequence[float],
    delta_lex: float,
    affine_steps,
    x0_local,
    u_prev,
    v_cap,
    Xref_local,
    rule_pack,
) -> Tuple[LCPTrajectoryPlanner, float]:
    """Solve the L+1-th stage: minimise J with every V_i ≤ V_i* + delta budget."""
    n_levels = len(base_params.level_specs)
    perf_params = LCPParameters(
        horizon_s=base_params.horizon_s,
        dt_s=base_params.dt_s,
        desired_speed_mps=base_params.desired_speed_mps,
        penalty_form=base_params.penalty_form,
        level_specs=base_params.level_specs,
        weights_per_level=tuple(0.0 for _ in range(n_levels)),  # turn off level-cost terms
        tikhonov_slack=base_params.tikhonov_slack,
        limits=base_params.limits,
        solver_options=base_params.solver_options,
        weight_pos=base_params.weight_pos,
        weight_heading=base_params.weight_heading,
        weight_speed=base_params.weight_speed,
        weight_control=base_params.weight_control,
        weight_control_rate=base_params.weight_control_rate,
    )
    planner = LCPTrajectoryPlanner(vehicle_parameters, perf_params)

    opti = planner.opti()
    for i_prev, V_prev in enumerate(V_star_all):
        if base_params.penalty_form == "l1":
            opti.subject_to(ca.sum1(ca.sum2(planner._T[i_prev])) <= V_prev + delta_lex)
        elif base_params.penalty_form == "l2":
            opti.subject_to(ca.sumsqr(planner._T[i_prev]) <= V_prev + delta_lex)

    planner.push_parameters(
        affine_steps=affine_steps,
        x0_local=x0_local,
        u_prev=u_prev,
        v_cap=v_cap,
        Xref_local=Xref_local,
        rule_pack=rule_pack,
    )

    try:
        X_sol, U_sol, T_sol = planner.solve_once()
    except RuntimeError as exc:
        logger.warning("Cascade performance stage failed: %s; using warm-start fallback", exc)
        N = planner.horizon_steps
        X_sol = np.zeros((4, N + 1))
        X_sol[:, 0] = x0_local
        U_sol = np.zeros((2, N))
        T_sol = [np.zeros((spec.slots_per_step, N)) for spec in base_params.level_specs]
    planner._last_solution = (X_sol, U_sol, T_sol)
    J_value = _evaluate_performance_objective(base_params, X_sol, U_sol, Xref_local)
    return planner, J_value


def _evaluate_performance_objective(
    params: LCPParameters,
    X_sol: np.ndarray,
    U_sol: np.ndarray,
    Xref: np.ndarray,
) -> float:
    """Evaluate J(z) outside the OCP for reporting.

    Mirrors the cost expression built in :meth:`LCPTrajectoryPlanner._build_problem`'s
    performance section.
    """
    N = U_sol.shape[1]
    J = 0.0
    for k in range(N + 1):
        ex = X_sol[0, k] - Xref[0, k]
        ey = X_sol[1, k] - Xref[1, k]
        ev = X_sol[3, k] - Xref[3, k]
        d_psi = X_sol[2, k] - Xref[2, k]
        J += params.weight_pos * (ex * ex + ey * ey)
        J += params.weight_heading * (1.0 - np.cos(d_psi))
        J += params.weight_speed * ev * ev
    for k in range(N):
        J += params.weight_control * (U_sol[0, k] ** 2 + U_sol[1, k] ** 2)
        if k > 0:
            d_a = U_sol[0, k] - U_sol[0, k - 1]
            d_d = U_sol[1, k] - U_sol[1, k - 1]
            J += params.weight_control_rate * (d_a * d_a + d_d * d_d)
    return float(J)
