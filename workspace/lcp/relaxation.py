"""Relaxation Decision Framework (v10_2 Section 10).

This module implements the complementary question to Sections 4-9: *when
should one relax the constraints in the first place?* The framework provides
two independent justifications for relaxation, exactly as specified in §10:

- **(A) Necessity (§10.2):** relaxation is *necessary* when the higher-
  priority levels are jointly infeasible without it. Detected by sequential
  Phase-I LP per Definition 10.1.

- **(B) Significance (§10.3):** relaxation is *justified* under the
  operator-utility model :math:`U_i(\\delta) = J^\\star_i(\\delta) +
  \\pi_i \\delta` when the marginal improvement in J per unit of violation
  exceeds the operator-supplied priority weight :math:`\\pi_i`. Detected
  via the knee condition of Theorem 10.1.

The combined decision rule is the disjunction of (A) and (B), split into
two theorems per §10.4:

- **Theorem 10.2a** (feasibility-required): if :math:`i > i^\\star_{nec}`,
  relax — no operator input required.
- **Theorem 10.2b** (utility-optimal): if :math:`i \\leq i^\\star_{nec}` and
  there exists MRS in :math:`-\\partial J^\\star_i(0)` with
  :math:`\\text{MRS} > \\pi_i`, relax to depth :math:`\\delta^\\star_i`.

The iterative procedure (Procedure 10.1, §10.5) composes these per-level
decisions into a multi-level relaxation policy. It is a coordinate-descent
heuristic on :math:`J^\\star(\\boldsymbol\\delta) + \\sum_i \\pi_i \\delta_i`
and inherits coordinate-descent convergence properties under convexity.

This module is dynamics-agnostic. The caller supplies three solver callbacks:

- ``feasibility_solver(active_levels) -> is_feasible``: Phase-I feasibility
  recovery for §10.2.
- ``solve_with_relaxation(active_levels, deltas) -> (z, J_star, multipliers)``:
  parametric solve for §10.3 (value function and MRS).
- ``current_lex_multipliers() -> dict[level -> lambda_i]``: Lagrange multipliers
  at the current hard-constrained optimum for §10.5 Step 3.

This factoring keeps the framework's mathematical content separated from
the solver backend (scipy / CasADi / CVXPY / custom).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# §10.2 — Necessity (Phase-I feasibility recovery)
# ----------------------------------------------------------------------


@dataclass
class NecessityReport:
    """Outcome of §10.2 sequential Phase-I feasibility recovery.

    ``i_star_nec`` is :math:`i^\\star_{nec}` of Definition 10.1: the largest
    priority depth at which the cascade is feasible with all higher-priority
    levels enforced as hard.

    Conventions per Definition 10.1:

    - ``i_star_nec = 0`` if level 1 (the highest priority) is already infeasible
      on the polyhedral feasibility set :math:`\\mathcal{Z}`.
    - ``i_star_nec = L`` if all :math:`L` levels are jointly feasible (no
      relaxation necessary).
    - ``0 < i_star_nec < L`` if levels :math:`1..i^\\star_{nec}` can be held
      hard but level :math:`i^\\star_{nec} + 1` is the first to require
      relaxation.

    ``forced_relaxation_levels`` lists every level index :math:`i >
    i^\\star_{nec}` that must be softened (1-based to match the paper's
    notation).
    """
    i_star_nec: int
    n_levels: int
    forced_relaxation_levels: List[int]
    feasibility_trace: List[Tuple[int, bool]]
    """Per-depth feasibility result: ``[(0, True), (1, True), (2, False), ...]``
    for each ``i`` checked in sequence."""

    @property
    def any_relaxation_required(self) -> bool:
        return self.i_star_nec < self.n_levels


def compute_necessary_relaxation_level(
    *,
    feasibility_solver: Callable[[int], bool],
    n_levels: int,
) -> NecessityReport:
    """Sequential Phase-I feasibility recovery per v10_2 §10.2.

    Solves :math:`L` Phase-I LPs in priority order. ``feasibility_solver(i)``
    must return ``True`` iff :math:`\\mathcal{Z} \\cap \\bigcap_{i' \\leq i}
    \\mathcal{C}_{i'} \\neq \\emptyset` — i.e., a trajectory exists that
    satisfies all priority levels :math:`1, \\ldots, i` as hard constraints
    plus the polyhedral feasibility set.

    The function walks :math:`i = 1, 2, \\ldots, L` and returns the first
    :math:`i` at which feasibility fails. Per Definition 10.1, if level 1 is
    already infeasible the result is ``i_star_nec = 0``; if all levels
    feasible, ``i_star_nec = L``.

    Definition 10.1 uses 1-based level indexing (i = 1, ..., L); we mirror
    that here. The ``feasibility_solver`` is called with the priority depth
    (1-based) and should return True/False.
    """
    trace: List[Tuple[int, bool]] = []
    # Depth-0 feasibility (only the polyhedral Z): always True by hypothesis (A1).
    trace.append((0, True))
    i_star_nec = n_levels  # optimistic default: all jointly feasible
    for i in range(1, n_levels + 1):
        feas = bool(feasibility_solver(i))
        trace.append((i, feas))
        if not feas:
            i_star_nec = i - 1
            break

    forced = list(range(i_star_nec + 1, n_levels + 1))
    return NecessityReport(
        i_star_nec=i_star_nec,
        n_levels=n_levels,
        forced_relaxation_levels=forced,
        feasibility_trace=trace,
    )


# ----------------------------------------------------------------------
# §10.3 — Significance (value-function knee)
# ----------------------------------------------------------------------


@dataclass
class ValueFunctionPoint:
    """One sample of the value function :math:`J^\\star_i(\\delta)`.

    ``delta`` is the relaxation depth (V_i tolerance), ``J_star`` is
    :math:`\\min_z J(z)` subject to :math:`V_i(z) \\leq \\delta` and other
    levels at their lex status. ``multiplier_lambda_i`` is the Lagrange
    multiplier of the :math:`V_i \\leq \\delta` constraint at this sample —
    equals :math:`-\\partial J^\\star_i / \\partial \\delta` for convex
    perturbation theory.
    """
    delta: float
    J_star: float
    multiplier_lambda_i: float


def evaluate_value_function(
    *,
    solve_with_relaxation: Callable[[float], Tuple[float, float]],
    delta_grid: Sequence[float],
) -> List[ValueFunctionPoint]:
    """Sample the value function :math:`J^\\star_i(\\delta)` at the given
    relaxation depths.

    ``solve_with_relaxation(delta)`` must return ``(J_star, lambda_i)`` where
    ``J_star`` is the parametric optimum and ``lambda_i`` is the Lagrange
    multiplier of the :math:`V_i \\leq \\delta` constraint (i.e. the negative
    subgradient of the value function with respect to delta at this point).

    The sampled points satisfy the v10_2 §10.3 properties: :math:`J^\\star_i`
    is convex non-increasing in :math:`\\delta`; :math:`\\lambda_i =
    -\\partial J^\\star_i(\\delta) \\geq 0`.
    """
    points: List[ValueFunctionPoint] = []
    for d in sorted(delta_grid):
        J_star, lam = solve_with_relaxation(float(d))
        points.append(
            ValueFunctionPoint(
                delta=float(d),
                J_star=float(J_star),
                multiplier_lambda_i=float(lam),
            )
        )
    return points


def mrs_at_zero(
    *,
    solve_with_relaxation: Callable[[float], Tuple[float, float]],
    epsilon_probe: float = 1e-6,
) -> float:
    """Compute :math:`\\text{MRS} \\in -\\partial J^\\star_i(0)` per Theorem
    10.2b condition (30).

    Returns the magnitude of the value-function slope at :math:`\\delta = 0`,
    equivalently the Lagrange multiplier of the :math:`V_i \\leq 0`
    constraint at the hard-constrained optimum. A small positive ``epsilon_probe``
    is used to evaluate :math:`-\\partial J^\\star_i(0^+)` when the value
    function is non-differentiable at zero.

    This is the central quantity in the significance test (B): relax iff
    ``mrs_at_zero > pi_i``.
    """
    _, lam_zero = solve_with_relaxation(0.0)
    if lam_zero > 0:
        return float(lam_zero)
    # Fallback: numerical right-derivative.
    J0, _ = solve_with_relaxation(0.0)
    J_eps, _ = solve_with_relaxation(float(epsilon_probe))
    return float(max(0.0, -(J_eps - J0) / epsilon_probe))


def find_knee_depth(
    *,
    solve_with_relaxation: Callable[[float], Tuple[float, float]],
    pi_i: float,
    delta_max: float,
    tol: float = 1e-6,
    max_iter: int = 60,
) -> float:
    """Find :math:`\\delta^\\star_i` satisfying :math:`\\pi_i \\in
    -\\partial J^\\star_i(\\delta^\\star_i)` per Theorem 10.1 condition (29).

    Solved via bisection on the monotone non-increasing
    :math:`\\lambda(\\delta) := -\\partial J^\\star_i(\\delta)` — the
    Lagrange multiplier of the :math:`V_i \\leq \\delta` constraint.

    - If :math:`\\lambda(0) \\leq \\pi_i`: returns :math:`\\delta^\\star = 0`
      (no relaxation justified).
    - If :math:`\\lambda(\\delta_{\\max}) > \\pi_i`: returns
      :math:`\\delta_{\\max}` (relaxation justified beyond the operator's
      search range; the operator should widen the range).
    - Otherwise: binary search until :math:`|\\lambda(\\delta) - \\pi_i|
      \\leq tol`.

    Returns the optimal relaxation depth :math:`\\delta^\\star_i \\geq 0`.
    """
    if pi_i <= 0:
        raise ValueError(f"pi_i must be positive; got {pi_i}")
    if delta_max <= 0:
        raise ValueError(f"delta_max must be positive; got {delta_max}")
    # Endpoint checks.
    _, lam_lo = solve_with_relaxation(0.0)
    if lam_lo <= pi_i:
        return 0.0
    _, lam_hi = solve_with_relaxation(delta_max)
    if lam_hi > pi_i:
        logger.warning(
            "knee depth >= delta_max=%g; pi_i=%g may be too low for search range",
            delta_max, pi_i,
        )
        return float(delta_max)
    # Bisection on lambda(delta).
    lo, hi = 0.0, delta_max
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        _, lam_mid = solve_with_relaxation(mid)
        if abs(lam_mid - pi_i) <= tol:
            return float(mid)
        if lam_mid > pi_i:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))


# ----------------------------------------------------------------------
# §10.4 — Combined Decision Rule
# ----------------------------------------------------------------------


@dataclass
class LevelRelaxationDecision:
    """Per-level relaxation decision (Theorems 10.2a and 10.2b)."""
    level: int                              # 1-based, matches the paper
    decision: Literal["hard", "necessity-relaxed", "utility-relaxed"]
    delta_star: float                        # optimal relaxation depth; 0 if hard
    mrs_at_zero: Optional[float] = None      # populated when 10.2b was evaluated
    rationale: str = ""

    @property
    def is_relaxed(self) -> bool:
        return self.decision in ("necessity-relaxed", "utility-relaxed")


def decide_level_relaxation(
    *,
    level: int,
    is_above_necessity: bool,
    solve_with_relaxation: Optional[Callable[[float], Tuple[float, float]]] = None,
    pi_i: Optional[float] = None,
    delta_max: float = 1.0,
    feasibility_required_delta: Optional[float] = None,
) -> LevelRelaxationDecision:
    """Per-level decision per Theorems 10.2a and 10.2b.

    Parameters
    ----------
    level
        1-based priority level index.
    is_above_necessity
        True iff :math:`i > i^\\star_{nec}` — this level is feasibility-forced
        (Theorem 10.2a). When True, the decision is "necessity-relaxed" and
        ``feasibility_required_delta`` must be supplied.
    solve_with_relaxation
        Parametric solver callback; required when ``is_above_necessity = False``
        for the utility-optimality check (Theorem 10.2b).
    pi_i
        Operator-supplied priority weight; required when
        ``is_above_necessity = False``.
    delta_max
        Upper bound for the knee-depth search.
    feasibility_required_delta
        :math:`\\delta_i^{feas}` from Theorem 10.2a — the minimum relaxation
        required for cascade feasibility at this level. Required when
        ``is_above_necessity = True``.
    """
    if is_above_necessity:
        if feasibility_required_delta is None:
            raise ValueError(
                f"feasibility_required_delta required for necessity-forced level {level}"
            )
        return LevelRelaxationDecision(
            level=level,
            decision="necessity-relaxed",
            delta_star=float(feasibility_required_delta),
            mrs_at_zero=None,
            rationale=(
                f"Theorem 10.2a: level {level} > i*_nec; cascade infeasible at delta=0; "
                f"minimum feasibility-required delta = {feasibility_required_delta:.6g}"
            ),
        )
    if solve_with_relaxation is None or pi_i is None:
        raise ValueError(
            f"solve_with_relaxation + pi_i required for utility-optimality check (level {level})"
        )
    mrs = mrs_at_zero(solve_with_relaxation=solve_with_relaxation)
    if mrs <= pi_i:
        return LevelRelaxationDecision(
            level=level,
            decision="hard",
            delta_star=0.0,
            mrs_at_zero=mrs,
            rationale=(
                f"Theorem 10.2b: MRS(0)={mrs:.6g} <= pi_i={pi_i:.6g}; "
                f"hard enforcement is utility-optimal."
            ),
        )
    delta_star = find_knee_depth(
        solve_with_relaxation=solve_with_relaxation,
        pi_i=pi_i,
        delta_max=delta_max,
    )
    return LevelRelaxationDecision(
        level=level,
        decision="utility-relaxed",
        delta_star=delta_star,
        mrs_at_zero=mrs,
        rationale=(
            f"Theorem 10.2b: MRS(0)={mrs:.6g} > pi_i={pi_i:.6g}; "
            f"utility-optimal relaxation depth delta*={delta_star:.6g} "
            f"via Theorem 10.1 knee condition."
        ),
    )


# ----------------------------------------------------------------------
# §10.5 — Iterative Procedure
# ----------------------------------------------------------------------


@dataclass
class IterativeRelaxationStep:
    """One iteration of Procedure 10.1."""
    iteration: int
    softened_level: Optional[int]
    softened_delta: Optional[float]
    softened_ratio: Optional[float]
    rationale: str
    deltas_after: Dict[int, float]


@dataclass
class IterativeRelaxationResult:
    """Final outcome of Procedure 10.1."""
    final_deltas: Dict[int, float]
    """Map from 1-based level index to chosen relaxation depth :math:`\\delta_i`."""
    steps: List[IterativeRelaxationStep]
    necessity_report: NecessityReport
    converged: bool
    """True iff the procedure terminated by passing the significance test on
    every still-hard level (Step 4 found no level with :math:`\\lambda_i >
    \\pi_i`). False if ``max_iterations`` was hit first."""


def iterative_lex_relaxation(
    *,
    n_levels: int,
    pi_weights: Dict[int, float],
    feasibility_solver: Callable[[int], bool],
    current_lex_multipliers: Callable[[Dict[int, float]], Dict[int, float]],
    solve_with_relaxation_per_level: Callable[[int], Callable[[float], Tuple[float, float]]],
    feasibility_required_delta_per_level: Callable[[int], float],
    delta_max: float = 1.0,
    max_iterations: int = 20,
) -> IterativeRelaxationResult:
    """Procedure 10.1: iterative lex relaxation per v10_2 §10.5.

    The procedure composes Theorems 10.2a (necessity) and 10.2b (significance)
    into a multi-level relaxation policy.

    Parameters
    ----------
    n_levels
        Number of priority levels L.
    pi_weights
        Operator-supplied priority weight per level (1-based dict).
    feasibility_solver
        ``feasibility_solver(i) -> bool``: Phase-I feasibility at depth i.
        Used in Step 2 to identify :math:`i^\\star_{nec}`.
    current_lex_multipliers
        ``current_lex_multipliers(current_deltas) -> dict[i -> lambda_i]``:
        Lagrange multipliers at the current optimum, used in Step 3 to test
        the significance condition (B).
    solve_with_relaxation_per_level
        ``solve_with_relaxation_per_level(i) -> (callable(delta) -> (J*, lambda))``:
        for each level i, returns the parametric solver that fixes other
        levels at their current depths and varies level i.
    feasibility_required_delta_per_level
        ``feasibility_required_delta_per_level(i) -> delta_feas``: minimum
        feasibility-required relaxation per Theorem 10.2a.
    delta_max
        Upper bound for the knee-depth search per level.
    max_iterations
        Soft cap on the outer loop iterations.

    Returns
    -------
    IterativeRelaxationResult
        Final per-level relaxation depths + iteration trace.
    """
    # Step 1 + 2: identify necessity-forced relaxations and seed deltas.
    necessity = compute_necessary_relaxation_level(
        feasibility_solver=feasibility_solver, n_levels=n_levels
    )
    deltas: Dict[int, float] = {i: 0.0 for i in range(1, n_levels + 1)}
    steps: List[IterativeRelaxationStep] = []
    for i in necessity.forced_relaxation_levels:
        delta_feas = float(feasibility_required_delta_per_level(i))
        deltas[i] = delta_feas
        steps.append(
            IterativeRelaxationStep(
                iteration=0,
                softened_level=i,
                softened_delta=delta_feas,
                softened_ratio=None,
                rationale=(
                    f"Theorem 10.2a: level {i} > i*_nec={necessity.i_star_nec}; "
                    f"forced delta = {delta_feas:.6g}"
                ),
                deltas_after={**deltas},
            )
        )

    # Steps 3-5: iterative significance test on still-hard levels.
    still_hard_levels = sorted(
        i for i in range(1, n_levels + 1)
        if i not in necessity.forced_relaxation_levels
    )
    converged = False
    for it in range(1, max_iterations + 1):
        lam_dict = current_lex_multipliers({**deltas})
        # Identify the still-hard level with maximum lambda_i / pi_i > 1.
        violators: List[Tuple[int, float]] = []
        for i in still_hard_levels:
            pi_i = pi_weights.get(i)
            if pi_i is None or pi_i <= 0:
                continue
            lam_i = float(lam_dict.get(i, 0.0))
            if lam_i > pi_i:
                violators.append((i, lam_i / pi_i))
        if not violators:
            converged = True
            break
        # Soften the level with maximum ratio.
        violators.sort(key=lambda kv: kv[1], reverse=True)
        i_pick, ratio = violators[0]
        pi_i = pi_weights[i_pick]
        # Compute the optimal depth via the knee condition.
        per_level_solver = solve_with_relaxation_per_level(i_pick)
        delta_star = find_knee_depth(
            solve_with_relaxation=per_level_solver,
            pi_i=pi_i, delta_max=delta_max,
        )
        deltas[i_pick] = delta_star
        still_hard_levels.remove(i_pick)
        steps.append(
            IterativeRelaxationStep(
                iteration=it,
                softened_level=i_pick,
                softened_delta=delta_star,
                softened_ratio=ratio,
                rationale=(
                    f"Theorem 10.2b: level {i_pick} has lambda/pi = {ratio:.3g} > 1; "
                    f"softened to delta*={delta_star:.6g} via knee condition."
                ),
                deltas_after={**deltas},
            )
        )

    return IterativeRelaxationResult(
        final_deltas={**deltas},
        steps=steps,
        necessity_report=necessity,
        converged=converged,
    )
