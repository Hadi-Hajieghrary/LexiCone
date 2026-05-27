"""Tests for lcp.relaxation (v10_2 Section 10 framework).

Constructed 1D toy problems with closed-form value functions exercise each
sub-component of the Relaxation Decision Framework:

- Toy 1: feasible at all levels → necessity returns ``i_star_nec = L``.
- Toy 2: top-level alone infeasible → ``i_star_nec = 0``.
- Toy 3: feasible up to level 2 of 3 → ``i_star_nec = 2``.
- Toy 4 (value-function): piecewise-affine J*_2(delta) exposes the knee
  condition; pi_2 chosen to land on each branch of the piecewise function.
- Toy 5 (procedure 10.1): combines necessity and significance over 2 levels.
"""
from __future__ import annotations

import numpy as np
import pytest

from lcp.relaxation import (
    compute_necessary_relaxation_level,
    decide_level_relaxation,
    evaluate_value_function,
    find_knee_depth,
    iterative_lex_relaxation,
    mrs_at_zero,
)


# ----------------------------------------------------------------------
# §10.2 — Necessity tests
# ----------------------------------------------------------------------


def test_necessity_all_feasible():
    """All levels jointly feasible ⇒ i*_nec = L."""
    report = compute_necessary_relaxation_level(
        feasibility_solver=lambda i: True, n_levels=3,
    )
    assert report.i_star_nec == 3
    assert report.forced_relaxation_levels == []
    assert not report.any_relaxation_required


def test_necessity_top_level_infeasible():
    """Level 1 alone infeasible ⇒ i*_nec = 0."""
    report = compute_necessary_relaxation_level(
        feasibility_solver=lambda i: False, n_levels=3,
    )
    assert report.i_star_nec == 0
    assert report.forced_relaxation_levels == [1, 2, 3]
    assert report.any_relaxation_required


def test_necessity_breaks_at_level_3():
    """Feasible through level 2; level 3 infeasible ⇒ i*_nec = 2."""
    report = compute_necessary_relaxation_level(
        feasibility_solver=lambda i: i <= 2, n_levels=3,
    )
    assert report.i_star_nec == 2
    assert report.forced_relaxation_levels == [3]
    assert report.any_relaxation_required


# ----------------------------------------------------------------------
# §10.3 — Value-function and knee-condition tests
# ----------------------------------------------------------------------


def _piecewise_solver(delta_break: float, slope_below: float = 1.0):
    """Return a parametric solver matching J*(delta) = max(c - slope*delta, c_floor).

    - For ``delta < delta_break``: J* decreases linearly with slope ``slope_below``,
      lambda = slope_below.
    - For ``delta >= delta_break``: J* is constant (lower bound hit), lambda = 0.

    Concretely: J*(delta) = c - slope*min(delta, delta_break), with c = 8.
    """
    c = 8.0

    def solve(delta: float):
        if delta < delta_break:
            return (c - slope_below * delta, slope_below)
        return (c - slope_below * delta_break, 0.0)

    return solve


def test_mrs_at_zero_returns_slope():
    """MRS(0) equals the right-derivative magnitude of J*."""
    solver = _piecewise_solver(delta_break=3.0, slope_below=1.5)
    mrs = mrs_at_zero(solve_with_relaxation=solver)
    assert mrs == pytest.approx(1.5)


def test_knee_depth_no_relaxation_when_pi_above_slope():
    """pi > slope ⇒ knee at delta = 0 (hard enforcement utility-optimal)."""
    solver = _piecewise_solver(delta_break=3.0, slope_below=1.0)
    delta_star = find_knee_depth(
        solve_with_relaxation=solver, pi_i=2.0, delta_max=5.0,
    )
    assert delta_star == 0.0


def test_knee_depth_at_break_when_pi_below_slope():
    """pi < slope ⇒ knee at the elbow where slope drops to zero."""
    solver = _piecewise_solver(delta_break=3.0, slope_below=1.0)
    delta_star = find_knee_depth(
        solve_with_relaxation=solver, pi_i=0.5, delta_max=5.0,
        tol=1e-4,
    )
    # The piecewise function jumps from lambda=1 to lambda=0 at delta=3,
    # bisection should converge to the break point.
    assert abs(delta_star - 3.0) < 1e-2


def test_value_function_sampling_monotone():
    """J*(delta) is non-increasing in delta."""
    solver = _piecewise_solver(delta_break=2.0, slope_below=1.0)
    points = evaluate_value_function(
        solve_with_relaxation=solver,
        delta_grid=[0.0, 0.5, 1.0, 2.0, 3.0],
    )
    for a, b in zip(points[:-1], points[1:]):
        assert a.J_star >= b.J_star - 1e-12


# ----------------------------------------------------------------------
# §10.4 — Per-level decision tests
# ----------------------------------------------------------------------


def test_decision_above_necessity_returns_necessity_relaxed():
    """Level above i*_nec ⇒ necessity-relaxed at the feasibility-required depth."""
    decision = decide_level_relaxation(
        level=3,
        is_above_necessity=True,
        feasibility_required_delta=0.7,
    )
    assert decision.decision == "necessity-relaxed"
    assert decision.delta_star == pytest.approx(0.7)
    assert decision.is_relaxed


def test_decision_hard_when_mrs_below_pi():
    """Level at/below i*_nec, MRS <= pi ⇒ hard."""
    solver = _piecewise_solver(delta_break=2.0, slope_below=1.0)
    decision = decide_level_relaxation(
        level=1, is_above_necessity=False,
        solve_with_relaxation=solver, pi_i=2.0, delta_max=5.0,
    )
    assert decision.decision == "hard"
    assert decision.delta_star == 0.0
    assert decision.mrs_at_zero == pytest.approx(1.0)
    assert not decision.is_relaxed


def test_decision_utility_relaxed_when_mrs_above_pi():
    """Level at/below i*_nec, MRS > pi ⇒ utility-relaxed at knee."""
    solver = _piecewise_solver(delta_break=3.0, slope_below=1.0)
    decision = decide_level_relaxation(
        level=2, is_above_necessity=False,
        solve_with_relaxation=solver, pi_i=0.5, delta_max=5.0,
    )
    assert decision.decision == "utility-relaxed"
    assert decision.delta_star > 0.0
    assert decision.is_relaxed
    assert decision.mrs_at_zero == pytest.approx(1.0)


def test_decision_requires_feasibility_delta_for_above_necessity():
    """Theorem 10.2a needs an explicit feasibility-required delta."""
    with pytest.raises(ValueError, match="feasibility_required_delta"):
        decide_level_relaxation(level=2, is_above_necessity=True)


# ----------------------------------------------------------------------
# §10.5 — Procedure 10.1 tests
# ----------------------------------------------------------------------


def test_procedure_no_relaxation_converges_immediately():
    """All-feasible, low multipliers ⇒ procedure converges at iter 1 with zeros."""
    result = iterative_lex_relaxation(
        n_levels=2,
        pi_weights={1: 10.0, 2: 10.0},
        feasibility_solver=lambda i: True,
        current_lex_multipliers=lambda d: {1: 0.5, 2: 0.5},
        solve_with_relaxation_per_level=lambda i: (
            _piecewise_solver(delta_break=1.0, slope_below=0.5)
        ),
        feasibility_required_delta_per_level=lambda i: 0.0,
    )
    assert result.converged
    assert result.final_deltas == {1: 0.0, 2: 0.0}
    assert result.necessity_report.i_star_nec == 2


def test_procedure_necessity_seeds_then_converges():
    """Level 2 forced by necessity; level 1 stays hard."""
    result = iterative_lex_relaxation(
        n_levels=2,
        pi_weights={1: 10.0, 2: 10.0},
        feasibility_solver=lambda i: i <= 1,
        current_lex_multipliers=lambda d: {1: 0.5},  # only level 1 still hard
        solve_with_relaxation_per_level=lambda i: (
            _piecewise_solver(delta_break=1.0, slope_below=0.5)
        ),
        feasibility_required_delta_per_level=lambda i: 0.4,
    )
    assert result.necessity_report.i_star_nec == 1
    assert result.final_deltas[2] == pytest.approx(0.4)
    assert result.final_deltas[1] == 0.0
    assert result.converged


def test_procedure_significance_relaxes_high_multiplier_level():
    """Level 1 has lambda > pi ⇒ softened via Theorem 10.2b on first iteration."""
    # Multiplier drops to 0 after relaxation so the procedure converges.
    call_count = {"n": 0}

    def multipliers(d):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return {1: 5.0}  # > pi_1 = 1.0
        return {1: 0.0}

    result = iterative_lex_relaxation(
        n_levels=1,
        pi_weights={1: 1.0},
        feasibility_solver=lambda i: True,
        current_lex_multipliers=multipliers,
        solve_with_relaxation_per_level=lambda i: (
            _piecewise_solver(delta_break=2.0, slope_below=2.5)
        ),
        feasibility_required_delta_per_level=lambda i: 0.0,
        delta_max=5.0,
    )
    assert result.converged
    assert result.final_deltas[1] > 0.0
    # Knee where slope (2.5) drops to 0 is at delta=2 ⇒ bisection lands near 2.
    assert abs(result.final_deltas[1] - 2.0) < 0.05
