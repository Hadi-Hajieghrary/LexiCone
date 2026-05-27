"""Algorithm 1A validation against the paper's worked Example 1.

The paper's Example 1 (Section 11.1) is:

- ``Z = [0, 10]^2``
- Two priority levels:
    - ``C_1`` (high): ``z_1 + z_2 ≤ 8``,
    - ``C_2`` (low):  ``z_1 ≤ 3``.
- Performance: ``J(z) = -2 z_1 - z_2``.
- L₁ penalty.

Cascade outputs (computed analytically in the paper):
- ``z_lex* = (3, 5)``, ``J* = -11``, ``p* = (0, 0, -11)``.
- Both rule constraints are *boundary-binding* at ``z_lex*``.
- Cascade LICQ holds (the two gradients ``∇g_1 = (1, 1)`` and ``∇g_2 = (1, 0)``
  are linearly independent).

Equivalence region: ``Ω(p*) = {(w_1, w_2) : w_1 ≥ 1, w_2 ≥ 1}``.

Chebyshev centre on ``[1, 10]^2``: ``w† = (5.5, 5.5)``, ``r† = 4.5``.

This test reproduces the entire computation through :func:`algorithm_1a` and
:func:`omega_half_space_description` and asserts the paper's numerical values
to 1e-6 precision.
"""

from __future__ import annotations

import numpy as np
import pytest

from lcp.equivalence import (
    ActiveConstraint,
    WeightCalibrationInputs,
    algorithm_1a,
    omega_half_space_description,
)


def _example1_inputs(box=(1.0, 10.0)) -> WeightCalibrationInputs:
    """Build the calibration inputs for the paper's Example 1."""
    grad_J = np.array([-2.0, -1.0])
    grad_g1 = np.array([1.0, 1.0])   # ∇(z_1 + z_2 - 8)
    grad_g2 = np.array([1.0, 0.0])   # ∇(z_1 - 3)
    actives = [
        ActiveConstraint(level_index=0, slot_index=0, gradient=grad_g1, kind="boundary_binding"),
        ActiveConstraint(level_index=1, slot_index=0, gradient=grad_g2, kind="boundary_binding"),
    ]
    return WeightCalibrationInputs(
        grad_J=grad_J,
        active_rule_constraints=actives,
        active_phys_inequalities=[],
        active_equalities=[],
        n_levels=2,
        box_lower=np.array([box[0], box[0]]),
        box_upper=np.array([box[1], box[1]]),
    )


def test_algorithm_1a_recovers_example1_chebyshev_center():
    """w† = (5.5, 5.5), r† = 4.5 on the [1, 10]² box."""
    inputs = _example1_inputs(box=(1.0, 10.0))
    result = algorithm_1a(inputs)
    assert result.lp_status, f"LP must succeed; got {result.lp_status}"
    np.testing.assert_allclose(result.w_dagger, [5.5, 5.5], atol=1e-6)
    assert result.r_dagger == pytest.approx(4.5, abs=1e-6)


def test_algorithm_1a_recovers_example1_beta_multipliers():
    """The lex KKT multipliers should be β_1 = 1, β_2 = 1 at w†."""
    inputs = _example1_inputs(box=(1.0, 10.0))
    result = algorithm_1a(inputs)
    np.testing.assert_allclose(result.beta_at_optimum, [1.0, 1.0], atol=1e-6)


def test_algorithm_1a_chebyshev_shifts_with_box():
    """A wider box (lower=1, upper=100) should push w† toward the box centre,
    not its lower-Ω-facet."""
    inputs = _example1_inputs(box=(1.0, 100.0))
    result = algorithm_1a(inputs)
    # With Ω facets at w_i = 1 and box ceiling at 100, the Chebyshev centre
    # balances distance to (w_i = 1) against (w_i = 100). The active facets
    # are w_i = 1 (since the Ω lower bound is 1 from the bound β_i ≤ w_i with
    # β_i = 1); the upper face is at 100. Centre is at w_i = 50.5 with r†=49.5.
    np.testing.assert_allclose(result.w_dagger, [50.5, 50.5], atol=1e-6)
    assert result.r_dagger == pytest.approx(49.5, abs=1e-6)


def test_algorithm_1a_infeasible_box_falls_back():
    """If the box's upper bound is below the Ω(p*) lower facet, the LP is
    infeasible. The result falls back to the box centre with r†=0."""
    # Box = [0.5, 0.9]² is entirely BELOW Ω(p*) = {w ≥ 1}, so infeasible.
    inputs = _example1_inputs(box=(0.5, 0.9))
    result = algorithm_1a(inputs)
    # Either the LP reports failure, or it returns a fallback weight.
    # Either way, r† should be 0 (no robustness within Ω(p*) ∩ box).
    assert result.r_dagger == pytest.approx(0.0, abs=1e-6)


def test_omega_half_space_description_matches_paper():
    """Explicit ``Cw ≤ d`` must reduce to ``-w_1 ≤ -1`` and ``-w_2 ≤ -1``
    (i.e., the rays ``w_i ≥ 1``)."""
    inputs = _example1_inputs(box=(1.0, 10.0))
    C, d = omega_half_space_description(inputs)
    # We expect 4 rows: β_1 ≥ 0 (trivial), β_1 ≤ w_1, β_2 ≥ 0 (trivial), β_2 ≤ w_2.
    # The non-trivial facets are the β ≤ w ones. Since β_1 = β_2 = 1 are forced
    # by the unique solution to the stationarity system, the β ≥ 0 rows say
    # 1 ≥ 0 (trivially satisfied; encoded as 0 ≤ 1).
    assert C.shape == (4, 2)
    # Find the two non-trivial rows (the ones with a nonzero coefficient on w).
    nontrivial = [i for i in range(4) if not np.allclose(C[i], 0.0)]
    assert len(nontrivial) == 2
    # The two non-trivial rows should encode ``-w_1 ≤ -1`` and ``-w_2 ≤ -1``.
    for idx in nontrivial:
        row = C[idx]
        offset = d[idx]
        # Up to sign / ordering, one row has (negative on w_1, offset=-1) and
        # the other has (negative on w_2, offset=-1).
        nonzero = np.where(np.abs(row) > 1e-9)[0]
        assert len(nonzero) == 1
        i = nonzero[0]
        # Row entry on w_i should be -1, offset should be -1 (i.e., -w_i ≤ -1).
        assert row[i] == pytest.approx(-1.0, abs=1e-9)
        assert offset == pytest.approx(-1.0, abs=1e-9)


def test_algorithm_1a_handles_one_violated_one_bdy():
    """When one rule is *violated* (V_i > 0) and the other is boundary-binding,
    the violated rule contributes w_i ∇V_i directly to the stationarity
    equation (no β multiplier; α = 1)."""
    # Modify Example 1: pretend the high-priority rule is violated rather than
    # boundary-binding. Stationarity: ∇J + w_1 ∇g_1 + β_2 ∇g_2 = 0.
    # ⇒ -2 + w_1 + β_2 = 0, -1 + w_1 = 0  ⇒  w_1 = 1, β_2 = 1.
    # With β_2 ≤ w_2 → w_2 ≥ 1. And w_1 is FIXED to 1 (not a free range).
    grad_J = np.array([-2.0, -1.0])
    grad_g1 = np.array([1.0, 1.0])
    grad_g2 = np.array([1.0, 0.0])
    inputs = WeightCalibrationInputs(
        grad_J=grad_J,
        active_rule_constraints=[
            ActiveConstraint(level_index=0, slot_index=0, gradient=grad_g1, kind="violated"),
            ActiveConstraint(level_index=1, slot_index=0, gradient=grad_g2, kind="boundary_binding"),
        ],
        active_phys_inequalities=[],
        active_equalities=[],
        n_levels=2,
        box_lower=np.array([1.0, 1.0]),
        box_upper=np.array([10.0, 10.0]),
    )
    result = algorithm_1a(inputs)
    # Stationarity uniquely fixes w_1 = 1. r† should reflect that w_1 is at the
    # lower box face — no slack.
    assert result.w_dagger[0] == pytest.approx(1.0, abs=1e-6)
    assert result.r_dagger == pytest.approx(0.0, abs=1e-6)


def test_algorithm_1a_zero_active_set_handles_gracefully():
    """If no rule constraints are active (V_i* = 0 for all i with strict
    interior satisfaction), the lex point is in ``J``'s unconstrained minimum
    on Z. The stationarity system has no rule terms; the LP should return a
    feasible w† anywhere in the box."""
    inputs = WeightCalibrationInputs(
        grad_J=np.zeros(2),
        active_rule_constraints=[],
        active_phys_inequalities=[],
        active_equalities=[],
        n_levels=2,
        box_lower=np.array([1.0, 1.0]),
        box_upper=np.array([10.0, 10.0]),
    )
    result = algorithm_1a(inputs)
    # The Chebyshev centre on the box is (5.5, 5.5), r=4.5.
    np.testing.assert_allclose(result.w_dagger, [5.5, 5.5], atol=1e-6)
    assert result.r_dagger == pytest.approx(4.5, abs=1e-6)
