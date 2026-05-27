"""Algorithm 1B validation against the paper's worked Example 2.

The paper's Example 2 (Section 11.2) is a three-priority L₂ convex quadratic
program:

- Discrete dynamics ``x_{k+1} = x_k + u_k`` for ``k = 0..3``, ``x_0 = 0``,
  ``u_k ∈ [0, 10]``.
- ``J(z) = (x_4 - 10)²``.
- Three priority levels:
    - ``C_1`` (safety): ``x_k ≤ 3`` for ``k = 1..4``,
    - ``C_2`` (legal):  ``u_k ≤ 2``  for ``k = 0..3``,
    - ``C_3`` (comfort): ``u_k ≥ 1.5`` for ``k = 0..3``.
- L₂ penalty.

Lex outputs (analytic in the paper):

- ``z_lex* = (u_0, u_1, u_2, u_3) = (0.75, 0.75, 0.75, 0.75)``.
- ``V_1* = 0`` (safety boundary-binding at ``x_4 = 3``).
- ``V_2* = 0`` (legal strictly satisfied, ``u_k = 0.75 < 2``).
- ``V_3* = 2.25`` (comfort violated; ``u_k = 0.75 < 1.5``).
- ``J* = 49``.

Active set at z_lex*:
- Level 1 (safety): the *terminal* x_4 ≤ 3 constraint is boundary-binding.
  ``∇g_1 = (1, 1, 1, 1)`` (partials w.r.t. each u_k).
- Level 2 (legal): no actives (all strictly satisfied).
- Level 3 (comfort): violated. ``∇V_3 = -1.5 · (1, 1, 1, 1)`` (the L₂ gradient
  at the four kink points of (1.5 - u_k)_+²).

Performance gradient: ``∇J = ∂(x_4 - 10)²/∂u_k = 2(x_4 - 10) · 1 = 2(3 - 10) = -14``
in every direction, so ``∇J = (-14, -14, -14, -14)``.

Singular-perturbation expansion (Section 11.2 of the paper, my derivation):

- ``c_1 = |⟨∇J, ∇g_1⟩| / (2 ||∇g_1||²) = 56 / 8 = 7``.
- ``κ_1[3] = -⟨∇V_3, ∇g_1⟩ / (2 ||∇g_1||²) = -(-6) / 8 = 0.75``.
- ``κ_1[0] = κ_1[1] = κ_1[2] = 0`` (no other violated levels).

Threshold function (V-form, since paper uses ε_1 as bound on V_1):

.. math::

    W_1(\\epsilon_1, w_3) = \\frac{7 + 0.75 w_3}{\\sqrt{\\epsilon_1}}.

At ``ε_1 = 0.01``, ``w_3 = 1``: ``W_1 ≈ 7.75 / 0.1 = 77.5``. ✓ matches paper.

LP outcome on box ``[10⁻², 10⁴]³`` (paper's primary pointwise LP):
- Feasible at e.g. ``(w_1, w_2, w_3) = (200, 100, 10)`` with threshold slack
  ``0.1·200 − 0.75·10 − 7 = 5.5 > 0``.

This file verifies our :func:`compute_l2_sensitivity_constants`,
:func:`l2_threshold`, and :func:`algorithm_1b` reproduce these signatures
numerically to within 1e-6.
"""

from __future__ import annotations

import numpy as np
import pytest

from lcp.equivalence import (
    L2SensitivityConstants,
    L2SensitivityInputs,
    WeightCalibrationResult,
    algorithm_1b,
    compute_l2_sensitivity_constants,
    l2_threshold,
)


def _example2_inputs(
    epsilon=(0.01, 1.0, 1.0),
    box=(1e-2, 1e4),
    tolerance_form: str = "squared",
) -> L2SensitivityInputs:
    """Build the L₂ calibration inputs for the paper's Example 2."""
    grad_J = np.array([-14.0, -14.0, -14.0, -14.0])
    grad_g1_terminal_safety = np.array([1.0, 1.0, 1.0, 1.0])
    grad_V3_comfort = -1.5 * np.array([1.0, 1.0, 1.0, 1.0])
    return L2SensitivityInputs(
        grad_J=grad_J,
        boundary_binding_per_level={0: [(0, grad_g1_terminal_safety)]},
        violated_grad_V_per_level={2: grad_V3_comfort},
        n_levels=3,
        box_lower=np.array([box[0], box[0], box[0]]),
        box_upper=np.array([box[1], box[1], box[1]]),
        epsilon_per_level=np.array(epsilon),
        tolerance_form=tolerance_form,
    )


def test_l2_sensitivity_constants_example2():
    """``c_1 = 7``, ``κ_1[3] = 0.75``, ``κ_1[2] = 0``, ``κ_1[1] = 0``."""
    inputs = _example2_inputs()
    constants = compute_l2_sensitivity_constants(inputs)
    assert constants.levels_with_threshold == [0]
    assert constants.c_const_per_level[0] == pytest.approx(7.0, abs=1e-9)
    kappa = constants.kappa_per_level[0]
    assert kappa.shape == (3,)
    assert kappa[2] == pytest.approx(0.75, abs=1e-9)
    assert kappa[0] == pytest.approx(0.0, abs=1e-9)
    assert kappa[1] == pytest.approx(0.0, abs=1e-9)


def test_l2_threshold_function_matches_paper_example2():
    """``W_1(0.01, w_3=1) ≈ 77.5`` under V-form (ρ = √ε)."""
    inputs = _example2_inputs(epsilon=(0.01, 1.0, 1.0), tolerance_form="squared")
    constants = compute_l2_sensitivity_constants(inputs)
    # w_-1 has dimensions L-1 = 2 (levels 2 and 3, i.e., legal and comfort).
    w_minus_i = (1.0, 1.0)  # w_legal = 1, w_comfort = 1
    threshold = l2_threshold(
        level_index=0,
        w_minus_i=w_minus_i,
        constants=constants,
        epsilon=0.01,
        tolerance_form="squared",
    )
    # (7 + 0.75 * 1) / sqrt(0.01) = 7.75 / 0.1 = 77.5
    assert threshold == pytest.approx(77.5, abs=1e-6)


def test_l2_threshold_function_raw_form():
    """Raw form ρ(ε) = ε gives an inverse-linear (rather than inverse-sqrt)
    scaling: a 10× tighter tolerance demands a 10× larger weight."""
    inputs = _example2_inputs(tolerance_form="raw")
    constants = compute_l2_sensitivity_constants(inputs)
    threshold_eps_01 = l2_threshold(0, (1.0, 1.0), constants, 0.1, tolerance_form="raw")
    threshold_eps_001 = l2_threshold(0, (1.0, 1.0), constants, 0.01, tolerance_form="raw")
    # 10× tighter ε ⇒ 10× larger threshold under raw form.
    assert threshold_eps_001 / threshold_eps_01 == pytest.approx(10.0, rel=1e-9)


def test_algorithm_1b_example2_pointwise_lp_feasible_in_large_box():
    """The paper's primary LP on the ``[10⁻², 10⁴]³`` box should be feasible
    and produce a robust interior weight."""
    inputs = _example2_inputs(
        epsilon=(0.01, 1.0, 1.0),
        box=(1e-2, 1e4),
        tolerance_form="squared",
    )
    result = algorithm_1b(inputs)
    assert result.r_dagger > 0, f"LP must be feasible; got r†={result.r_dagger}"
    # Threshold at the chosen w_3 must be satisfied: rho(0.01)*w_1 = 0.1*w_1 ≥ 7 + 0.75*w_3.
    w = result.w_dagger
    threshold_slack = 0.1 * w[0] - 0.75 * w[2] - 7.0
    assert threshold_slack >= -1e-6


def test_algorithm_1b_example2_paper_witness_satisfies_threshold():
    """The paper's exhibited witness (200, 100, 10) satisfies the coupled
    threshold with slack 5.5."""
    inputs = _example2_inputs(
        epsilon=(0.01, 1.0, 1.0),
        box=(1e-2, 1e4),
        tolerance_form="squared",
    )
    constants = compute_l2_sensitivity_constants(inputs)
    # At w_3 = 10: W_1(0.01, w_3=10) = (7 + 0.75*10) / 0.1 = 14.5/0.1 = 145.
    threshold = l2_threshold(
        level_index=0,
        w_minus_i=(100.0, 10.0),  # w_legal=100, w_comfort=10
        constants=constants,
        epsilon=0.01,
        tolerance_form="squared",
    )
    assert threshold == pytest.approx(145.0, abs=1e-6)
    # Witness w_1 = 200 satisfies w_1 ≥ 145, so the slack at this witness is 55.
    # The paper computes slack as 0.1·200 − 0.75·10 − 7 = 5.5 in the un-divided form;
    # equivalent up to the ρ(ε) = 0.1 scale factor. Our LP uses the un-divided form
    # directly (rho * w_i - <kappa, w_{-i}> - c ≥ r), so slack = 5.5 matches the paper.
    slack_undivided = 0.1 * 200.0 - 0.75 * 10.0 - 7.0
    assert slack_undivided == pytest.approx(5.5, abs=1e-9)


def test_algorithm_1b_example2_tight_tolerance_needs_larger_w_1():
    """Tightening ε_1 by 100× should push w_1† upward proportional to
    sqrt(100) = 10× (V-form) or 100× (raw form)."""
    inputs_loose = _example2_inputs(epsilon=(0.01, 1.0, 1.0), box=(1e-2, 1e4))
    inputs_tight = _example2_inputs(epsilon=(0.0001, 1.0, 1.0), box=(1e-2, 1e4))
    r_loose = algorithm_1b(inputs_loose)
    r_tight = algorithm_1b(inputs_tight)
    # The Chebyshev centre under the tighter tolerance must lie at higher w_1.
    assert r_tight.w_dagger[0] > r_loose.w_dagger[0]


def test_algorithm_1b_no_boundary_binding_means_box_centre():
    """If no level has a boundary-binding constraint, no threshold inequality
    enters the LP and the result is the geometric box centre with full
    Chebyshev margin."""
    inputs = L2SensitivityInputs(
        grad_J=np.zeros(2),
        boundary_binding_per_level={},
        violated_grad_V_per_level={},
        n_levels=2,
        box_lower=np.array([1.0, 1.0]),
        box_upper=np.array([10.0, 10.0]),
        epsilon_per_level=np.array([0.01, 0.01]),
        tolerance_form="squared",
    )
    result = algorithm_1b(inputs)
    np.testing.assert_allclose(result.w_dagger, [5.5, 5.5], atol=1e-6)
    assert result.r_dagger == pytest.approx(4.5, abs=1e-6)


def test_algorithm_1b_infeasible_box_falls_back():
    """An impossibly small upper bound on w_1 (below the threshold) leaves
    the LP infeasible. We fall back to the box centre with r† = 0."""
    # Box's w_1 ceiling at 50 is below the W_1 threshold (≈77.5 at w_3 = 1
    # boundary), so the coupled-linear constraint cannot be satisfied.
    inputs = _example2_inputs(
        epsilon=(0.01, 1.0, 1.0),
        box=(1e-2, 50.0),
        tolerance_form="squared",
    )
    result = algorithm_1b(inputs)
    assert result.r_dagger == pytest.approx(0.0, abs=1e-6)
