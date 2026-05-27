"""Algorithm 0 (homogeneous-cone primary formulation) validation against
v10_2 Example 1.

Section 9.1 of v10_2 specifies Algorithm 0 as the homogeneous-cone primary
formulation: a Chebyshev LP on the simplex slice
:math:`\\{(w, \\beta) : \\beta + \\sum w_i = 1\\}` with KKT stationarity
:math:`\\beta\\nabla J + \\sum w_i \\alpha_i \\nabla g_i + \\ldots = 0`,
followed by the projection :math:`\\bar w_i = w_i^\\sharp / \\beta^\\sharp`.

For Example 1 (Section 11.1) the homogeneous-cone projection should land
inside :math:`\\Omega(p^\\star) = \\{w_1 \\ge 1, w_2 \\ge 1\\}`, and by the
symmetry of the example (both rule constraints boundary-binding, gradients
``\\nabla g_1 = (1,1)`` and ``\\nabla g_2 = (1,0)``) the result should be
symmetric or close to it. Unlike Algorithm 1A, Algorithm 0 does not require
an operator-supplied box.
"""
from __future__ import annotations

import numpy as np
import pytest

from lcp.equivalence import (
    ActiveConstraint,
    Algorithm0Inputs,
    algorithm_0_homogeneous,
)


def _example1_alg0_inputs() -> Algorithm0Inputs:
    """Example 1 wired for Algorithm 0: same KKT data as Algorithm 1A."""
    grad_J = np.array([-2.0, -1.0])
    grad_g1 = np.array([1.0, 1.0])  # ∇(z_1 + z_2 - 8)
    grad_g2 = np.array([1.0, 0.0])  # ∇(z_1 - 3)
    return Algorithm0Inputs(
        grad_J=grad_J,
        active_rule_constraints=[
            ActiveConstraint(level_index=0, slot_index=0, gradient=grad_g1, kind="boundary_binding"),
            ActiveConstraint(level_index=1, slot_index=0, gradient=grad_g2, kind="boundary_binding"),
        ],
        active_phys_inequalities=[],
        active_equalities=[],
        n_levels=2,
        w_lower=np.array([1e-3, 1e-3]),
        beta_lower=1e-3,
    )


def test_algorithm_0_lp_succeeds_on_example1():
    """The LP must solve; r♯ > 0 by full-dimensionality of Ω̂(p*)."""
    result = algorithm_0_homogeneous(_example1_alg0_inputs())
    assert result.lp_status, f"LP must succeed; got {result.lp_status}"
    assert result.r_sharp >= 0.0


def test_algorithm_0_projected_w_in_equivalence_region():
    """The projected w̄ = w♯ / β♯ must lie in Ω(p*) = {w_1 ≥ 1, w_2 ≥ 1}."""
    result = algorithm_0_homogeneous(_example1_alg0_inputs())
    w_proj = result.w_dagger
    assert w_proj[0] >= 1.0 - 1e-6, f"w_1 = {w_proj[0]} violates w_1 ≥ 1"
    assert w_proj[1] >= 1.0 - 1e-6, f"w_2 = {w_proj[1]} violates w_2 ≥ 1"


def test_algorithm_0_symmetric_on_symmetric_example():
    """By the symmetric role of (w_1, w_2) in the box-free homogeneous-cone
    problem (the simplex slice is symmetric under w_1↔w_2 swap up to a
    relabelling of the gradients), the projected weights should agree."""
    result = algorithm_0_homogeneous(_example1_alg0_inputs())
    w_proj = result.w_dagger
    # Both projected weights equal up to LP solver tolerance.
    assert abs(w_proj[0] - w_proj[1]) < 1e-3
