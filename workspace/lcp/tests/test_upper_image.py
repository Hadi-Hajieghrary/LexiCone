"""Tests for ``lcp.upper_image`` (v10_2 Section 3 reification).

Constructs Example 1 (\\cite{lcp2025} \\S11.1) and verifies:

- :math:`\\Phi(z_\\text{lex}^\\star) = (0, 0, -11)`.
- ``AchievementImage`` lex-order arithmetic.
- Lemma 3.2 numerical extreme-point check passes for :math:`p^\\star` against
  a probe set of feasible vertex images.
"""
from __future__ import annotations

import numpy as np
import pytest

from lcp.upper_image import (
    AchievementImage,
    evaluate_achievement_map,
    lex_image_from_cascade,
    upper_image_membership,
    verify_lex_extreme_point,
)


# Example 1 from v10_2 §11.1:
#   Z = [0, 10]^2
#   Level 1 (high):  z_1 + z_2 <= 8     -> V_1 = max(0, z_1 + z_2 - 8)
#   Level 2 (low):   z_1       <= 3     -> V_2 = max(0, z_1 - 3)
#   J(z) = -2 z_1 - z_2
# Lex optimum z_lex* = (3, 5); p* = (0, 0, -11).


def _ex1_V() -> list:
    return [
        lambda z: max(0.0, z[0] + z[1] - 8.0),
        lambda z: max(0.0, z[0] - 3.0),
    ]


def _ex1_J():
    return lambda z: -2.0 * z[0] - z[1]


def test_achievement_image_validates_negative_violation():
    with pytest.raises(ValueError, match="V must be component-wise"):
        AchievementImage(V=np.array([-1.0, 0.0]), J=0.0)


def test_evaluate_achievement_map_at_lex_optimum_example1():
    z_lex = np.array([3.0, 5.0])
    p = evaluate_achievement_map(z_lex, _ex1_V(), _ex1_J())
    np.testing.assert_allclose(p.V, [0.0, 0.0], atol=1e-12)
    assert p.J == pytest.approx(-11.0, abs=1e-12)


def test_lex_image_from_cascade_matches_paper():
    z_lex = np.array([3.0, 5.0])
    p_star = lex_image_from_cascade(z_lex, _ex1_V(), _ex1_J())
    assert p_star.n_levels == 2
    np.testing.assert_allclose(p_star.vector, [0.0, 0.0, -11.0], atol=1e-12)


def test_dominates_lex_order_basic():
    p1 = AchievementImage(V=np.array([0.0, 0.0]), J=-11.0)   # paper lex point
    p2 = AchievementImage(V=np.array([0.0, 0.5]), J=-12.0)   # worse at L_2
    # p1 lex-dominates p2 because V_1 ties but V_2 is smaller.
    assert p1.dominates(p2)
    assert not p2.dominates(p1)


def test_extreme_point_property_holds_at_lex_optimum():
    # Polytope vertices of {z in [0,10]^2 : V_1=0, V_2=0} are
    # (0,0), (3,0), (3,5), (0,8). Their images:
    V_fns = _ex1_V()
    J_fn = _ex1_J()
    probes = [
        evaluate_achievement_map(np.array([0.0, 0.0]), V_fns, J_fn),    # (0, 0, 0)
        evaluate_achievement_map(np.array([3.0, 0.0]), V_fns, J_fn),    # (0, 0, -6)
        evaluate_achievement_map(np.array([0.0, 8.0]), V_fns, J_fn),    # (0, 0, -8)
        evaluate_achievement_map(np.array([3.0, 5.0]), V_fns, J_fn),    # (0, 0, -11)
    ]
    p_star = probes[-1]
    # The lex point is the strictly minimal-J vertex among the V=0 vertices.
    # No two distinct probes have midpoint == p_star (no pair averages to -11),
    # so the extreme-point check passes vacuously on this probe set.
    assert verify_lex_extreme_point(p_star, probes)


def test_extreme_point_check_catches_non_extreme():
    # Construct a probe set where p_star is the midpoint of two distinct probes.
    p1 = AchievementImage(V=np.array([0.0, 0.0]), J=-10.0)
    p2 = AchievementImage(V=np.array([0.0, 0.0]), J=-12.0)
    fake_p_star = AchievementImage(V=np.array([0.0, 0.0]), J=-11.0)
    # The fake p* is the midpoint of p1 and p2 ⇒ not an extreme point.
    assert not verify_lex_extreme_point(fake_p_star, [p1, p2, fake_p_star])


def test_upper_image_membership_positive():
    V_fns = _ex1_V()
    J_fn = _ex1_J()
    p_star = evaluate_achievement_map(np.array([3.0, 5.0]), V_fns, J_fn)  # (0,0,-11)
    # Any image (y_1, y_2, t) with y_i >= 0 and t >= -11 is in the upper image.
    candidate_in = AchievementImage(V=np.array([1.0, 1.0]), J=-5.0)
    assert upper_image_membership(candidate_in, [p_star])


def test_upper_image_membership_negative():
    V_fns = _ex1_V()
    J_fn = _ex1_J()
    p_star = evaluate_achievement_map(np.array([3.0, 5.0]), V_fns, J_fn)
    # A point with t < -11 cannot be dominated by p_star (J would have to be lower).
    candidate_out = AchievementImage(V=np.array([0.0, 0.0]), J=-20.0)
    assert not upper_image_membership(candidate_out, [p_star])
