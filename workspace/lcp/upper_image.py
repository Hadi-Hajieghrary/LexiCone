"""Upper image and achievement map (v10_2 Section 3).

Reifies the geometric objects of v10_2 Section 3 that the algorithms of
:mod:`lcp.equivalence` use implicitly:

- The **achievement map** :math:`\\Phi : \\mathcal{Z} \\to \\mathbb{R}^{L+1}`
  sending a trajectory to its violation+performance image
  :math:`(V_1(z), \\ldots, V_L(z), J(z))` (Eq. 1).
- The **upper image** :math:`\\overline{\\mathcal{P}}` of :math:`\\Phi`
  (Definition 3.1).
- The **lex image** :math:`p^\\star = \\Phi(z_\\text{lex}^\\star)` that is
  the extreme point of :math:`\\overline{\\mathcal{P}}` per Lemma 3.2.

The upper image is in general an infinite-dimensional object (a closed
convex set in :math:`\\mathbb{R}^{L+1}`); this module does not try to
represent it as a polytope. What it does represent is the **lex image
point** :math:`p^\\star`, together with a numerical extreme-point check
that confirms the property of Lemma 3.2 on a finite probe set.

The algorithms in :mod:`lcp.equivalence` and :mod:`lcp.diagnostics` use
:math:`p^\\star` implicitly through the active-set signature; this module
makes the geometric object explicit so that callers can:

- Evaluate :math:`\\Phi(z)` at any trajectory.
- Construct :math:`p^\\star` from a lex cascade output.
- Numerically verify that :math:`p^\\star` is an extreme point of the upper
  image (Lemma 3.2).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


@dataclass(frozen=True)
class AchievementImage:
    """One point :math:`\\Phi(z) = (V_1(z), \\ldots, V_L(z), J(z))` in image
    space :math:`\\mathbb{R}^{L+1}`.

    Per v10_2 Eq. (1), the first :math:`L` coordinates are the per-priority
    level integrated violations :math:`V_i \\geq 0` and the last coordinate
    is the performance objective :math:`J`. Two image points are equal iff
    every coordinate matches to within numerical tolerance.
    """
    V: np.ndarray   # shape (L,), each entry V_i(z) >= 0
    J: float

    def __post_init__(self) -> None:
        if self.V.ndim != 1:
            raise ValueError(f"V must be 1-D; got shape {self.V.shape}")
        if np.any(self.V < -1e-9):
            raise ValueError(
                f"V must be component-wise >= 0; got min = {float(self.V.min())}"
            )

    @property
    def n_levels(self) -> int:
        return int(self.V.shape[0])

    @property
    def vector(self) -> np.ndarray:
        """Concatenated representation ``(V_1, ..., V_L, J)``."""
        return np.concatenate([self.V, [self.J]])

    def dominates(self, other: "AchievementImage", lex_tol: float = 1e-9) -> bool:
        """True iff ``self`` is lex-less-than-or-equal-to ``other``.

        The lex order on :math:`\\mathbb{R}^{L+1}` compares the violation
        coordinates first (in priority order) and then the performance
        coordinate as the final tie-breaker, exactly the order under which
        the lex cascade chooses its optimum.
        """
        if self.n_levels != other.n_levels:
            raise ValueError(f"shape mismatch {self.V.shape} vs {other.V.shape}")
        for i in range(self.n_levels):
            if self.V[i] < other.V[i] - lex_tol:
                return True
            if self.V[i] > other.V[i] + lex_tol:
                return False
        return self.J <= other.J + lex_tol


def evaluate_achievement_map(
    z: np.ndarray,
    violation_functionals: Sequence[Callable[[np.ndarray], float]],
    performance_objective: Callable[[np.ndarray], float],
) -> AchievementImage:
    """Compute :math:`\\Phi(z) = (V_1(z), \\ldots, V_L(z), J(z))`.

    Parameters
    ----------
    z
        Trajectory vector in :math:`\\mathcal{Z}`.
    violation_functionals
        Length-``L`` sequence ``[V_1, ..., V_L]`` of callables returning
        non-negative scalars.
    performance_objective
        The callable :math:`J(z)`.
    """
    V = np.array([float(Vi(z)) for Vi in violation_functionals], dtype=np.float64)
    J = float(performance_objective(z))
    return AchievementImage(V=V, J=J)


def lex_image_from_cascade(
    z_lex: np.ndarray,
    violation_functionals: Sequence[Callable[[np.ndarray], float]],
    performance_objective: Callable[[np.ndarray], float],
) -> AchievementImage:
    """Construct :math:`p^\\star = \\Phi(z_\\text{lex}^\\star)` from a lex
    cascade output.

    By construction of the cascade (\\cite{lcp2025} Section 2.3),
    :math:`p^\\star` is the smallest image under the lex order and is the
    extreme point of :math:`\\overline{\\mathcal{P}}` characterised by
    Lemma 3.2.
    """
    return evaluate_achievement_map(z_lex, violation_functionals, performance_objective)


def verify_lex_extreme_point(
    p_star: AchievementImage,
    feasible_image_probes: Sequence[AchievementImage],
    lex_tol: float = 1e-6,
) -> bool:
    """Numerically verify Lemma 3.2: :math:`p^\\star` is an extreme point of
    :math:`\\overline{\\mathcal{P}}`.

    An extreme point of a convex set is not a strict convex combination of
    two distinct points of the set. We probe this property on a finite
    sample of feasible image points: for every pair
    :math:`(p^{(1)}, p^{(2)})` of probes, we check that
    :math:`p^\\star = \\frac{1}{2}(p^{(1)} + p^{(2)})` implies
    :math:`p^{(1)} = p^{(2)} = p^\\star` (up to ``lex_tol``).

    A counterexample is sufficient to refute extremity; passing on a finite
    probe set is necessary but not sufficient (the theoretical Lemma 3.2
    proof in v10_2 \\S3 is the sufficient certificate).
    """
    n = len(feasible_image_probes)
    for i in range(n):
        for j in range(i + 1, n):
            p1, p2 = feasible_image_probes[i], feasible_image_probes[j]
            mid = 0.5 * (p1.vector + p2.vector)
            if not np.allclose(mid, p_star.vector, atol=lex_tol):
                continue
            # Midpoint coincides with p_star ⇒ p1, p2 must coincide with p_star.
            if not (np.allclose(p1.vector, p_star.vector, atol=lex_tol)
                    and np.allclose(p2.vector, p_star.vector, atol=lex_tol)):
                return False
    return True


def upper_image_membership(
    candidate: AchievementImage,
    feasible_image_probes: Sequence[AchievementImage],
    lex_tol: float = 1e-9,
) -> bool:
    """Check whether ``candidate`` componentwise dominates *some* feasible
    image probe (i.e., whether ``candidate`` is in the upward closure of the
    probe set, a finite under-approximation of :math:`\\overline{\\mathcal P}`).

    Per Definition 3.1, :math:`(y, t) \\in \\overline{\\mathcal{P}}` iff
    there exists :math:`z \\in \\mathcal{Z}` with :math:`V_i(z) \\leq y_i`
    for all :math:`i` and :math:`J(z) \\leq t`. We approximate the
    "exists :math:`z`" by "exists a probe image" --- a finite check whose
    positive result is sufficient for membership and whose negative result
    is necessary but not sufficient for non-membership.
    """
    cv = candidate.vector
    for p in feasible_image_probes:
        if np.all(cv >= p.vector - lex_tol):
            return True
    return False
