"""Pre-flight failure-mode diagnostics (v10_2 Section 8).

This module implements the practitioner checklist of Section 8.5 so that
deployments can verify the framework's standing assumptions *before*
running Algorithm 1A or 1B. Two failure modes are checked:

- **Failure Mode I (Section 8.1)** — degenerate active set (LICQ failure).
  The combined active-gradient matrix at :math:`z_\\text{lex}^\\star` must
  have full column rank ``|E*| + |I*|``; a deficit reduces the dimension of
  the equivalence region and invalidates the full-dimensionality guarantee
  of Theorem 5.1.

- **Failure Mode II (Section 8.2)** — non-convexity. Each :math:`V_i` and
  :math:`J` must be convex on :math:`\\mathcal{Z}`. For the affine-encoder
  case (every :math:`g_{i,j}(z) = a^T z - b` is affine), convexity holds by
  construction since :math:`V_i = \\sum_j \\rho(g_{i,j})` with :math:`\\rho`
  monotone-convex. For non-affine penalties this must be checked explicitly.

The diagnostics return a :class:`DiagnosticReport` which the caller can use
to decide whether to proceed with Algorithm 1A/1B (full guarantees) or fall
back to the lex cascade (Section 8.5 Step 1 fallback).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

DiagnosticOutcome = Literal["pass", "fail", "warn"]


@dataclass
class LICQReport:
    """Failure Mode I (Section 8.1) — cascade LICQ at :math:`z_\\text{lex}^\\star`.

    LICQ holds iff the combined active-gradient matrix
    :math:`G^\\star = [\\nabla h_e \\mid \\nabla g_{i,j}^{\\text{active}}]`
    has full column rank ``|E*| + |I*|``.
    """
    outcome: DiagnosticOutcome
    rank: int
    n_columns: int
    deficit: int
    smallest_singular_value: float
    notes: str = ""

    @property
    def holds(self) -> bool:
        return self.outcome == "pass"


@dataclass
class ConvexityReport:
    """Failure Mode II (Section 8.2) — convexity of each :math:`V_i` and J.

    For affine encoders this reduces to a structural check (every
    constraint emitted as :math:`a^T z + b^T u + e \\leq 0`); for non-affine
    penalties the caller must supply explicit convexity certificates per
    level or rule.
    """
    outcome: DiagnosticOutcome
    n_levels: int
    non_affine_levels: List[int] = field(default_factory=list)
    performance_is_convex: bool = True
    notes: str = ""

    @property
    def holds(self) -> bool:
        return self.outcome == "pass"


@dataclass
class DiagnosticReport:
    """Combined pre-flight report covering Section 8.5 Steps 1 and 2.

    The remaining Steps 3 (penalty form identification), 4 (region
    computation), and 5 (runtime compliance) are downstream of this report.
    A caller typically:

    1. Runs :func:`run_diagnostics` before Algorithm 1A/1B.
    2. If :attr:`framework_applies` is False, falls back to the lex cascade
       for the affected instances.
    3. Otherwise proceeds with the weight calibration.
    """
    licq: LICQReport
    convexity: ConvexityReport
    penalty_form: Literal["l1", "l2"]

    @property
    def framework_applies(self) -> bool:
        """True iff both FM I and FM II diagnostics pass."""
        return self.licq.holds and self.convexity.holds

    @property
    def practitioner_summary(self) -> str:
        """One-line summary suitable for logging."""
        if self.framework_applies:
            return (
                f"DIAGNOSTICS PASS: LICQ rank={self.licq.rank}/{self.licq.n_columns}, "
                f"convexity holds, {self.penalty_form.upper()} regime."
            )
        reasons = []
        if not self.licq.holds:
            reasons.append(f"FM I LICQ deficit={self.licq.deficit}")
        if not self.convexity.holds:
            reasons.append(
                f"FM II non-affine levels={self.convexity.non_affine_levels}"
            )
        return f"DIAGNOSTICS FAIL: {'; '.join(reasons)}; recommend lex-cascade fallback."


# ----------------------------------------------------------------------
# Failure Mode I — LICQ rank check
# ----------------------------------------------------------------------


def check_licq(
    active_equality_grads: Sequence[np.ndarray],
    active_rule_grads: Sequence[np.ndarray],
    active_phys_grads: Sequence[np.ndarray],
    tol_singular_value: float = 1e-9,
) -> LICQReport:
    """Compute the rank of the combined active-gradient matrix per v10_2 §8.1.

    Cascade LICQ holds iff the rank of
    :math:`G^\\star = [\\nabla h \\mid \\nabla g_{\\text{rule}}^{\\text{active}}
    \\mid \\nabla g_{\\text{phys}}^{\\text{active}}]`
    equals the total number of active gradients (i.e. the matrix has full
    column rank). We compute this via SVD and compare the smallest singular
    value to ``tol_singular_value``.
    """
    cols: List[np.ndarray] = []
    cols.extend(np.asarray(g, dtype=np.float64).reshape(-1) for g in active_equality_grads)
    cols.extend(np.asarray(g, dtype=np.float64).reshape(-1) for g in active_rule_grads)
    cols.extend(np.asarray(g, dtype=np.float64).reshape(-1) for g in active_phys_grads)
    n_cols = len(cols)
    if n_cols == 0:
        return LICQReport(
            outcome="pass",
            rank=0,
            n_columns=0,
            deficit=0,
            smallest_singular_value=float("inf"),
            notes="no active constraints; LICQ trivially holds",
        )
    # Stack as (d, n_cols).
    G = np.column_stack(cols)
    # Singular values; rank = count of values > tol.
    sv = np.linalg.svd(G, compute_uv=False)
    smallest = float(sv[-1]) if len(sv) > 0 else 0.0
    rank = int(np.sum(sv > tol_singular_value))
    deficit = n_cols - rank
    if deficit == 0:
        return LICQReport(
            outcome="pass",
            rank=rank,
            n_columns=n_cols,
            deficit=0,
            smallest_singular_value=smallest,
            notes=f"full rank ({rank} / {n_cols})",
        )
    return LICQReport(
        outcome="fail",
        rank=rank,
        n_columns=n_cols,
        deficit=deficit,
        smallest_singular_value=smallest,
        notes=(
            f"LICQ DEFICIT: rank({rank}) < n_active({n_cols}); "
            f"equivalence region is at most ({rank})-dimensional. "
            f"Section 8.1 remedy: regularise the cascade or restrict weight selection."
        ),
    )


# ----------------------------------------------------------------------
# Failure Mode II — convexity check
# ----------------------------------------------------------------------


def check_convexity(
    n_levels: int,
    all_constraints_affine: bool = True,
    non_affine_level_indices: Optional[Sequence[int]] = None,
    performance_is_convex: bool = True,
) -> ConvexityReport:
    """Verify convexity of every :math:`V_i` and the performance objective J
    per v10_2 §8.2.

    For the deployment-default case where every rule encoder emits affine
    inequalities :math:`a^T z + b^T u + e \\leq 0` and J is linear or convex
    quadratic, the check is structural — pass ``all_constraints_affine=True``
    and ``performance_is_convex=True``. For applications that use non-affine
    encoders (e.g. log-barrier penalties), pass the indices of the offending
    levels in ``non_affine_level_indices``.

    The convex case holds because :math:`V_i = \\sum_j \\rho([g_{i,j}]_+)`
    with :math:`\\rho \\in \\{x, x^2\\}` is monotone-convex composed with
    convex (affine in this case), so :math:`V_i` is convex; and J is given
    as convex by hypothesis.
    """
    non_affine = list(non_affine_level_indices) if non_affine_level_indices else []
    if all_constraints_affine and performance_is_convex and not non_affine:
        return ConvexityReport(
            outcome="pass",
            n_levels=n_levels,
            performance_is_convex=True,
            notes=(
                "all encoders emit affine inequalities and J is convex; "
                f"V_i is monotone-composition-convex on all {n_levels} levels"
            ),
        )
    if non_affine or not all_constraints_affine:
        return ConvexityReport(
            outcome="fail",
            n_levels=n_levels,
            non_affine_levels=non_affine,
            performance_is_convex=performance_is_convex,
            notes=(
                "NON-CONVEX V_i: non-affine encoders flagged "
                f"on levels {non_affine}. "
                f"Section 8.2 remedy: independent LEX-vs-WS verification per instance."
            ),
        )
    return ConvexityReport(
        outcome="fail",
        n_levels=n_levels,
        non_affine_levels=[],
        performance_is_convex=False,
        notes=(
            "PERFORMANCE OBJECTIVE NON-CONVEX: J is non-convex; "
            f"Section 8.2 remedy: independent LEX-vs-WS verification per instance."
        ),
    )


# ----------------------------------------------------------------------
# Combined pre-flight check (Section 8.5 Steps 1 + 2)
# ----------------------------------------------------------------------


def run_diagnostics(
    active_equality_grads: Sequence[np.ndarray],
    active_rule_grads: Sequence[np.ndarray],
    active_phys_grads: Sequence[np.ndarray],
    n_levels: int,
    penalty_form: Literal["l1", "l2"] = "l1",
    all_constraints_affine: bool = True,
    non_affine_level_indices: Optional[Sequence[int]] = None,
    performance_is_convex: bool = True,
    licq_singular_value_tol: float = 1e-9,
) -> DiagnosticReport:
    """Run the combined Section 8.5 practitioner checklist (Steps 1 + 2).

    Step 1 (FM II convexity) and Step 2 (FM I LICQ) are returned in a
    combined :class:`DiagnosticReport`. The caller inspects
    :attr:`DiagnosticReport.framework_applies` to decide whether to proceed
    with Algorithm 1A / 1B (full guarantees) or fall back to the lex cascade
    on the affected instances (Section 8.5 Steps 3 onward).
    """
    licq = check_licq(
        active_equality_grads=active_equality_grads,
        active_rule_grads=active_rule_grads,
        active_phys_grads=active_phys_grads,
        tol_singular_value=licq_singular_value_tol,
    )
    convexity = check_convexity(
        n_levels=n_levels,
        all_constraints_affine=all_constraints_affine,
        non_affine_level_indices=non_affine_level_indices,
        performance_is_convex=performance_is_convex,
    )
    report = DiagnosticReport(licq=licq, convexity=convexity, penalty_form=penalty_form)
    logger.info(report.practitioner_summary)
    return report
