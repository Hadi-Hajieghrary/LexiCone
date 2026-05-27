"""Online deployment policy (v10_2 Section 9.6).

Implements the three-step online deployment recipe of Section 9.6:

1. **Solve** the WS problem with the cached calibrated weights
   :math:`w^\\dagger`.
2. **Monitor** the binary compliance vector at the WS solution.
3. **Verify**: compare to the expected compliance pattern from the offline
   lex cascade. On match, accept the WS solution. On mismatch, the active-set
   structure has changed; the operator chooses between two responses:

   - **(a)** trigger Algorithm 1A/1B to recompute :math:`w^\\dagger` for the
     new active set (online recalibration), or
   - **(b)** fall back to the lex cascade for this MPC instance.

This module is the abstract policy layer above the compliance check. The
caller wires in:

- a ``cascade_solver`` callable that returns the lex optimum for an
  arbitrary :class:`ConvexPriorityProblem`,
- a ``ws_solver`` callable that returns the WS optimum for an arbitrary
  problem + weight vector, and
- (optionally) a ``recalibration_callable`` that re-runs Algorithm 1A/1B on
  the current active set and returns updated :math:`w^\\dagger`.

The policy honors a per-tick budget: if the cascade is too slow to run
inline at the simulator rate, the operator may disable fallback and rely on
recalibration alone, or vice versa.

This module is dynamics-agnostic — it does not depend on the bicycle model,
nuPlan, or the observer. It composes the lcp/ pieces (cascade, equivalence,
compliance) into an end-to-end online policy.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, List, Literal, Optional

import numpy as np

logger = logging.getLogger(__name__)

FallbackStrategy = Literal["cascade", "recalibrate", "cascade_then_recalibrate", "log_only"]


@dataclass
class OnlineDeploymentConfig:
    """Configuration for the online deployment policy.

    Parameters
    ----------
    strategy
        How to respond to a compliance mismatch:

        - ``"cascade"`` — fall back to the lex cascade for this tick
          (Section 9.6 Step 3 option b). Slow but trivially correct.
        - ``"recalibrate"`` — trigger Algorithm 1A/1B to recompute
          :math:`w^\\dagger` for the new active set (Section 9.6 Step 3
          option a). Faster than cascade but more complex; the
          recalibration may fail (LP infeasible) requiring escalation.
        - ``"cascade_then_recalibrate"`` — fall back to cascade for the
          current tick AND trigger recalibration to update the cache for
          future ticks. The safest option when both budgets are available.
        - ``"log_only"`` — log the mismatch and accept the WS solution.
          Useful in diagnostic deployments where the operator wants to
          measure mismatch frequency without affecting trajectories.
    cascade_budget_s
        Soft per-tick wall-time budget for the cascade fallback. If the
        cascade exceeds this budget, log a warning. ``None`` disables the
        budget check.
    recalibration_budget_s
        Soft per-tick wall-time budget for Algorithm 1A/1B recalibration.
        ``None`` disables.
    """
    strategy: FallbackStrategy = "cascade_then_recalibrate"
    cascade_budget_s: Optional[float] = 0.5
    recalibration_budget_s: Optional[float] = 0.1


@dataclass
class TickResult:
    """Outcome of one online-deployment tick."""
    accepted_solution: np.ndarray
    """The solution returned to the caller (z_ws or z_lex)."""
    compliance_matched: bool
    """True iff the WS compliance vector matched the cached lex pattern."""
    fallback_triggered: bool
    """True iff the cascade fallback was invoked this tick."""
    recalibration_triggered: bool
    """True iff Algorithm 1A/1B recalibration was invoked this tick."""
    new_w_dagger: Optional[np.ndarray]
    """If recalibration succeeded, the updated weight vector. None otherwise."""
    solve_wall_time_s: float
    """Wall time of the WS solve (always reported)."""
    cascade_wall_time_s: float
    """Wall time of the cascade fallback (0 if not triggered)."""
    recalibration_wall_time_s: float
    """Wall time of recalibration (0 if not triggered)."""
    note: str = ""


@dataclass
class OnlineDeploymentStats:
    """Cumulative statistics across many online-deployment ticks."""
    total_ticks: int = 0
    matched_ticks: int = 0
    mismatched_ticks: int = 0
    fallback_count: int = 0
    recalibration_count: int = 0
    recalibration_failures: int = 0
    total_solve_time_s: float = 0.0
    total_cascade_time_s: float = 0.0
    total_recalibration_time_s: float = 0.0

    @property
    def mismatch_rate(self) -> float:
        """Fraction of ticks at which the WS compliance vector disagreed with
        the cached lex pattern."""
        return self.mismatched_ticks / max(1, self.total_ticks)

    def update(self, r: TickResult) -> None:
        self.total_ticks += 1
        if r.compliance_matched:
            self.matched_ticks += 1
        else:
            self.mismatched_ticks += 1
        if r.fallback_triggered:
            self.fallback_count += 1
        if r.recalibration_triggered:
            self.recalibration_count += 1
            if r.new_w_dagger is None:
                self.recalibration_failures += 1
        self.total_solve_time_s += r.solve_wall_time_s
        self.total_cascade_time_s += r.cascade_wall_time_s
        self.total_recalibration_time_s += r.recalibration_wall_time_s


def deploy_tick(
    *,
    ws_solver: Callable[[np.ndarray], np.ndarray],
    compliance_vector: Callable[[np.ndarray], np.ndarray],
    expected_compliance: np.ndarray,
    w_dagger: np.ndarray,
    config: OnlineDeploymentConfig,
    cascade_solver: Optional[Callable[[], np.ndarray]] = None,
    recalibrate: Optional[Callable[[np.ndarray], Optional[np.ndarray]]] = None,
) -> TickResult:
    """Execute one online-deployment tick per v10_2 Section 9.6.

    Parameters
    ----------
    ws_solver
        Callable ``ws_solver(w) -> z`` returning the WS optimum for the
        current MPC instance at weight vector ``w``.
    compliance_vector
        Callable ``compliance_vector(z) -> b`` returning the binary
        compliance vector (one entry per priority level) for trajectory
        ``z``.
    expected_compliance
        Cached compliance pattern from the offline lex cascade,
        ``b(z_lex*)`` for L_1 or ``b_eps(z_lex*)`` for L_2.
    w_dagger
        Current calibrated weight vector from Algorithm 1A or 1B.
    config
        Online deployment configuration (strategy + budgets).
    cascade_solver
        Required when ``config.strategy`` involves cascade fallback. Returns
        ``z_lex*`` for the current MPC instance.
    recalibrate
        Required when ``config.strategy`` involves recalibration. Called
        with the WS trajectory ``z_ws`` whose active set differs from the
        cached pattern; returns the updated ``w_dagger`` or ``None`` if
        Algorithm 1A/1B was infeasible.

    Returns
    -------
    TickResult
        Per-tick outcome with diagnostic timing.
    """
    t0 = time.perf_counter()
    z_ws = ws_solver(w_dagger)
    solve_time = time.perf_counter() - t0

    b_actual = np.asarray(compliance_vector(z_ws))
    matched = bool(np.array_equal(b_actual, expected_compliance))

    if matched:
        return TickResult(
            accepted_solution=z_ws,
            compliance_matched=True,
            fallback_triggered=False,
            recalibration_triggered=False,
            new_w_dagger=None,
            solve_wall_time_s=solve_time,
            cascade_wall_time_s=0.0,
            recalibration_wall_time_s=0.0,
            note="WS compliance matched cached lex pattern",
        )

    logger.warning(
        "compliance mismatch: actual=%s expected=%s; applying strategy=%s",
        b_actual.tolist(), expected_compliance.tolist(), config.strategy,
    )

    accepted = z_ws
    fallback = False
    recalibrated = False
    new_w = None
    cascade_time = 0.0
    recal_time = 0.0
    note = "WS compliance mismatched"

    if config.strategy in ("cascade", "cascade_then_recalibrate"):
        if cascade_solver is None:
            raise ValueError(
                f"strategy={config.strategy} requires cascade_solver callable"
            )
        t1 = time.perf_counter()
        z_lex = cascade_solver()
        cascade_time = time.perf_counter() - t1
        if (config.cascade_budget_s is not None
                and cascade_time > config.cascade_budget_s):
            logger.warning(
                "cascade fallback exceeded budget: %.3fs > %.3fs",
                cascade_time, config.cascade_budget_s,
            )
        accepted = z_lex
        fallback = True
        note = "WS compliance mismatched; cascade fallback applied"

    if config.strategy in ("recalibrate", "cascade_then_recalibrate"):
        if recalibrate is None:
            raise ValueError(
                f"strategy={config.strategy} requires recalibrate callable"
            )
        t2 = time.perf_counter()
        candidate_w = recalibrate(z_ws)
        recal_time = time.perf_counter() - t2
        if (config.recalibration_budget_s is not None
                and recal_time > config.recalibration_budget_s):
            logger.warning(
                "recalibration exceeded budget: %.3fs > %.3fs",
                recal_time, config.recalibration_budget_s,
            )
        recalibrated = True
        new_w = candidate_w
        if candidate_w is None:
            note += " (recalibration failed; keeping cached w_dagger)"
        else:
            note += f" (recalibration produced new w_dagger={candidate_w.tolist()})"

    if config.strategy == "log_only":
        note = "WS compliance mismatched; logged only, no fallback"

    return TickResult(
        accepted_solution=accepted,
        compliance_matched=False,
        fallback_triggered=fallback,
        recalibration_triggered=recalibrated,
        new_w_dagger=new_w,
        solve_wall_time_s=solve_time,
        cascade_wall_time_s=cascade_time,
        recalibration_wall_time_s=recal_time,
        note=note,
    )
