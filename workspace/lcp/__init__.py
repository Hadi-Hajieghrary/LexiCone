"""Lexicographic Constraint Programming (LCP) — reference implementation of
the framework specified in ``References/lex_constraint_programming_report_v10_2.md``.

This package implements the algorithmic content of the v10_2 paper as a
dynamics-agnostic library. Public API:

Problem definition (v10_2 Section 2):
    :class:`ConvexPriorityProblem`, :class:`AffineDynamics`, :class:`LevelSpec`,
    :class:`AffineConstraint`, :class:`BoxBounds`, :class:`PerformanceObjective`.

Equivalence theory and weight calibration (Sections 4-7, 9):
    :func:`algorithm_0_homogeneous` — homogeneous-cone primary (Section 9.1)
    :func:`algorithm_1a` — box-bounded L1 Chebyshev (Section 9.3)
    :func:`algorithm_1b` — pointwise coupled-linear L2 Chebyshev (Section 9.4)
    :func:`omega_half_space_description` — explicit polyhedral form of
        Omega(p*) (Section 5).

Pre-flight diagnostics (Section 8):
    :func:`run_diagnostics` — combined FM I (LICQ) + FM II (convexity) check.

Compliance and online deployment (Sections 7, 9.6):
    :class:`ComplianceChecker` — binary compliance vector b(z) / b_eps(z).
    :func:`deploy_tick` — online deployment with cascade fallback and
        Algorithm 1A/1B recalibration on mismatch.

Relaxation Decision Framework (Section 10):
    :func:`compute_necessary_relaxation_level` — Phase-I sequential necessity
    :func:`decide_level_relaxation` — Theorems 10.2a + 10.2b combined
    :func:`iterative_lex_relaxation` — Procedure 10.1

Calibration cache:
    :class:`CalibrationCache` — JSON-backed cache of computed w_dagger.

The kinematic-bicycle-specific MPC application lives outside this package
(under ``lexicone/`` for the nuPlan demo); this package solves arbitrary
convex priority-ordered programs that match the v10_2 problem structure.
"""

from lcp.cache import CalibrationCache, CalibrationEntry, HEURISTIC_DEFAULTS
from lcp.compliance import ComplianceChecker, ComplianceResult, MismatchRecord
from lcp.diagnostics import (
    ConvexityReport,
    DiagnosticReport,
    LICQReport,
    check_convexity,
    check_licq,
    run_diagnostics,
)
from lcp.equivalence import (
    ActiveConstraint,
    Algorithm0Inputs,
    Algorithm0Result,
    L2SensitivityConstants,
    L2SensitivityInputs,
    WeightCalibrationInputs,
    WeightCalibrationResult,
    algorithm_0_homogeneous,
    algorithm_1a,
    algorithm_1b,
    compute_l2_sensitivity_constants,
    l2_threshold,
    omega_half_space_description,
)
from lcp.online import (
    OnlineDeploymentConfig,
    OnlineDeploymentStats,
    TickResult,
    deploy_tick,
)
from lcp.problem import (
    AffineConstraint,
    AffineDynamics,
    BoxBounds,
    ConvexPriorityProblem,
    LevelSpec,
    PerformanceObjective,
)
from lcp.relaxation import (
    IterativeRelaxationResult,
    IterativeRelaxationStep,
    LevelRelaxationDecision,
    NecessityReport,
    ValueFunctionPoint,
    compute_necessary_relaxation_level,
    decide_level_relaxation,
    evaluate_value_function,
    find_knee_depth,
    iterative_lex_relaxation,
    mrs_at_zero,
)
from lcp.upper_image import (
    AchievementImage,
    evaluate_achievement_map,
    lex_image_from_cascade,
    upper_image_membership,
    verify_lex_extreme_point,
)

__all__ = [
    # Problem
    "AffineConstraint", "AffineDynamics", "BoxBounds",
    "ConvexPriorityProblem", "LevelSpec", "PerformanceObjective",
    # Equivalence + calibration
    "ActiveConstraint", "Algorithm0Inputs", "Algorithm0Result",
    "L2SensitivityConstants", "L2SensitivityInputs",
    "WeightCalibrationInputs", "WeightCalibrationResult",
    "algorithm_0_homogeneous", "algorithm_1a", "algorithm_1b",
    "compute_l2_sensitivity_constants", "l2_threshold",
    "omega_half_space_description",
    # Diagnostics
    "ConvexityReport", "DiagnosticReport", "LICQReport",
    "check_convexity", "check_licq", "run_diagnostics",
    # Compliance + online
    "ComplianceChecker", "ComplianceResult", "MismatchRecord",
    "OnlineDeploymentConfig", "OnlineDeploymentStats",
    "TickResult", "deploy_tick",
    # Relaxation
    "IterativeRelaxationResult", "IterativeRelaxationStep",
    "LevelRelaxationDecision", "NecessityReport", "ValueFunctionPoint",
    "compute_necessary_relaxation_level", "decide_level_relaxation",
    "evaluate_value_function", "find_knee_depth",
    "iterative_lex_relaxation", "mrs_at_zero",
    # Cache
    "CalibrationCache", "CalibrationEntry", "HEURISTIC_DEFAULTS",
    # Upper image (Section 3 reification)
    "AchievementImage", "evaluate_achievement_map", "lex_image_from_cascade",
    "upper_image_membership", "verify_lex_extreme_point",
]
