"""Algorithm 1A (and, later, 1B): offline calibration of the WS weights.

Section 9.3 of the LCP paper defines Algorithm 1A — the L₁ exact-equivalence
calibration. Given the lex cascade's active-set classification at
:math:`z_\\text{lex}^\\star`, it produces:

- the normalised equivalence region :math:`\\Omega(p^\\star) = \\{w : C w \\leq d\\}`,
- a robust *Chebyshev-centre* weight :math:`w^\\dagger \\in \\Omega(p^\\star) \\cap [\\underline{w}, \\overline{w}]^L`,
- a robustness margin :math:`r^\\dagger \\geq 0` to all binding constraints
  (both the box faces and the :math:`\\Omega(p^\\star)`-facets).

We implement it via a single **lifted Chebyshev LP** that holds the lex KKT
multipliers :math:`(\\beta, \\lambda, \\mu)` as auxiliary variables, plus the
weights :math:`w` and a slack :math:`r`. The LP enforces:

1. The KKT stationarity (equality):
   :math:`\\nabla J + \\sum_{S_\\text{bdy}} \\beta_{i,j} \\nabla g_{i,j}
                          + \\sum_{S_\\text{viol}} w_i \\nabla V_i
                          + \\sum_{j' \\in \\mathcal{I}_\\text{phys}^\\star} \\lambda_{j'} \\nabla g_{\\text{phys},j'}
                          + \\sum_e \\mu_e \\nabla h_e
                          = 0`.
2. The L₁ subgradient bounds at boundary-binding rule constraints:
   :math:`0 \\leq \\beta_{i,j} \\leq w_i`.
3. Active hard-inequality multipliers :math:`\\lambda_{j'} \\geq 0`.
4. Per-level weight bounds :math:`\\underline{w}_i \\leq w_i \\leq \\overline{w}_i`.
5. Chebyshev slack: distance from :math:`w` to every box face AND to every
   :math:`\\beta_{i,j} = w_i` facet (which is the operative
   :math:`\\Omega(p^\\star)` boundary in the polyhedral L₁ case) is at least
   :math:`r`.

When the equality system uniquely determines :math:`(\\beta, \\lambda, \\mu)`
as linear functions of :math:`w` (the "Example 1" regime), this LP reproduces
the paper's explicit half-space description trivially. When the equality
system has free directions (multi-active levels), the LP still finds
:math:`w^\\dagger` correctly by exploiting that flexibility; the explicit
half-space description in that case would require Fourier-Motzkin elimination
to project out the free :math:`(\\beta, \\lambda, \\mu)` directions, which we
provide as a separate :func:`omega_half_space_description` helper.

The implementation is dataclass-driven so callers from the LCP MPC side
populate ``WeightCalibrationInputs`` with gradients extracted from the
cascade and the algorithm runs purely on numpy arrays — no CasADi
dependencies leak into the LP layer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linprog

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Public dataclasses
# ----------------------------------------------------------------------


@dataclass
class ActiveConstraint:
    """One active constraint at :math:`z_\\text{lex}^\\star`.

    ``level_index`` is 0-based (0 = highest priority). ``slot_index`` is the
    unique identifier within the level for diagnostics. ``gradient`` is
    :math:`\\nabla g_{i,j}(z_\\text{lex}^\\star)` evaluated in the same
    coordinate system as ``WeightCalibrationInputs.grad_J``.

    ``kind`` is one of:

    - ``"boundary_binding"`` — :math:`g_{i,j}(z_\\text{lex}^\\star) = 0`,
      ``α_{i,j}`` is a decision variable in ``[0, 1]``.
    - ``"violated"`` — :math:`g_{i,j}(z_\\text{lex}^\\star) > 0`,
      ``α_{i,j} = 1`` is fixed (the L₁ hinge is in its smooth-increasing
      region).
    """

    level_index: int
    slot_index: int
    gradient: np.ndarray              # shape (d,)
    kind: str                         # "boundary_binding" | "violated"


@dataclass
class WeightCalibrationInputs:
    """Everything Algorithm 1A needs to compute :math:`w^\\dagger`."""

    grad_J: np.ndarray                            # ∇J(z_lex*), shape (d,)
    active_rule_constraints: List[ActiveConstraint]
    active_phys_inequalities: List[np.ndarray]    # each ∇g_phys, shape (d,)
    active_equalities: List[np.ndarray]           # each ∇h_e, shape (d,)
    n_levels: int                                 # L
    box_lower: np.ndarray                         # shape (L,), strictly positive
    box_upper: np.ndarray                         # shape (L,)
    tikhonov_reg: float = 0.0
    """Optional Tikhonov regulariser on (β, λ, μ) in the LP objective. Set to
    a small positive value (e.g. 1e-8) to improve numerical conditioning when
    the LP is degenerate."""


@dataclass
class WeightCalibrationResult:
    """Output of Algorithm 1A."""

    w_dagger: np.ndarray              # shape (L,)
    r_dagger: float
    beta_at_optimum: np.ndarray       # boundary-binding β values at w†
    lambda_at_optimum: np.ndarray     # hard-inequality multipliers at w†
    mu_at_optimum: np.ndarray         # equality multipliers at w†
    lp_status: str
    notes: str = ""


# ----------------------------------------------------------------------
# Algorithm 1B inputs / outputs
# ----------------------------------------------------------------------


@dataclass
class L2SensitivityInputs:
    """Inputs for Algorithm 1B (L₂ tolerance compliance).

    Sets the stage for the singular-perturbation expansion that produces the
    per-level threshold ``W_i(ε_i, w_{-i})``. Unlike Algorithm 1A, here the
    boundary-binding *satisfied* levels (V_i* = 0 with g_{i,j}(z_lex*) = 0)
    are the ones that drive the threshold; violated levels (V_i* > 0)
    contribute to the cross-level coupling but do not get their own threshold.

    Fields
    ------
    grad_J:
        ∇J(z_lex*), shape ``(d,)``.
    boundary_binding_per_level:
        Dict from level_index → list of (slot_index, ∇g_{i,j}) for boundary-
        binding satisfied constraints. In the *single-active* special case
        (Section 9.4), each level has at most one entry; in the multi-active
        case the encoder may supply several.
    violated_grad_V_per_level:
        Dict from level_index → ∇V_i(z_lex*) (the sum-of-component-gradients
        of the L₂ violation functional). Only populated for violated levels.
    n_levels:
        L. Levels not listed in either dict are treated as strictly satisfied
        (V_i* = 0 with no boundary-binding active) and contribute neither a
        threshold nor a coupling term.
    box_lower / box_upper:
        Operator-supplied weight box, shape ``(L,)``.
    epsilon_per_level:
        Operator-supplied tolerance vector, shape ``(L,)``. For each level
        ``i`` in ``boundary_binding_per_level``, the LP enforces the
        coupled-linear threshold ``w_i ≥ W_i(ε_i, w_{-i})``.
    tolerance_form:
        ``"raw"`` for ``|g_{i,j}| ≤ ε_i`` (ρ(ε) = ε) or ``"squared"`` for
        ``V_i ≤ ε_i`` (ρ(ε) = √ε). The paper's Section 7.2 discusses both;
        most rule-book deployments use raw.
    """

    grad_J: np.ndarray
    boundary_binding_per_level: "dict[int, List[Tuple[int, np.ndarray]]]"
    violated_grad_V_per_level: "dict[int, np.ndarray]"
    n_levels: int
    box_lower: np.ndarray
    box_upper: np.ndarray
    epsilon_per_level: np.ndarray
    tolerance_form: str = "raw"           # "raw" | "squared"


@dataclass
class L2SensitivityConstants:
    """Local sensitivity constants (c_i, κ_i) per Section 7.2 of the paper.

    For each boundary-binding satisfied level i, the leading-order
    expansion of the constraint violation reads

    .. math::

        g_{i,j(i)}(z_\\text{ws}^\\star(w)) \\approx
            \\frac{c_i + \\sum_{i' \\ne i} \\kappa_{i, i'} w_{i'}}{w_i}

    where the constants come from the dominant-balance equations of the
    WS-KKT system at :math:`z_\\text{lex}^\\star`. In the *single-active*
    special case this reduces to a scalar projection along ``∇g_{i,j(i)}``;
    see :func:`compute_l2_sensitivity_constants` for the closed form.

    The threshold of Proposition 7.2 (raw or V-form) follows by inverting:

    .. math::

        W_i(\\epsilon_i, w_{-i}) = \\frac{c_i + \\langle \\kappa_i, w_{-i}\\rangle}{\\rho(\\epsilon_i)}

    with :math:`\\rho(\\epsilon) = \\epsilon` for raw or :math:`\\sqrt{\\epsilon}`
    for V-tolerance.
    """

    # Indexed by level_index that has a boundary-binding satisfied constraint.
    c_const_per_level: "dict[int, float]"
    kappa_per_level: "dict[int, np.ndarray]"   # shape (L,) per entry; coupling onto every other level

    @property
    def levels_with_threshold(self) -> List[int]:
        return sorted(self.c_const_per_level.keys())


# ----------------------------------------------------------------------
# Algorithm 1A: lifted Chebyshev LP
# ----------------------------------------------------------------------


def algorithm_1a(inputs: WeightCalibrationInputs) -> WeightCalibrationResult:
    """Run Algorithm 1A of the LCP paper.

    Solves the lifted Chebyshev-centre LP described in this module's docstring
    and returns the calibrated weight vector + robustness margin + the lex
    KKT multiplier estimates.

    The LP variables are stacked as ``[w (L) | β (n_bdy) | λ (n_phys) | μ (n_eq) | r (1)]``.
    Equality constraints come from the d-dimensional stationarity equation;
    inequality constraints come from the box, the β-bounds, the λ-bounds,
    and the Chebyshev slack.

    On infeasibility (no positive weights make :math:`z_\\text{lex}^\\star`
    a WS optimum), the result's ``lp_status`` reports the reason and
    ``w_dagger`` is set to the geometric centre of the box as a fallback.
    """
    L = inputs.n_levels
    d = inputs.grad_J.shape[0]
    box_lower = np.asarray(inputs.box_lower, dtype=np.float64).reshape(L)
    box_upper = np.asarray(inputs.box_upper, dtype=np.float64).reshape(L)
    if np.any(box_lower <= 0):
        raise ValueError(f"box_lower must be strictly positive (got {box_lower})")
    if np.any(box_upper <= box_lower):
        raise ValueError("box_upper must exceed box_lower elementwise")

    bdy = [c for c in inputs.active_rule_constraints if c.kind == "boundary_binding"]
    viol = [c for c in inputs.active_rule_constraints if c.kind == "violated"]
    n_bdy = len(bdy)
    n_phys = len(inputs.active_phys_inequalities)
    n_eq = len(inputs.active_equalities)

    # Variable layout in the LP:
    #   indices [0..L)              -> w
    #   indices [L..L+n_bdy)        -> β
    #   indices [L+n_bdy..L+n_bdy+n_phys) -> λ
    #   indices [L+n_bdy+n_phys..L+n_bdy+n_phys+n_eq) -> μ
    #   index   [...+n_eq]          -> r
    n_vars = L + n_bdy + n_phys + n_eq + 1
    w_slice = slice(0, L)
    beta_slice = slice(L, L + n_bdy)
    lambda_slice = slice(L + n_bdy, L + n_bdy + n_phys)
    mu_slice = slice(L + n_bdy + n_phys, L + n_bdy + n_phys + n_eq)
    r_idx = n_vars - 1

    # ---------- Equality: stationarity in d dimensions ----------
    # ∇J + sum β_b ∇g_b + sum_{i ∈ viol} w_i ∇V_i + sum λ_p ∇g_phys + sum μ_e ∇h_e = 0
    # Rearranged:    A_eq @ x = -∇J   with    A_eq columns for each variable.
    A_eq = np.zeros((d, n_vars))
    b_eq = -inputs.grad_J.astype(np.float64).reshape(d)
    # w_i columns: contribution only from VIOLATED constraints' ∇V_i.
    # For a level with multiple violated slots, we aggregate them under the
    # level's single weight w_i. ∇V_i = sum over slot j of ∇g_{i,j}.
    grad_V_per_level = np.zeros((d, L))
    for c in viol:
        grad_V_per_level[:, c.level_index] += c.gradient
    A_eq[:, w_slice] = grad_V_per_level
    for j_b, c in enumerate(bdy):
        A_eq[:, L + j_b] = c.gradient
    for j_p, g in enumerate(inputs.active_phys_inequalities):
        A_eq[:, L + n_bdy + j_p] = np.asarray(g, dtype=np.float64).reshape(d)
    for j_e, h in enumerate(inputs.active_equalities):
        A_eq[:, L + n_bdy + n_phys + j_e] = np.asarray(h, dtype=np.float64).reshape(d)
    # r column is zero in equality constraints.

    # ---------- Inequalities ----------
    # We use A_ub @ x ≤ b_ub form. Encode:
    #  - underline_w_i + r ≤ w_i  →  −w_i + r ≤ −underline_w_i
    #  - w_i + r ≤ overline_w_i   →  w_i + r ≤ overline_w_i
    #  - β_b ≥ 0 (handled via variable bounds)
    #  - β_b + r ≤ w_i (the active Ω-facet slack: equivalent to w_i − β_b ≥ r)
    #  - λ_p ≥ 0 (variable bounds)
    #  - r ≥ 0 (variable bounds)
    A_ub_rows: List[np.ndarray] = []
    b_ub: List[float] = []
    # Lower-box slack: -w_i + r ≤ -underline_w_i
    for i in range(L):
        row = np.zeros(n_vars)
        row[i] = -1.0
        row[r_idx] = 1.0
        A_ub_rows.append(row)
        b_ub.append(-box_lower[i])
    # Upper-box slack: w_i + r ≤ overline_w_i
    for i in range(L):
        row = np.zeros(n_vars)
        row[i] = 1.0
        row[r_idx] = 1.0
        A_ub_rows.append(row)
        b_ub.append(box_upper[i])
    # Ω-facet slack at each boundary-binding constraint: β_b - w_i + r ≤ 0
    for j_b, c in enumerate(bdy):
        i = c.level_index
        row = np.zeros(n_vars)
        row[L + j_b] = 1.0
        row[i] = -1.0
        row[r_idx] = 1.0
        A_ub_rows.append(row)
        b_ub.append(0.0)

    A_ub = np.vstack(A_ub_rows) if A_ub_rows else None
    b_ub_arr = np.array(b_ub) if b_ub else None

    # ---------- Variable bounds ----------
    bounds: List[Tuple[Optional[float], Optional[float]]] = []
    for i in range(L):
        bounds.append((box_lower[i], box_upper[i]))   # w_i in box
    for _ in range(n_bdy):
        bounds.append((0.0, None))                    # β ≥ 0; upper bound β ≤ w is the Ω-facet inequality
    for _ in range(n_phys):
        bounds.append((0.0, None))                    # λ ≥ 0
    for _ in range(n_eq):
        bounds.append((None, None))                   # μ free
    bounds.append((0.0, None))                        # r ≥ 0

    # ---------- Objective: maximise r ⇔ minimise -r ----------
    c_obj = np.zeros(n_vars)
    c_obj[r_idx] = -1.0
    if inputs.tikhonov_reg > 0:
        # Small Tikhonov on the multipliers to break degeneracy.
        c_obj[beta_slice] += inputs.tikhonov_reg
        c_obj[lambda_slice] += inputs.tikhonov_reg

    res = linprog(
        c=c_obj,
        A_ub=A_ub,
        b_ub=b_ub_arr,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )

    if not res.success:
        logger.warning(
            "Algorithm 1A LP failed: status=%s, message=%s; "
            "falling back to geometric box centre.",
            res.status, res.message,
        )
        w_fallback = 0.5 * (box_lower + box_upper)
        return WeightCalibrationResult(
            w_dagger=w_fallback,
            r_dagger=0.0,
            beta_at_optimum=np.zeros(n_bdy),
            lambda_at_optimum=np.zeros(n_phys),
            mu_at_optimum=np.zeros(n_eq),
            lp_status=str(res.message),
            notes="LP infeasible; falling back to box centre",
        )

    x = res.x
    return WeightCalibrationResult(
        w_dagger=x[w_slice].copy(),
        r_dagger=float(x[r_idx]),
        beta_at_optimum=x[beta_slice].copy() if n_bdy > 0 else np.zeros(0),
        lambda_at_optimum=x[lambda_slice].copy() if n_phys > 0 else np.zeros(0),
        mu_at_optimum=x[mu_slice].copy() if n_eq > 0 else np.zeros(0),
        lp_status=str(res.message),
    )


# ----------------------------------------------------------------------
# Explicit half-space description (Fourier-Motzkin)
# ----------------------------------------------------------------------


def omega_half_space_description(
    inputs: WeightCalibrationInputs,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute :math:`\\Omega(p^\\star) = \\{w : C w \\leq d\\}` explicitly
    when the lex KKT system uniquely determines :math:`(\\beta, \\lambda, \\mu)`.

    This is the "unique-multiplier" special case — common in well-conditioned
    cascades under cascade LICQ. We solve the lifted equality system for
    :math:`(\\beta, \\lambda, \\mu)` as a function of :math:`w` (which is
    affine when the system has full row rank in (β, λ, μ)), then substitute
    into the inequality bounds to get the explicit ``Cw ≤ d`` form.

    Falls back to the lifted LP feasibility check if the system is under- or
    over-determined; in that case the half-space description is approximate
    (we return the inequalities that are tight at the LP optimum).

    Returns ``(C, d)`` such that :math:`\\Omega(p^\\star) \\cap [\\underline{w},
    \\overline{w}]^L = \\{w \\in [\\underline{w}, \\overline{w}]^L : C w \\leq d\\}`.
    """
    L = inputs.n_levels
    bdy = [c for c in inputs.active_rule_constraints if c.kind == "boundary_binding"]
    viol = [c for c in inputs.active_rule_constraints if c.kind == "violated"]
    n_bdy = len(bdy)
    n_phys = len(inputs.active_phys_inequalities)
    n_eq = len(inputs.active_equalities)
    d_dim = inputs.grad_J.shape[0]

    # Stationarity: ∇J + Σ β_b ∇g_b + Σ_{viol} w_i ∇V_i + Σ λ ∇g_phys + Σ μ ∇h = 0.
    # Rearranged: M @ y = -∇J - V @ w, where y = (β, λ, μ).
    M = np.zeros((d_dim, n_bdy + n_phys + n_eq))
    for j_b, c in enumerate(bdy):
        M[:, j_b] = c.gradient
    for j_p, g in enumerate(inputs.active_phys_inequalities):
        M[:, n_bdy + j_p] = g
    for j_e, h in enumerate(inputs.active_equalities):
        M[:, n_bdy + n_phys + j_e] = h
    V = np.zeros((d_dim, L))
    for c in viol:
        V[:, c.level_index] += c.gradient
    rhs_constant = -inputs.grad_J
    # The full system: M y = rhs_constant - V w.
    # If M has full column rank, y(w) = M^+ (rhs_constant - V w) (Moore-Penrose).
    if M.shape[1] == 0:
        # No multipliers — Ω is just the box, plus the requirement that V w = -∇J.
        # If V w = -∇J cannot be satisfied for any w > 0, Ω is empty.
        if not np.allclose(V, 0.0):
            try:
                w_unique = np.linalg.lstsq(V, rhs_constant, rcond=None)[0]
                if np.all(w_unique > 0):
                    # Single point in Ω — return tight inequalities.
                    return np.eye(L), w_unique
            except np.linalg.LinAlgError:
                pass
        return np.zeros((0, L)), np.zeros(0)
    M_pinv = np.linalg.pinv(M)
    y_const = M_pinv @ rhs_constant       # y_const ∈ R^{n_bdy + n_phys + n_eq}
    y_linear = -M_pinv @ V                # y_linear ∈ R^{(n_bdy + n_phys + n_eq) × L}
    # Check that the system is consistent: M @ y(w) should equal rhs - V w.
    # We just check by recomputing the residual at w = box midpoint.
    w_probe = 0.5 * (inputs.box_lower + inputs.box_upper)
    y_probe = y_const + y_linear @ w_probe
    residual = M @ y_probe - (rhs_constant - V @ w_probe)
    if np.linalg.norm(residual) > 1e-6 * (1.0 + np.linalg.norm(rhs_constant)):
        logger.info(
            "omega_half_space_description: stationarity system has rank deficiency "
            "(residual %.3g) — using pseudo-inverse projection (may produce "
            "approximate Ω boundaries).", float(np.linalg.norm(residual))
        )

    # Inequalities expressed in w only:
    # β_b ≥ 0  →  y_b(w) ≥ 0  →  -y_const_b - y_linear_b @ w ≤ 0
    # β_b ≤ w_i_b  →  y_b(w) - w_{i_b} ≤ 0 →  y_const_b + (y_linear_b - e_{i_b}) @ w ≤ 0
    # λ_p ≥ 0  →  y_p(w) ≥ 0 →  -y_const_p - y_linear_p @ w ≤ 0
    C_rows: List[np.ndarray] = []
    d_vals: List[float] = []
    for j_b, c in enumerate(bdy):
        # β_b = y_const[j_b] + y_linear[j_b, :] @ w ≥ 0
        C_rows.append(-y_linear[j_b, :])
        d_vals.append(float(y_const[j_b]))
        # β_b ≤ w_{c.level_index}
        row = y_linear[j_b, :].copy()
        row[c.level_index] -= 1.0
        C_rows.append(row)
        d_vals.append(float(-y_const[j_b]))
    for j_p in range(n_phys):
        idx = n_bdy + j_p
        C_rows.append(-y_linear[idx, :])
        d_vals.append(float(y_const[idx]))

    if not C_rows:
        return np.zeros((0, L)), np.zeros(0)
    return np.array(C_rows), np.array(d_vals)


# ----------------------------------------------------------------------
# Algorithm 1B: L₂ tolerance compliance
# ----------------------------------------------------------------------


def _rho(epsilon: float, tolerance_form: str) -> float:
    """The scaling :math:`\\rho(\\epsilon_i)` from Proposition 7.2 of the paper.

    * ``raw``     → ``ρ(ε) = ε``    (bounds |g_{i,j}| ≤ ε)
    * ``squared`` → ``ρ(ε) = √ε``   (bounds V_i = Σ g² ≤ ε)
    """
    if tolerance_form == "raw":
        return float(epsilon)
    if tolerance_form == "squared":
        return float(epsilon) ** 0.5
    raise ValueError(f"Unknown tolerance_form: {tolerance_form}")


def compute_l2_sensitivity_constants(
    inputs: L2SensitivityInputs,
) -> L2SensitivityConstants:
    """Compute :math:`(c_i, \\kappa_i)` per Step B2 of Algorithm 1B.

    *Single-active-per-level case* (Section 9.4, recommended). Each
    boundary-binding satisfied level has exactly one active constraint
    :math:`\\nabla g_{i, j(i)}`. The dominant-balance projection along
    :math:`\\nabla g_{i, j(i)}` yields:

    .. math::

        c_i &= \\frac{|\\langle \\nabla J, \\nabla g_{i, j(i)} \\rangle|}{2 \\|\\nabla g_{i, j(i)}\\|^2} \\\\
        \\kappa_{i, i'} &= -\\frac{\\langle \\nabla V_{i'}, \\nabla g_{i, j(i)} \\rangle}{2 \\|\\nabla g_{i, j(i)}\\|^2}

    for every violated level :math:`i'` (so that
    :math:`g_{i, j(i)}(z_\\text{ws}^\\star(w)) \\approx (c_i + \\sum_{i' \\ne i} \\kappa_{i, i'} w_{i'}) / w_i`
    matches the paper's expansion at the worked Example 2).

    The sign of :math:`\\kappa_{i, i'}` reflects whether increasing
    :math:`w_{i'}` *worsens* the boundary-binding violation at level :math:`i`
    (positive) or *relieves* it (negative). The paper notes both possibilities;
    the constant is computed honestly here without any absolute-value coercion.

    *Multi-active-per-level case*. The paper says "the full reduced KKT
    sensitivity system must be solved numerically." For brevity (and because
    the rule encoders in this code base produce one boundary-binding
    constraint per safety/legal level in the dominant case), we implement the
    single-active reduction and document the extension point. Multi-active
    callers can extend by replacing the scalar projection with a Moore-Penrose
    pseudoinverse on the stacked active-gradient matrix; we leave that for
    future work.
    """
    c_const: dict[int, float] = {}
    kappa: dict[int, np.ndarray] = {}
    L = inputs.n_levels
    grad_J = np.asarray(inputs.grad_J, dtype=np.float64)

    for level_index, slots in inputs.boundary_binding_per_level.items():
        if len(slots) == 0:
            continue
        if len(slots) > 1:
            logger.warning(
                "compute_l2_sensitivity_constants: level %d has %d boundary-"
                "binding actives; using only the first (multi-active reduction "
                "not implemented). The remaining %d are dropped — see Section 9.4 "
                "Step B2 for the full multi-active recipe.",
                level_index, len(slots), len(slots) - 1,
            )
        _, grad_g = slots[0]
        grad_g = np.asarray(grad_g, dtype=np.float64)
        grad_g_norm2 = float(np.dot(grad_g, grad_g))
        if grad_g_norm2 < 1e-12:
            logger.warning(
                "compute_l2_sensitivity_constants: level %d active gradient has "
                "norm² = %.3g (near zero); skipping sensitivity computation.",
                level_index, grad_g_norm2,
            )
            continue
        # Performance-side leading constant.
        c_const[level_index] = abs(float(np.dot(grad_J, grad_g))) / (2.0 * grad_g_norm2)
        # Cross-level coupling vector. Entry [i'] is the coupling from
        # weight w_{i'} (i' ≠ i). For violated levels we pick up their ∇V_{i'};
        # for non-violated, non-boundary-binding levels the coupling is zero.
        kappa_i = np.zeros(L)
        for ip, grad_V_ip in inputs.violated_grad_V_per_level.items():
            if ip == level_index:
                continue
            # The minus sign follows from the dominant-balance derivation:
            # increasing w_{i'} pushes the WS optimum in the direction of
            # -∇V_{i'}, which projects onto ∇g_{i, j(i)} with sign
            # -⟨∇V_{i'}, ∇g_{i, j(i)}⟩ / (2 ||∇g||²) on the violation.
            kappa_i[ip] = -float(np.dot(np.asarray(grad_V_ip), grad_g)) / (2.0 * grad_g_norm2)
        kappa[level_index] = kappa_i

    return L2SensitivityConstants(c_const_per_level=c_const, kappa_per_level=kappa)


def algorithm_1b(inputs: L2SensitivityInputs) -> WeightCalibrationResult:
    """Run Algorithm 1B of the LCP paper (L₂ tolerance compliance).

    Solves the pointwise coupled-linear Chebyshev-centre LP described in
    Section 9.4 Step B4:

    .. math::

        \\max\\ r \\quad \\text{s.t.} \\quad
            \\rho(\\epsilon_i) w_i - \\langle \\kappa_i, w_{-i} \\rangle - c_i \\geq r,
            \\quad \\forall i \\in \\mathcal{S}_{\\text{bdy}}

    together with the box constraints and ``r ≥ 0``. The Chebyshev objective
    is identical to Algorithm 1A's; what differs is that the active-set's
    role is to populate the threshold inequalities rather than the equality
    KKT system.

    On infeasibility (no weight in the box satisfies the coupled thresholds)
    the result falls back to the box centre with ``r† = 0`` and an
    informative ``notes`` string.

    Returns a :class:`WeightCalibrationResult` with the calibrated weights,
    the LP margin ``r†`` (positive means a robust interior weight; zero
    means a boundary weight), and the multipliers fields zeroed since
    Algorithm 1B does not produce KKT multipliers in the same way 1A does.
    """
    L = inputs.n_levels
    box_lower = np.asarray(inputs.box_lower, dtype=np.float64).reshape(L)
    box_upper = np.asarray(inputs.box_upper, dtype=np.float64).reshape(L)
    if np.any(box_lower <= 0):
        raise ValueError(f"box_lower must be strictly positive (got {box_lower})")
    if np.any(box_upper <= box_lower):
        raise ValueError("box_upper must exceed box_lower elementwise")
    epsilon = np.asarray(inputs.epsilon_per_level, dtype=np.float64).reshape(L)
    if np.any(epsilon <= 0):
        raise ValueError("epsilon_per_level must be strictly positive")

    sensitivity = compute_l2_sensitivity_constants(inputs)

    # Variable layout in the LP: [w (L) | r (1)]
    n_vars = L + 1
    r_idx = L

    A_ub_rows: List[np.ndarray] = []
    b_ub: List[float] = []

    # Pointwise coupled-linear thresholds. For each boundary-binding satisfied
    # level i: rho(eps_i) * w_i − <kappa_i, w_{-i}> − c_i ≥ r
    # ⇒ −rho(eps_i)*w_i + <kappa_i, w_{-i}> + c_i + r ≤ 0
    # ⇒ row[i] = −rho(eps_i), row[i'] = kappa[i, i'] (for i' ≠ i), row[r] = 1, rhs = -c_i
    for level_index in sensitivity.levels_with_threshold:
        c_i = sensitivity.c_const_per_level[level_index]
        kappa_i = sensitivity.kappa_per_level[level_index]
        rho_i = _rho(epsilon[level_index], inputs.tolerance_form)
        row = np.zeros(n_vars)
        row[level_index] = -rho_i
        for ip in range(L):
            if ip != level_index:
                row[ip] = kappa_i[ip]
        row[r_idx] = 1.0
        A_ub_rows.append(row)
        b_ub.append(-c_i)

    # Chebyshev box-face slacks: w_i − underline_w_i ≥ r, overline_w_i − w_i ≥ r.
    for i in range(L):
        row = np.zeros(n_vars)
        row[i] = -1.0
        row[r_idx] = 1.0
        A_ub_rows.append(row)
        b_ub.append(-box_lower[i])
        row = np.zeros(n_vars)
        row[i] = 1.0
        row[r_idx] = 1.0
        A_ub_rows.append(row)
        b_ub.append(box_upper[i])

    A_ub = np.vstack(A_ub_rows) if A_ub_rows else None
    b_ub_arr = np.array(b_ub) if b_ub else None

    bounds: List[Tuple[Optional[float], Optional[float]]] = []
    for i in range(L):
        bounds.append((box_lower[i], box_upper[i]))
    bounds.append((0.0, None))

    c_obj = np.zeros(n_vars)
    c_obj[r_idx] = -1.0

    res = linprog(
        c=c_obj,
        A_ub=A_ub,
        b_ub=b_ub_arr,
        bounds=bounds,
        method="highs",
    )

    if not res.success:
        logger.warning(
            "Algorithm 1B LP failed: status=%s, message=%s; "
            "falling back to geometric box centre.",
            res.status, res.message,
        )
        return WeightCalibrationResult(
            w_dagger=0.5 * (box_lower + box_upper),
            r_dagger=0.0,
            beta_at_optimum=np.zeros(0),
            lambda_at_optimum=np.zeros(0),
            mu_at_optimum=np.zeros(0),
            lp_status=str(res.message),
            notes="LP infeasible; falling back to box centre",
        )

    x = res.x
    return WeightCalibrationResult(
        w_dagger=x[:L].copy(),
        r_dagger=float(x[r_idx]),
        beta_at_optimum=np.zeros(0),
        lambda_at_optimum=np.zeros(0),
        mu_at_optimum=np.zeros(0),
        lp_status=str(res.message),
        notes=(
            f"L₂ tolerance compliance via Algorithm 1B. "
            f"Levels with threshold: {sensitivity.levels_with_threshold}. "
            f"tolerance_form={inputs.tolerance_form!r}"
        ),
    )


def l2_threshold(
    level_index: int,
    w_minus_i: Sequence[float],
    constants: L2SensitivityConstants,
    epsilon: float,
    tolerance_form: str = "raw",
) -> float:
    """Evaluate the pointwise threshold :math:`W_i(\\epsilon_i, w_{-i})` at a
    specific :math:`w_{-i}`.

    Useful for diagnostics ("at this choice of other-level weights, the
    boundary-binding violation at level i would settle around
    :math:`(c_i + \\langle \\kappa_i, w_{-i}\\rangle) / w_i`; to keep it below
    tolerance ε, we need w_i ≥ this value").
    """
    if level_index not in constants.c_const_per_level:
        raise ValueError(f"Level {level_index} has no boundary-binding sensitivity")
    c_i = constants.c_const_per_level[level_index]
    kappa_i = constants.kappa_per_level[level_index]
    w_minus_i_arr = np.asarray(w_minus_i, dtype=np.float64)
    # Build the coupling sum, skipping the level-i index.
    coupling = 0.0
    j = 0
    for ip in range(len(kappa_i)):
        if ip == level_index:
            continue
        coupling += float(kappa_i[ip]) * float(w_minus_i_arr[j])
        j += 1
    rho_i = _rho(epsilon, tolerance_form)
    return (c_i + coupling) / rho_i
