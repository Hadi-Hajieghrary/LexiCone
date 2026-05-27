"""Abstract convex priority-ordered MPC problem (v10_2 Section 2).

The :class:`ConvexPriorityProblem` is the central data structure consumed by
every algorithm in this package. It captures:

- The decision-variable trajectory ``z = (x_{0:N}, u_{0:N-1}) in R^d`` with
  ``d = (N+1)*nx + N*nu``.
- Affine dynamics ``x_{k+1} = A_k x_k + B_k u_k + c_k`` for each timestep
  (v10_2 standing assumption ``A1`` for the convex setting).
- Box constraints on x and u (the polyhedral feasibility set ``Z``).
- A convex performance objective ``J(z)`` (linear, affine, or convex quadratic).
- Priority-partitioned constraints ``g_{i,j}(z) <= 0`` for ``i = 1, ..., L``
  and ``j = 1, ..., m_i``. The framework is L_1 when ``penalty_form="l1"``
  (g's are affine; v10_2 Section 5) or L_2 when ``penalty_form="l2"`` (g's
  may be convex quadratic, v10_2 Section 6).

The performance objective J and the level-violation functionals V_i are
derived from the per-step constraint data; no caller is required to provide
gradients or KKT structure — the package extracts what it needs from this
problem description.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional, Sequence, Tuple

import numpy as np


PenaltyForm = Literal["l1", "l2"]


@dataclass(frozen=True)
class AffineDynamics:
    """Affine discrete-time dynamics for one timestep.

    Represents ``x_{k+1} = A x_k + B u_k + c`` with shapes ``A: (nx, nx)``,
    ``B: (nx, nu)``, ``c: (nx,)``. The triple ``(A, B, c)`` is the data
    produced by linearising any nonlinear dynamics about a warm-start at the
    timestep; for inherently linear plants it is the dynamics matrix directly.
    """
    A: np.ndarray
    B: np.ndarray
    c: np.ndarray

    def __post_init__(self) -> None:
        if self.A.ndim != 2 or self.A.shape[0] != self.A.shape[1]:
            raise ValueError(f"A must be square (nx, nx); got {self.A.shape}")
        nx = self.A.shape[0]
        if self.B.ndim != 2 or self.B.shape[0] != nx:
            raise ValueError(f"B must be (nx, nu) with nx={nx}; got {self.B.shape}")
        if self.c.shape != (nx,):
            raise ValueError(f"c must be (nx,) with nx={nx}; got {self.c.shape}")


@dataclass
class BoxBounds:
    """Component-wise lower/upper bounds.

    ``lo[i] <= var[i] <= hi[i]`` for each component. Use ``-np.inf`` /
    ``+np.inf`` to indicate unbounded entries.
    """
    lo: np.ndarray
    hi: np.ndarray

    def __post_init__(self) -> None:
        if self.lo.shape != self.hi.shape:
            raise ValueError(f"lo {self.lo.shape} and hi {self.hi.shape} must match")
        if np.any(self.lo > self.hi):
            raise ValueError("lo must be element-wise <= hi")


@dataclass(frozen=True)
class AffineConstraint:
    """A single affine constraint ``a^T x + b^T u + e <= 0`` over one
    timestep's (x_k, u_k) pair.

    For a constraint that only involves the state (e.g. drivable area), set
    ``b = np.zeros(nu)``. For a constraint that only involves control (e.g.
    actuator effort), set ``a = np.zeros(nx)``. Use ``e > 0`` to encode an
    upper bound, ``e < 0`` to require a strict margin.
    """
    a: np.ndarray   # (nx,)
    b: np.ndarray   # (nu,)
    e: float
    timestep: int   # which step k in {0, 1, ..., N-1} this applies to


@dataclass
class LevelSpec:
    """One priority level: a name, a list of per-tick affine constraints, and
    the per-level epsilon tolerance for L_2 tolerance compliance.

    The constraints in ``constraints`` are evaluated across all timesteps of
    the horizon; the level's integrated violation is

    .. math::
        V_i(z) = \\sum_{k} \\sum_{j} \\rho(g_{i,j,k}(z))

    with rho = max(0, .) for L_1 and rho = max(0, .)^2 for L_2.
    """
    name: str
    constraints: List[AffineConstraint] = field(default_factory=list)
    epsilon_tolerance: float = 1e-4   # L_2 only (v10_2 Section 7.2)


@dataclass
class PerformanceObjective:
    """The performance objective J(z).

    Two forms supported per v10_2 Section 2:

    - ``form = "linear"``: ``J(z) = q^T z`` (Setting (P) of Section 5).
    - ``form = "quadratic"``: ``J(z) = 0.5 z^T Q z + q^T z`` with ``Q``
      positive semi-definite (Setting (Q) of Section 6).

    The vectors and matrices live in the full decision-variable space
    ``z in R^{(N+1)*nx + N*nu}`` packed row-major: x_0, x_1, ..., x_N,
    u_0, u_1, ..., u_{N-1}.
    """
    form: Literal["linear", "quadratic"]
    q: np.ndarray
    Q: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if self.form == "quadratic" and self.Q is None:
            raise ValueError("quadratic form requires Q")
        if self.form == "linear" and self.Q is not None:
            raise ValueError("linear form must not specify Q")


@dataclass
class ConvexPriorityProblem:
    """The full v10_2 problem instance.

    Parameters
    ----------
    nx, nu
        State and control dimensions.
    horizon
        Number of control steps ``N``. The state trajectory has ``N+1``
        timesteps (indices 0..N); the control trajectory has ``N``
        (indices 0..N-1).
    dynamics
        Length-``N`` sequence of :class:`AffineDynamics` (one per step).
        For inherently linear plants this is just the linear dynamics
        repeated; for nonlinear plants linearised about a warm-start it is
        per-step.
    x0
        Initial state, shape ``(nx,)``.
    x_bounds, u_bounds
        Box bounds on state and control vectors.
    levels
        Priority-ordered list of :class:`LevelSpec`, ordered from highest
        priority (level 0 = most important) to lowest. Equivalent to v10_2's
        ``i = 1, ..., L``.
    performance
        The performance objective ``J(z)``.
    penalty_form
        ``"l1"`` for the polyhedral exact-equivalence regime (v10_2 §5),
        ``"l2"`` for the convex quadratic tolerance-compliance regime
        (v10_2 §6).
    """
    nx: int
    nu: int
    horizon: int
    dynamics: Sequence[AffineDynamics]
    x0: np.ndarray
    x_bounds: BoxBounds
    u_bounds: BoxBounds
    levels: List[LevelSpec]
    performance: PerformanceObjective
    penalty_form: PenaltyForm = "l1"

    def __post_init__(self) -> None:
        if len(self.dynamics) != self.horizon:
            raise ValueError(
                f"dynamics length {len(self.dynamics)} must match horizon {self.horizon}"
            )
        for k, dyn in enumerate(self.dynamics):
            if dyn.A.shape[0] != self.nx:
                raise ValueError(f"dynamics[{k}].A nx mismatch")
            if dyn.B.shape[1] != self.nu:
                raise ValueError(f"dynamics[{k}].B nu mismatch")
        if self.x0.shape != (self.nx,):
            raise ValueError(f"x0 must be (nx,); got {self.x0.shape}")
        if self.x_bounds.lo.shape != (self.nx,):
            raise ValueError(f"x_bounds shape mismatch with nx={self.nx}")
        if self.u_bounds.lo.shape != (self.nu,):
            raise ValueError(f"u_bounds shape mismatch with nu={self.nu}")
        # Performance vector lives in full z-space.
        d = (self.horizon + 1) * self.nx + self.horizon * self.nu
        if self.performance.q.shape != (d,):
            raise ValueError(
                f"performance.q must be ({d},); got {self.performance.q.shape}"
            )
        if self.performance.Q is not None and self.performance.Q.shape != (d, d):
            raise ValueError(
                f"performance.Q must be ({d}, {d}); got {self.performance.Q.shape}"
            )
        # Validate per-level constraint timesteps.
        for i, lvl in enumerate(self.levels):
            for j, c in enumerate(lvl.constraints):
                if not (0 <= c.timestep < self.horizon):
                    raise ValueError(
                        f"levels[{i}].constraints[{j}].timestep={c.timestep} "
                        f"out of range [0, {self.horizon})"
                    )
                if c.a.shape != (self.nx,):
                    raise ValueError(f"levels[{i}].constraints[{j}].a shape mismatch")
                if c.b.shape != (self.nu,):
                    raise ValueError(f"levels[{i}].constraints[{j}].b shape mismatch")

    @property
    def n_levels(self) -> int:
        """Number of priority levels L."""
        return len(self.levels)

    @property
    def n_decision_vars(self) -> int:
        """Total dimension of z = (x_{0:N}, u_{0:N-1})."""
        return (self.horizon + 1) * self.nx + self.horizon * self.nu

    def x_index(self, k: int) -> slice:
        """Index range of state x_k inside z."""
        if not (0 <= k <= self.horizon):
            raise IndexError(f"state index {k} out of range [0, {self.horizon}]")
        start = k * self.nx
        return slice(start, start + self.nx)

    def u_index(self, k: int) -> slice:
        """Index range of control u_k inside z."""
        if not (0 <= k < self.horizon):
            raise IndexError(f"control index {k} out of range [0, {self.horizon})")
        start = (self.horizon + 1) * self.nx + k * self.nu
        return slice(start, start + self.nu)
