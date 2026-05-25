"""Per-tick encoders that turn observer rules into convex LCP MPC constraints.

The LCP framework needs every rule to be expressible as a *linear* inequality
``a^T x_k + b^T u_k + e <= t_{i,j,k}`` in the decision variables ``(X, U)``,
with ``t >= 0`` the epigraph slack. Many of the observer's per-rule
``violation()`` functions are quadratic or piecewise-defined; we **linearise
them at the warm-start trajectory** every tick, which is consistent with the
SLP/SQP scheme in :mod:`.slp_linearisation` (Section 12.3 of the paper
acknowledges this as the formal route to recover convexity).

Per-tick applicability filter
-----------------------------

Each :class:`RuleEncoder` has two methods:

- ``applies_to_horizon(ctx) -> bool`` — cheap gating predicate. If False, the
  rule contributes zero constraints this tick (every slot is filled with
  :func:`~.lcp_mpc.make_inactive_constraint` so the MPC ignores it).
- ``encode(ctx) -> List[List[LinearisedRuleConstraint]]`` — returns
  ``slots_per_step`` constraints per step. The encoder is responsible for
  padding with inactive slots when fewer constraints actually apply.

The :class:`RuleSet` aggregates all encoders for one priority level and
produces the per-level :class:`~.lcp_mpc.LCPRulePack` the MPC consumes.

Rule status in this session
---------------------------

**Fully implemented** (7 rules):

- ``9r0`` / ``10r0`` (combined as :class:`CollisionRule` — vehicle + VRU)
- ``7r0`` (:class:`DrivableBoundaryRule`)
- ``7r5`` (:class:`SidewalkDriveRule`)
- ``3r0`` (:class:`SpeedLimitRule`)
- ``3r3`` (:class:`SafeHeadwayRule`)
- ``0r2`` (:class:`LongitudinalComfortRule`)
- ``1r11`` (:class:`LateralAccelerationRule`)

**Stubbed** with documented linearisation but ``applies_to_horizon`` returning
False until the next session lands the map-data plumbing they need (9 rules):

- ``7r2`` opposing lane, ``7r3`` one-way directionality
- ``7r1`` traffic light at stop line
- ``7r4`` stop-in-crosswalk
- ``10r5`` bike-lane encroachment (NuPlan doesn't expose this layer; will
  remain a no-op on the mini split)
- ``3r5`` lateral clearance, ``3r6`` lane intrusion
- ``0r3`` lateral comfort (jerk)
- :class:`RouteAdherenceRule` is observer-only — already enforced by the
  global planner's reference, so it lives in the deferred set.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .bicycle_model import NU, NX
from .lcp_mpc import LCPRulePack, LinearisedRuleConstraint, make_inactive_constraint
from .map_lifter import LocalLane, LocalPolygon, MapHorizonView

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Context passed to every encoder
# ----------------------------------------------------------------------


@dataclass
class AgentSlot:
    """Minimal per-agent info needed by collision / headway / clearance rules.

    All coordinates are in ego-local frame at the current tick.
    """

    track_id: str
    x: float
    y: float
    vx: float                  # local-frame velocity (x = forward)
    vy: float                  # local-frame velocity (y = leftward)
    length: float
    width: float
    is_vru: bool               # pedestrian or cyclist


@dataclass
class EncoderContext:
    """Everything an encoder needs at one MPC tick, in ego-local frame.

    The orchestrator (:class:`~.two_level_planner.TwoLevelMPCPlanner`) builds
    this once per tick and passes it to every encoder.
    """

    horizon_steps: int
    dt_s: float
    warm_start_X: np.ndarray   # (NX, N+1) reference trajectory for linearisation
    warm_start_U: np.ndarray   # (NU, N)
    Xref_local: np.ndarray     # (NX, N+1) reference for the efficiency objective
    agents_local: Tuple[AgentSlot, ...]
    map_view: Optional[MapHorizonView]
    desired_speed_mps: float
    ego_radius_m: float
    wheel_base_m: float


# ----------------------------------------------------------------------
# Base class
# ----------------------------------------------------------------------


class RuleEncoder:
    """Abstract base for one rule's per-tick constraint generator.

    Subclasses set ``rule_id``, ``priority_level``, ``slots_per_step`` and
    implement :meth:`applies_to_horizon` and :meth:`encode`.

    Subclass contract:

    - ``encode`` must return a list of length ``ctx.horizon_steps``. Each
      element is a list of exactly ``slots_per_step`` :class:`LinearisedRuleConstraint`
      objects. Inactive slots use :func:`make_inactive_constraint`.
    - When ``applies_to_horizon`` returns False, the framework supplies an
      all-inactive constraint list automatically; ``encode`` is *not* called.
    """

    rule_id: str = "??"          # observer rule ID this maps to
    priority_level: int = -1     # 1, 2, or 3 in the LCP MPC's level indexing
    slots_per_step: int = 0      # number of constraint slots this rule reserves

    def applies_to_horizon(self, ctx: EncoderContext) -> bool:
        raise NotImplementedError

    def encode(self, ctx: EncoderContext) -> List[List[LinearisedRuleConstraint]]:
        raise NotImplementedError

    def all_inactive(self, ctx: EncoderContext) -> List[List[LinearisedRuleConstraint]]:
        """Convenience: build an all-inactive per-step constraint list."""
        return [
            [make_inactive_constraint() for _ in range(self.slots_per_step)]
            for _ in range(ctx.horizon_steps)
        ]


# ----------------------------------------------------------------------
# RuleSet aggregator
# ----------------------------------------------------------------------


@dataclass
class RuleSet:
    """All rule encoders for one priority level, plus the slot budget.

    The MPC's ``LCPLevelSpec.slots_per_step`` for level ``i`` must equal the
    sum of ``slots_per_step`` across the encoders in ``encoders_for_level[i]``.
    """

    levels: List[List[RuleEncoder]] = field(default_factory=list)

    def slots_per_step_per_level(self) -> List[int]:
        return [sum(enc.slots_per_step for enc in lvl) for lvl in self.levels]

    def encode_all(self, ctx: EncoderContext) -> LCPRulePack:
        """Build an :class:`LCPRulePack` from every encoder's output.

        Concatenates each encoder's per-step contribution side-by-side so the
        slot order within a level is stable (depends on encoder order in
        ``levels``). The MPC's parameter slot ``j`` always corresponds to the
        same encoder's constraint.
        """
        constraints_by_level: List[List[List[LinearisedRuleConstraint]]] = []
        for level_encoders in self.levels:
            # Per step, gather contributions from all encoders at this level.
            per_step_lists: List[List[LinearisedRuleConstraint]] = [
                [] for _ in range(ctx.horizon_steps)
            ]
            for enc in level_encoders:
                contrib = enc.encode(ctx) if enc.applies_to_horizon(ctx) else enc.all_inactive(ctx)
                if len(contrib) != ctx.horizon_steps:
                    raise ValueError(
                        f"Encoder {enc.rule_id} produced {len(contrib)} step entries, "
                        f"expected {ctx.horizon_steps}"
                    )
                for k in range(ctx.horizon_steps):
                    if len(contrib[k]) != enc.slots_per_step:
                        raise ValueError(
                            f"Encoder {enc.rule_id} produced {len(contrib[k])} slots "
                            f"at step {k}, expected {enc.slots_per_step}"
                        )
                    per_step_lists[k].extend(contrib[k])
            constraints_by_level.append(per_step_lists)
        return LCPRulePack(constraints_by_level=constraints_by_level)


# ----------------------------------------------------------------------
# Level 1 — Safety
# ----------------------------------------------------------------------


class CollisionRule(RuleEncoder):
    """Combined 9r0 (vehicle) + 10r0 (VRU) collision avoidance.

    Per-step quadratic constraint, linearised at the warm-start:

    .. math::

        r_{\\min,j}^2 - (x_k - o_{j,x})^2 - (y_k - o_{j,y})^2 \\leq 0

    Linearise around :math:`(\\bar x_k, \\bar y_k)`:

    .. math::

        f(\\bar x_k, \\bar y_k) - 2(\\bar x_k - o_{j,x})(x_k - \\bar x_k)
                                - 2(\\bar y_k - o_{j,y})(y_k - \\bar y_k) \\leq 0

    Rearranging into :math:`a^T x_k + e \\leq 0` form:

    .. math::

        a = (-2(\\bar x_k - o_{j,x}), -2(\\bar y_k - o_{j,y}), 0, 0), \\quad b = 0,
        \\quad e = f(\\bar x_k, \\bar y_k) + 2(\\bar x_k - o_{j,x})\\bar x_k + 2(\\bar y_k - o_{j,y})\\bar y_k.

    VRU agents (pedestrians, cyclists) have ``r_min`` inflated by
    ``vru_inflate_m`` to bias the planner toward larger clearance, matching
    the observer's behaviour.

    ``slots_per_step`` reserves a fixed number of agent slots (default 8);
    unused slots are inactive.
    """

    rule_id = "9r0_10r0"
    priority_level = 1

    def __init__(
        self,
        slots_per_step: int = 8,
        collision_buffer_m: float = 0.4,
        vru_inflate_m: float = 0.4,
        max_distance_m: float = 40.0,
    ) -> None:
        self.slots_per_step = slots_per_step
        self.collision_buffer_m = collision_buffer_m
        self.vru_inflate_m = vru_inflate_m
        self.max_distance_m = max_distance_m

    def applies_to_horizon(self, ctx: EncoderContext) -> bool:
        if not ctx.agents_local:
            return False
        return any(
            math.hypot(a.x, a.y) < self.max_distance_m for a in ctx.agents_local
        )

    def encode(self, ctx: EncoderContext) -> List[List[LinearisedRuleConstraint]]:
        # Rank agents by distance and pick the closest ``slots_per_step``.
        scored = sorted(
            ctx.agents_local, key=lambda a: (a.x ** 2 + a.y ** 2)
        )[: self.slots_per_step]

        per_step: List[List[LinearisedRuleConstraint]] = []
        N = ctx.horizon_steps
        for k in range(N):
            x_bar = ctx.warm_start_X[0, k]
            y_bar = ctx.warm_start_X[1, k]
            slots: List[LinearisedRuleConstraint] = []
            for ag in scored:
                # Predicted agent position at step k (constant-velocity model).
                ox = ag.x + ag.vx * ctx.dt_s * k
                oy = ag.y + ag.vy * ctx.dt_s * k
                # Effective radius (sum of bounding-circle radii + buffer).
                r_agent = 0.5 * math.hypot(ag.length, ag.width)
                r_ego = ctx.ego_radius_m
                r_min = r_agent + r_ego + self.collision_buffer_m
                if ag.is_vru:
                    r_min += self.vru_inflate_m
                dx = x_bar - ox
                dy = y_bar - oy
                f_bar = r_min ** 2 - (dx * dx + dy * dy)
                # Linearised: f_bar - 2*dx*(x - x_bar) - 2*dy*(y - y_bar) <= 0
                # = -2*dx*x - 2*dy*y + (f_bar + 2*dx*x_bar + 2*dy*y_bar) <= 0
                a = np.array([-2.0 * dx, -2.0 * dy, 0.0, 0.0])
                e = f_bar + 2.0 * dx * x_bar + 2.0 * dy * y_bar
                slots.append(
                    LinearisedRuleConstraint(a=a, b=np.zeros(NU), e=float(e), mask=1.0)
                )
            # Pad to fixed slot count.
            while len(slots) < self.slots_per_step:
                slots.append(make_inactive_constraint())
            per_step.append(slots)
        return per_step


class DrivableBoundaryRule(RuleEncoder):
    """7r0 — Stay inside the drivable surface.

    ⚠️ **Do not use this encoder.** This implementation is *structurally
    wrong*: the drivable surface is the **union** of lane polygons (a
    non-convex set), but a per-tick L₁ MPC cannot encode unions directly.
    The naive "use the K nearest half-planes across every polygon"
    formulation in this class enforces the **intersection** of the lane
    polygons instead — which is generically empty (lanes are disjoint), so
    the safety slacks blow up and the trajectory wanders away from any lane
    centre trying to minimise the aggregate violation. Use
    :class:`LaneCorridorRule` instead, which encodes the convex-locally
    "stay within ±half-width of the route centreline" constraint via two
    linear half-planes per step.

    The class is kept for documentation and to preserve a slot for a future
    proper drivable-surface encoder (per-tick active-lane selection or
    raster-based signed-distance). Today its ``applies_to_horizon`` returns
    False so it contributes no constraints.
    """

    rule_id = "7r0_deprecated"
    priority_level = 1

    def __init__(self, slots_per_step: int = 4, ego_buffer_m: float = 0.3) -> None:
        self.slots_per_step = slots_per_step
        self.ego_buffer_m = ego_buffer_m

    def applies_to_horizon(self, ctx: EncoderContext) -> bool:
        return False  # disabled — see class docstring

    def encode(self, ctx: EncoderContext) -> List[List[LinearisedRuleConstraint]]:
        return self.all_inactive(ctx)


class LaneCorridorRule(RuleEncoder):
    """7r0 — Stay inside a lateral corridor around the route centreline.

    Reformulation of the drivable-boundary rule as a *corridor* around the
    global planner's reference. At each MPC step, the constraint is:

    .. math::

        |(p_k - p_{\\text{ref},k}) \\cdot n_{\\text{ref},k}^{\\perp}|
        \\leq \\frac{w_{\\text{lane}}}{2} - r_{\\text{ego}}

    where :math:`n_{\\text{ref},k}^{\\perp}` is the left-pointing unit normal
    to the reference heading at step k. The constraint splits into two
    linear half-plane inequalities (one per side), so the encoder produces
    two slots per step. The geometry is exact (no Taylor expansion needed
    since the constraint is already linear in (x, y) once the reference
    pose is fixed by the orchestrator).

    The half-width parameter ``half_width_m`` is the maximum allowed
    *lateral* deviation of the ego rear-axle position from the reference
    centreline. Set it to ``(lane_width / 2) - (ego_half_width)`` to keep
    the entire ego body inside the lane — for a typical 3.5 m lane and a
    1.85 m wide ego, that's about 0.8 m.

    Unlike :class:`DrivableBoundaryRule`, this encoder does **not** require
    a per-tick map view — it reads the reference pose from
    :attr:`EncoderContext.Xref_local` directly. The route's lane structure
    is encoded implicitly through the centreline (the global planner walks
    the lane graph).
    """

    rule_id = "7r0"
    priority_level = 1

    def __init__(self, slots_per_step: int = 2, half_width_m: float = 1.0) -> None:
        self.slots_per_step = slots_per_step
        self.half_width_m = half_width_m

    def applies_to_horizon(self, ctx: EncoderContext) -> bool:
        # Always applies — the reference is always present.
        return True

    def encode(self, ctx: EncoderContext) -> List[List[LinearisedRuleConstraint]]:
        N = ctx.horizon_steps
        per_step: List[List[LinearisedRuleConstraint]] = []
        for k in range(N):
            x_ref = float(ctx.Xref_local[0, k])
            y_ref = float(ctx.Xref_local[1, k])
            psi_ref = float(ctx.Xref_local[2, k])
            # Left-pointing unit normal to the reference heading.
            nl_x = -math.sin(psi_ref)
            nl_y = math.cos(psi_ref)
            # Lateral offset of ego from reference centreline:
            # offset = (x - x_ref) * nl_x + (y - y_ref) * nl_y
            #        = nl_x * x + nl_y * y - (nl_x * x_ref + nl_y * y_ref)
            ref_proj = nl_x * x_ref + nl_y * y_ref

            # Constraint 1 (left bound): offset - half_width <= t
            #   nl_x*x + nl_y*y - ref_proj - half_width <= t
            slots = [
                LinearisedRuleConstraint(
                    a=np.array([nl_x, nl_y, 0.0, 0.0]),
                    b=np.zeros(NU),
                    e=float(-ref_proj - self.half_width_m),
                    mask=1.0,
                ),
                # Constraint 2 (right bound): -offset - half_width <= t
                #   -nl_x*x - nl_y*y + ref_proj - half_width <= t
                LinearisedRuleConstraint(
                    a=np.array([-nl_x, -nl_y, 0.0, 0.0]),
                    b=np.zeros(NU),
                    e=float(ref_proj - self.half_width_m),
                    mask=1.0,
                ),
            ]
            # Pad to slot count (typically 2 — left + right — but the API
            # allows reserving more for symmetry with other rules).
            while len(slots) < self.slots_per_step:
                slots.append(make_inactive_constraint())
            per_step.append(slots[: self.slots_per_step])
        return per_step


class OpposingLaneRule(RuleEncoder):
    """7r2 — Stay out of any nearby lane whose heading is opposite to ego's.

    Per step ``k``, for each lane in ``ctx.map_view.{nonroute_lanes,
    lane_connectors}``, sample the lane heading at its centerline nearest
    the warm-start ``(x̄_k, ȳ_k)``. If that heading differs from the
    ego's warm-start heading by ``≥ π − tol``, classify the lane as
    *oncoming*. Encode a stay-outside half-plane for the **closest** such
    lane (the lane the ego is currently nearest to).

    The half-plane is the lane-boundary normal pointing AWAY from the
    oncoming lane's interior. We use the lane centerline + a default
    half-width to build the boundary line.

    Slot count: per-step, the rule reserves up to ``slots_per_step`` slots
    so multiple oncoming lanes can be excluded simultaneously. With 2
    slots, this is "stay outside the nearest 2 oncoming lanes" — enough
    for most intersection geometries.
    """

    rule_id = "7r2"
    priority_level = 2

    def __init__(
        self,
        slots_per_step: int = 2,
        heading_tolerance_rad: float = 0.35,    # ~20° from anti-parallel
        max_distance_m: float = 8.0,
        lane_half_width_m: float = 1.75,
    ) -> None:
        self.slots_per_step = slots_per_step
        self.heading_tolerance_rad = heading_tolerance_rad
        self.max_distance_m = max_distance_m
        self.lane_half_width_m = lane_half_width_m

    def applies_to_horizon(self, ctx: EncoderContext) -> bool:
        if ctx.map_view is None:
            return False
        return bool(ctx.map_view.nonroute_lanes) or bool(ctx.map_view.lane_connectors)

    def _is_oncoming(self, lane_heading: float, ego_heading: float) -> bool:
        """True when |Δheading| ≥ π − tolerance — anti-parallel within tolerance."""
        diff = abs(_principal_value(lane_heading - ego_heading))
        return diff >= (math.pi - self.heading_tolerance_rad)

    def encode(self, ctx: EncoderContext) -> List[List[LinearisedRuleConstraint]]:
        N = ctx.horizon_steps
        lanes = []
        if ctx.map_view is not None:
            lanes = list(ctx.map_view.nonroute_lanes) + list(ctx.map_view.lane_connectors)
        per_step: List[List[LinearisedRuleConstraint]] = []
        for k in range(N):
            x_bar = float(ctx.warm_start_X[0, k])
            y_bar = float(ctx.warm_start_X[1, k])
            psi_ego = float(ctx.warm_start_X[2, k])

            scored: List[Tuple[float, LinearisedRuleConstraint]] = []
            for lane in lanes:
                if lane.centerline_xy.shape[0] < 2:
                    continue
                # Find the closest centerline vertex.
                deltas = lane.centerline_xy - np.array([x_bar, y_bar])
                d2 = np.einsum("ij,ij->i", deltas, deltas)
                j = int(np.argmin(d2))
                dist = float(math.sqrt(d2[j]))
                if dist > self.max_distance_m:
                    continue
                lane_heading = float(lane.headings[j])
                if not self._is_oncoming(lane_heading, psi_ego):
                    continue
                # Build the stay-outside half-plane: the lane's boundary on the
                # ego side. Normal points from the lane centerline TOWARD the
                # ego (i.e., AWAY from the lane interior).
                cx, cy = float(lane.centerline_xy[j, 0]), float(lane.centerline_xy[j, 1])
                # The lane's local left-perpendicular direction:
                nl_x = -math.sin(lane_heading)
                nl_y = math.cos(lane_heading)
                # Side of the ego relative to centerline (sign of lateral offset).
                lat = (x_bar - cx) * nl_x + (y_bar - cy) * nl_y
                # Choose the boundary on the ego's side: distance half_width
                # from centerline along nl_{x,y} if lat > 0, else along -nl.
                if lat >= 0:
                    bx = cx + nl_x * self.lane_half_width_m
                    by = cy + nl_y * self.lane_half_width_m
                    n_out_x, n_out_y = nl_x, nl_y
                else:
                    bx = cx - nl_x * self.lane_half_width_m
                    by = cy - nl_y * self.lane_half_width_m
                    n_out_x, n_out_y = -nl_x, -nl_y
                # Constraint: stay on the *outside* of the boundary, i.e.,
                # (p - boundary_point) · n_out ≥ 0  ⇔  -n_out·p + n_out·boundary ≤ 0
                a = np.array([-n_out_x, -n_out_y, 0.0, 0.0])
                e = float(n_out_x * bx + n_out_y * by)
                # Sort key: how close the ego is to violating (smaller = more binding).
                slack_at_warm = -n_out_x * x_bar - n_out_y * y_bar + e
                scored.append(
                    (slack_at_warm, LinearisedRuleConstraint(a=a, b=np.zeros(NU), e=e, mask=1.0))
                )
            scored.sort(key=lambda p: p[0])
            slots = [c for _, c in scored[: self.slots_per_step]]
            while len(slots) < self.slots_per_step:
                slots.append(make_inactive_constraint())
            per_step.append(slots)
        return per_step


class OneWayDirectionRule(RuleEncoder):
    """7r3 — Don't drive against the flow on a one-way segment.

    Looser predicate than :class:`OpposingLaneRule`: fires when the ego's
    warm-start position is *inside* a lane whose heading differs from the
    ego heading by more than 90° (anything past pure perpendicular counts
    as "wrong way" for the loose definition). Encoded the same way as
    7r2 — a stay-outside half-plane for the wrong-way lane the ego is
    currently nearest to.

    Because 7r2 (oncoming with traffic) is a strict subset of 7r3
    (one-way wrong-direction), we share the geometry: this encoder
    catches the broader case where there's no oncoming traffic but the
    ego is still on a one-way segment in the wrong direction.
    """

    rule_id = "7r3"
    priority_level = 2

    def __init__(
        self,
        slots_per_step: int = 2,
        wrong_way_threshold_rad: float = math.pi / 2,   # > 90° from aligned
        max_distance_m: float = 6.0,
        lane_half_width_m: float = 1.75,
    ) -> None:
        self.slots_per_step = slots_per_step
        self.wrong_way_threshold_rad = wrong_way_threshold_rad
        self.max_distance_m = max_distance_m
        self.lane_half_width_m = lane_half_width_m

    def applies_to_horizon(self, ctx: EncoderContext) -> bool:
        return ctx.map_view is not None and (
            bool(ctx.map_view.nonroute_lanes) or bool(ctx.map_view.lane_connectors)
        )

    def encode(self, ctx: EncoderContext) -> List[List[LinearisedRuleConstraint]]:
        N = ctx.horizon_steps
        lanes = []
        if ctx.map_view is not None:
            lanes = list(ctx.map_view.nonroute_lanes) + list(ctx.map_view.lane_connectors)
        per_step: List[List[LinearisedRuleConstraint]] = []
        for k in range(N):
            x_bar = float(ctx.warm_start_X[0, k])
            y_bar = float(ctx.warm_start_X[1, k])
            psi_ego = float(ctx.warm_start_X[2, k])
            scored: List[Tuple[float, LinearisedRuleConstraint]] = []
            for lane in lanes:
                if lane.centerline_xy.shape[0] < 2:
                    continue
                deltas = lane.centerline_xy - np.array([x_bar, y_bar])
                d2 = np.einsum("ij,ij->i", deltas, deltas)
                j = int(np.argmin(d2))
                dist = float(math.sqrt(d2[j]))
                if dist > self.max_distance_m:
                    continue
                lane_heading = float(lane.headings[j])
                diff = abs(_principal_value(lane_heading - psi_ego))
                if diff < self.wrong_way_threshold_rad:
                    continue   # lane is roughly aligned with ego — not wrong-way
                cx, cy = float(lane.centerline_xy[j, 0]), float(lane.centerline_xy[j, 1])
                nl_x = -math.sin(lane_heading)
                nl_y = math.cos(lane_heading)
                lat = (x_bar - cx) * nl_x + (y_bar - cy) * nl_y
                if lat >= 0:
                    bx = cx + nl_x * self.lane_half_width_m
                    by = cy + nl_y * self.lane_half_width_m
                    n_out_x, n_out_y = nl_x, nl_y
                else:
                    bx = cx - nl_x * self.lane_half_width_m
                    by = cy - nl_y * self.lane_half_width_m
                    n_out_x, n_out_y = -nl_x, -nl_y
                a = np.array([-n_out_x, -n_out_y, 0.0, 0.0])
                e = float(n_out_x * bx + n_out_y * by)
                slack_at_warm = -n_out_x * x_bar - n_out_y * y_bar + e
                scored.append(
                    (slack_at_warm, LinearisedRuleConstraint(a=a, b=np.zeros(NU), e=e, mask=1.0))
                )
            scored.sort(key=lambda p: p[0])
            slots = [c for _, c in scored[: self.slots_per_step]]
            while len(slots) < self.slots_per_step:
                slots.append(make_inactive_constraint())
            per_step.append(slots)
        return per_step


def _principal_value(angle: float) -> float:
    """Wrap angle to (-π, π]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class SidewalkDriveRule(RuleEncoder):
    """7r5 — Do not drive on sidewalks.

    Each walkway polygon's outward half-plane :math:`n \\cdot p \\leq d`
    defines a region the ego must STAY OUT OF. Combined with an ego-radius
    buffer, the constraint at step k is:

    .. math::

        -n \\cdot p_k + d + r_{ego} \\leq 0

    i.e., the ego must lie on the *outside* of the walkway boundary by at
    least ``ego_buffer_m``. Linear in (x, y).
    """

    rule_id = "7r5"
    priority_level = 1

    def __init__(self, slots_per_step: int = 4, ego_buffer_m: float = 0.3) -> None:
        self.slots_per_step = slots_per_step
        self.ego_buffer_m = ego_buffer_m

    def applies_to_horizon(self, ctx: EncoderContext) -> bool:
        return ctx.map_view is not None and len(ctx.map_view.walkways) > 0

    def encode(self, ctx: EncoderContext) -> List[List[LinearisedRuleConstraint]]:
        N = ctx.horizon_steps
        per_step: List[List[LinearisedRuleConstraint]] = []
        polys = ctx.map_view.walkways if ctx.map_view else ()
        for k in range(N):
            x_bar = ctx.warm_start_X[0, k]
            y_bar = ctx.warm_start_X[1, k]
            scored: List[Tuple[float, LinearisedRuleConstraint]] = []
            for poly in polys:
                # Use the polygon's nearest half-plane as a single representative.
                # The ego must lie OUTSIDE the polygon — i.e., some half-plane
                # must hold ``n·p > d``. We negate the half-plane to encode
                # "stay on the outside" as -(n·p) + d + buffer <= 0.
                best_hp = None
                best_margin = -math.inf
                for hp in poly.half_planes:
                    margin = (hp.n_x * x_bar + hp.n_y * y_bar) - hp.d
                    if margin > best_margin:
                        best_margin = margin
                        best_hp = hp
                if best_hp is None:
                    continue
                a = np.array([-best_hp.n_x, -best_hp.n_y, 0.0, 0.0])
                e = best_hp.d + self.ego_buffer_m
                scored.append((-best_margin, LinearisedRuleConstraint(a=a, b=np.zeros(NU), e=float(e), mask=1.0)))
            scored.sort(key=lambda p: p[0])
            slots = [c for _, c in scored[: self.slots_per_step]]
            while len(slots) < self.slots_per_step:
                slots.append(make_inactive_constraint())
            per_step.append(slots)
        return per_step


# ----------------------------------------------------------------------
# Level 2 — Legal
# ----------------------------------------------------------------------


class SpeedLimitRule(RuleEncoder):
    """3r0 — Obey posted speed limits.

    Constraint at step k: :math:`v_k - v_{\\lim,k} \\leq 0`. Already linear.

    The speed limit per step is read from the reference's per-vertex speed
    limit (which the global planner derives from the lane's
    ``speed_limit_mps``). If the reference's speed limit is unset (None), we
    use ``ctx.desired_speed_mps`` as a soft fallback.
    """

    rule_id = "3r0"
    priority_level = 2

    def __init__(self, slots_per_step: int = 1, tolerance_mps: float = 1.0) -> None:
        self.slots_per_step = slots_per_step
        self.tolerance_mps = tolerance_mps

    def applies_to_horizon(self, ctx: EncoderContext) -> bool:
        return True  # always applies

    def encode(self, ctx: EncoderContext) -> List[List[LinearisedRuleConstraint]]:
        N = ctx.horizon_steps
        per_step: List[List[LinearisedRuleConstraint]] = []
        for k in range(N):
            v_lim = float(ctx.Xref_local[3, k])
            if v_lim <= 0:
                v_lim = ctx.desired_speed_mps
            a = np.array([0.0, 0.0, 0.0, 1.0])
            e = -(v_lim + self.tolerance_mps)
            per_step.append(
                [LinearisedRuleConstraint(a=a, b=np.zeros(NU), e=float(e), mask=1.0)]
            )
        return per_step


# ----------------------------------------------------------------------
# Level 3 — Comfort
# ----------------------------------------------------------------------


class SafeHeadwayRule(RuleEncoder):
    """3r3 — Maintain a safe time headway to the in-lane lead vehicle.

    Constraint at step k: :math:`t_{hw} v_k + d_{\\min} - \\mathrm{gap}_k \\leq 0`
    where :math:`\\mathrm{gap}_k` is the in-lane distance to the lead.

    Linearise :math:`\\mathrm{gap}_k` around the warm-start. Approximation
    (lead is straight ahead, ego is in the same lane): :math:`\\mathrm{gap}_k
    \\approx \\mathrm{gap}_{\\bar k} - (x_k - \\bar x_k)`, where
    :math:`\\bar x_k` is the warm-start longitudinal position and we treat the
    lead as stationary in ego-local for this tick. Substituting,

    .. math::

        t_{hw} v_k + d_{\\min} - \\mathrm{gap}_{\\bar k} + (x_k - \\bar x_k) \\leq 0

    so :math:`a = (1, 0, 0, t_{hw})`, :math:`b = 0`,
    :math:`e = d_{\\min} - \\mathrm{gap}_{\\bar k} - \\bar x_k`.
    """

    rule_id = "3r3"
    priority_level = 3

    def __init__(
        self,
        slots_per_step: int = 1,
        time_headway_s: float = 1.2,
        min_gap_m: float = 2.0,
        lateral_tol_m: float = 1.6,
    ) -> None:
        self.slots_per_step = slots_per_step
        self.time_headway_s = time_headway_s
        self.min_gap_m = min_gap_m
        self.lateral_tol_m = lateral_tol_m

    def _find_lead(self, ctx: EncoderContext) -> Optional[AgentSlot]:
        # Find the closest agent ahead of the ego (positive x in local frame)
        # that lies within ``lateral_tol_m`` of the ego's centerline.
        leads = [
            a for a in ctx.agents_local
            if a.x > 0.0 and abs(a.y) < self.lateral_tol_m
        ]
        if not leads:
            return None
        return min(leads, key=lambda a: a.x)

    def applies_to_horizon(self, ctx: EncoderContext) -> bool:
        return self._find_lead(ctx) is not None

    def encode(self, ctx: EncoderContext) -> List[List[LinearisedRuleConstraint]]:
        lead = self._find_lead(ctx)
        if lead is None:
            return self.all_inactive(ctx)
        N = ctx.horizon_steps
        per_step: List[List[LinearisedRuleConstraint]] = []
        # Effective lead distance buffer: agent half-length + ego half-length.
        lead_offset = 0.5 * lead.length + ctx.ego_radius_m
        for k in range(N):
            x_bar = ctx.warm_start_X[0, k]
            # Constant-velocity prediction of the lead in ego-local frame.
            lead_x_k = lead.x + lead.vx * ctx.dt_s * k
            gap_bar_k = lead_x_k - x_bar - lead_offset
            a = np.array([1.0, 0.0, 0.0, self.time_headway_s])
            e = self.min_gap_m - gap_bar_k - x_bar
            per_step.append(
                [LinearisedRuleConstraint(a=a, b=np.zeros(NU), e=float(e), mask=1.0)]
            )
        return per_step


class LongitudinalComfortRule(RuleEncoder):
    """0r2 — Keep |a_x| under a comfort threshold.

    Since the longitudinal acceleration in our model is a *control* (``u[0]``),
    not a state, the constraint is already linear: ``u[0] - a_max <= 0``
    AND ``-u[0] - a_max <= 0``. Two slots per step.
    """

    rule_id = "0r2"
    priority_level = 3

    def __init__(self, slots_per_step: int = 2, a_max_comf_mps2: float = 1.8) -> None:
        self.slots_per_step = slots_per_step
        self.a_max_comf_mps2 = a_max_comf_mps2

    def applies_to_horizon(self, ctx: EncoderContext) -> bool:
        return True

    def encode(self, ctx: EncoderContext) -> List[List[LinearisedRuleConstraint]]:
        N = ctx.horizon_steps
        per_step: List[List[LinearisedRuleConstraint]] = []
        for k in range(N):
            slots = [
                LinearisedRuleConstraint(
                    a=np.zeros(NX), b=np.array([1.0, 0.0]),
                    e=-self.a_max_comf_mps2, mask=1.0,
                ),  # +a_x - a_max <= 0
                LinearisedRuleConstraint(
                    a=np.zeros(NX), b=np.array([-1.0, 0.0]),
                    e=-self.a_max_comf_mps2, mask=1.0,
                ),  # -a_x - a_max <= 0
            ]
            per_step.append(slots)
        return per_step


class LateralAccelerationRule(RuleEncoder):
    """1r11 — Keep lateral acceleration under a comfort threshold.

    Lateral acceleration of the kinematic bicycle (centre, small-slip approx):

    .. math::

        a_y \\approx \\frac{v_k^2 \\tan \\delta_k}{L}.

    Linearise at warm-start (:math:`\\bar v_k, \\bar \\delta_k`):

    .. math::

        a_y \\approx \\bar a_y + \\frac{2\\bar v_k \\tan \\bar \\delta_k}{L}(v_k - \\bar v_k)
                              + \\frac{\\bar v_k^2 \\sec^2 \\bar \\delta_k}{L}(\\delta_k - \\bar \\delta_k)

    Constraint :math:`|a_y| \\leq a_{y,\\max}` becomes two linear inequalities.
    """

    rule_id = "1r11"
    priority_level = 3

    def __init__(self, slots_per_step: int = 2, a_y_max_comf_mps2: float = 2.0) -> None:
        self.slots_per_step = slots_per_step
        self.a_y_max_comf_mps2 = a_y_max_comf_mps2

    def applies_to_horizon(self, ctx: EncoderContext) -> bool:
        return True

    def encode(self, ctx: EncoderContext) -> List[List[LinearisedRuleConstraint]]:
        N = ctx.horizon_steps
        L = ctx.wheel_base_m
        per_step: List[List[LinearisedRuleConstraint]] = []
        for k in range(N):
            v_bar = ctx.warm_start_X[3, k]
            d_bar = ctx.warm_start_U[1, k] if k < ctx.warm_start_U.shape[1] else 0.0
            tan_d_bar = math.tan(d_bar)
            sec2_d_bar = 1.0 / (math.cos(d_bar) ** 2 + 1e-9)
            a_y_bar = v_bar * v_bar * tan_d_bar / L
            d_dvdv = 2.0 * v_bar * tan_d_bar / L
            d_dvdd = v_bar * v_bar * sec2_d_bar / L
            # Linearised a_y in MPC variables:
            # a_y ≈ a_y_bar + d_dvdv * (v - v_bar) + d_dvdd * (δ - d_bar)
            #     = d_dvdv * v + d_dvdd * δ + (a_y_bar - d_dvdv*v_bar - d_dvdd*d_bar)
            slope_v = d_dvdv
            slope_d = d_dvdd
            const = a_y_bar - d_dvdv * v_bar - d_dvdd * d_bar
            # +a_y - a_max <= 0
            slots = [
                LinearisedRuleConstraint(
                    a=np.array([0.0, 0.0, 0.0, slope_v]),
                    b=np.array([0.0, slope_d]),
                    e=float(const - self.a_y_max_comf_mps2),
                    mask=1.0,
                ),
                # -a_y - a_max <= 0
                LinearisedRuleConstraint(
                    a=np.array([0.0, 0.0, 0.0, -slope_v]),
                    b=np.array([0.0, -slope_d]),
                    e=float(-const - self.a_y_max_comf_mps2),
                    mask=1.0,
                ),
            ]
            per_step.append(slots)
        return per_step


class TrafficLightRule(RuleEncoder):
    """7r1 — Stop before the stop line when the controlling light is RED/YELLOW.

    For every :class:`~.map_lifter.LocalStopLine` with
    ``stop_type == "TRAFFIC_LIGHT"`` whose ``associated_connector_id``
    has a RED or YELLOW state in ``ctx.map_view.traffic_lights``, encode

    .. math::

        x_{\\mathrm{ego},k} - (x_{\\mathrm{stop}} - \\mathrm{buffer}) \\leq 0

    where :math:`x_{\\mathrm{stop}}` is the longitudinal coordinate (in the
    ego rear-axle frame) of the stop-line's leading edge.

    The constraint is active at every step of the horizon. Stop lines
    that are already behind the ego (:math:`x_{\\mathrm{stop}} < 0`) are
    masked out. With ``slots_per_step = 1`` we encode at most the
    nearest live red/yellow stop line per step.
    """

    rule_id = "7r1"
    priority_level = 2

    def __init__(
        self,
        slots_per_step: int = 1,
        max_distance_m: float = 60.0,
        safety_buffer_m: float = 1.0,
    ) -> None:
        self.slots_per_step = slots_per_step
        self.max_distance_m = max_distance_m
        self.safety_buffer_m = safety_buffer_m

    def _live_red_lights(self, ctx: EncoderContext) -> set[str]:
        if ctx.map_view is None:
            return set()
        return {
            tl.connector_id for tl in ctx.map_view.traffic_lights
            if tl.state in ("RED", "YELLOW")
        }

    def _relevant_stop_x(self, ctx: EncoderContext) -> Optional[float]:
        """Return the smallest positive x_stop across all live red/yellow stops."""
        if ctx.map_view is None:
            return None
        red_ids = self._live_red_lights(ctx)
        if not red_ids:
            return None
        best_x: Optional[float] = None
        for sl in ctx.map_view.stop_lines:
            if sl.stop_type != "TRAFFIC_LIGHT":
                continue
            if sl.associated_connector_id not in red_ids:
                continue
            if sl.polyline_xy.shape[0] == 0:
                continue
            # Use the nearest-ahead vertex (smallest positive x).
            xs = sl.polyline_xy[:, 0]
            ys = sl.polyline_xy[:, 1]
            ahead = xs > 0.0
            if not ahead.any():
                continue
            # Require the stop line to span the ego's lane (|y| within a band).
            in_band = np.abs(ys) < 4.0
            mask = ahead & in_band
            if not mask.any():
                continue
            x_candidate = float(xs[mask].min())
            if x_candidate > self.max_distance_m:
                continue
            if best_x is None or x_candidate < best_x:
                best_x = x_candidate
        return best_x

    def applies_to_horizon(self, ctx: EncoderContext) -> bool:
        return self._relevant_stop_x(ctx) is not None

    def encode(self, ctx: EncoderContext) -> List[List[LinearisedRuleConstraint]]:
        x_stop = self._relevant_stop_x(ctx)
        N = ctx.horizon_steps
        if x_stop is None:
            return self.all_inactive(ctx)
        e = -x_stop + self.safety_buffer_m
        a = np.array([1.0, 0.0, 0.0, 0.0])
        per_step: List[List[LinearisedRuleConstraint]] = []
        for _ in range(N):
            per_step.append(
                [LinearisedRuleConstraint(a=a, b=np.zeros(NU), e=float(e), mask=1.0)]
            )
        return per_step


class LateralClearanceRule(RuleEncoder):
    """3r5 — Maintain lateral clearance to laterally-adjacent agents.

    Mirror of the observer's 3r5 (``min_lateral_m = 1.0``,
    ``vrel_coef_s = 0.5``). At warm-start, identify agents within a
    longitudinal window (``±max_long_m``) and lateral band (``±lateral_band_m``)
    of the ego. For each such agent, encode a one-sided half-plane that
    keeps the ego at lateral distance ``d_safe = min_lateral_m +
    vrel_coef_s · v_close + 0.5 · (W_ego + W_agent)`` from the agent's
    predicted lateral position.

    The constraint is in *y_ego*: linear, single slot per agent. We
    populate up to ``slots_per_step`` agents per step (sorted by current
    lateral shortfall, most-binding first).
    """

    rule_id = "3r5"
    priority_level = 3

    def __init__(
        self,
        slots_per_step: int = 2,
        min_lateral_m: float = 1.0,
        vrel_coef_s: float = 0.5,
        max_long_m: float = 8.0,
        lateral_band_m: float = 4.0,
        ego_half_width_m: float = 1.0,   # half of nuPlan ego width (~2.0 m)
    ) -> None:
        self.slots_per_step = slots_per_step
        self.min_lateral_m = min_lateral_m
        self.vrel_coef_s = vrel_coef_s
        self.max_long_m = max_long_m
        self.lateral_band_m = lateral_band_m
        self.ego_half_width_m = ego_half_width_m

    def _adjacent_agents(self, ctx: EncoderContext) -> List[AgentSlot]:
        return [
            a for a in ctx.agents_local
            if abs(a.x) < self.max_long_m and 1e-3 < abs(a.y) < self.lateral_band_m
        ]

    def applies_to_horizon(self, ctx: EncoderContext) -> bool:
        return bool(self._adjacent_agents(ctx))

    def encode(self, ctx: EncoderContext) -> List[List[LinearisedRuleConstraint]]:
        adj = self._adjacent_agents(ctx)
        if not adj:
            return self.all_inactive(ctx)
        N = ctx.horizon_steps
        per_step: List[List[LinearisedRuleConstraint]] = []
        for k in range(N):
            y_bar = float(ctx.warm_start_X[1, k])
            scored: List[Tuple[float, LinearisedRuleConstraint]] = []
            for a in adj:
                # Constant-velocity prediction in ego-local frame.
                y_a_k = a.y + a.vy * ctx.dt_s * k
                # Closing lateral velocity (positive => agent approaching ego).
                v_close = max(0.0, -a.vy if y_a_k > 0 else a.vy)
                half_w = 0.5 * a.width + self.ego_half_width_m
                d_safe = self.min_lateral_m + self.vrel_coef_s * v_close + half_w
                if y_a_k > 0:
                    # Agent to ego's left at step k.  y_ego ≤ y_a − d_safe
                    # → y_ego − y_a + d_safe ≤ 0  (a on x = (0, 1, 0, 0))
                    a_vec = np.array([0.0, 1.0, 0.0, 0.0])
                    e = -y_a_k + d_safe
                    shortfall = y_bar - y_a_k + d_safe
                else:
                    # Agent to ego's right.  y_ego ≥ y_a + d_safe
                    # → −y_ego + y_a + d_safe ≤ 0
                    a_vec = np.array([0.0, -1.0, 0.0, 0.0])
                    e = y_a_k + d_safe
                    shortfall = -y_bar + y_a_k + d_safe
                scored.append(
                    (-shortfall, LinearisedRuleConstraint(a=a_vec, b=np.zeros(NU), e=float(e), mask=1.0))
                )
            scored.sort(key=lambda p: p[0])
            slots = [c for _, c in scored[: self.slots_per_step]]
            while len(slots) < self.slots_per_step:
                slots.append(make_inactive_constraint())
            per_step.append(slots)
        return per_step


class LateralComfortRule(RuleEncoder):
    """0r3 — Soft cap on steering-angle magnitude (comfort proxy).

    The observer's 0r3 measures |a_y| (covered by 1r11 at a tighter
    threshold) and lateral jerk (Δa_y/dt). True jerk constraints couple
    consecutive steps and don't fit the per-step
    :class:`LinearisedRuleConstraint` shape; the MPC's existing
    ``weight_control_rate`` cost penalises |Δu|², which is the dominant
    jerk contributor.

    What we add here is a *tighter-than-actuator* bound on |δ| that
    serves as a comfort-level shave on aggressive steering — one
    LCP-level-3 violation slot whose hinge fires when the optimiser
    wants to use more than ``steer_max_comf_rad`` of steering.
    """

    rule_id = "0r3"
    priority_level = 3

    def __init__(self, slots_per_step: int = 2, steer_max_comf_rad: float = 0.30) -> None:
        self.slots_per_step = slots_per_step
        self.steer_max_comf_rad = steer_max_comf_rad

    def applies_to_horizon(self, ctx: EncoderContext) -> bool:
        return True

    def encode(self, ctx: EncoderContext) -> List[List[LinearisedRuleConstraint]]:
        N = ctx.horizon_steps
        per_step: List[List[LinearisedRuleConstraint]] = []
        for _ in range(N):
            slots = [
                LinearisedRuleConstraint(
                    a=np.zeros(NX), b=np.array([0.0, 1.0]),
                    e=-self.steer_max_comf_rad, mask=1.0,
                ),  # +δ − δ_max ≤ 0
                LinearisedRuleConstraint(
                    a=np.zeros(NX), b=np.array([0.0, -1.0]),
                    e=-self.steer_max_comf_rad, mask=1.0,
                ),  # −δ − δ_max ≤ 0
            ]
            per_step.append(slots)
        return per_step


# ----------------------------------------------------------------------
# Stubs: rules deferred to the next session
# ----------------------------------------------------------------------


class StubRule(RuleEncoder):
    """Documented placeholder for a rule whose encoding lands next session.

    Carries the priority level and slot count so the level's
    ``LCPLevelSpec.slots_per_step`` budget is reserved correctly, but
    :meth:`applies_to_horizon` always returns False — every slot is filled
    with :func:`make_inactive_constraint`. Replace the subclass body with the
    concrete encoding when ready.
    """

    def __init__(self, rule_id: str, priority_level: int, slots_per_step: int, doc: str) -> None:
        self.rule_id = rule_id
        self.priority_level = priority_level
        self.slots_per_step = slots_per_step
        self.__doc__ = doc

    def applies_to_horizon(self, ctx: EncoderContext) -> bool:
        return False

    def encode(self, ctx: EncoderContext) -> List[List[LinearisedRuleConstraint]]:
        return self.all_inactive(ctx)


def make_default_ruleset() -> RuleSet:
    """Build the default 4-level RuleSet wiring all 16 rules.

    .. NOTE::
       Earlier revisions of this function wired in the *polygon-based*
       :class:`DrivableBoundaryRule` and :class:`SidewalkDriveRule`. Both
       were structurally wrong (the drivable-surface predicate is a
       non-convex union of lane polygons, but the encoder enforced the
       intersection; the sidewalk rule needs an OR-of-half-planes that the
       single-half-plane reduction does not express). They are replaced
       here by :class:`LaneCorridorRule` — a corridor around the route
       centreline expressed as two linear half-planes per step, which is
       the convex localisation of "stay on the drivable surface" and
       requires no polygon-membership lookup at tick time.
    """
    safety = [
        CollisionRule(slots_per_step=8),
        # 7r0: stay within ±half_width of the route centreline (2 half-planes).
        LaneCorridorRule(slots_per_step=2, half_width_m=1.0),
        # 7r5 sidewalk drive — closest-face convex localisation per warm-start.
        SidewalkDriveRule(slots_per_step=4),
        StubRule("10r5", 1, slots_per_step=2, doc="Bike-lane encroachment (NuPlan-mini doesn't expose bike lanes)."),
    ]
    legal = [
        SpeedLimitRule(slots_per_step=1),
        OpposingLaneRule(slots_per_step=2),
        OneWayDirectionRule(slots_per_step=2),
        TrafficLightRule(slots_per_step=1),
        StubRule("7r4", 2, slots_per_step=1, doc="Stop-in-crosswalk — multi-tick condition."),
    ]
    comfort = [
        SafeHeadwayRule(slots_per_step=1),
        LongitudinalComfortRule(slots_per_step=2),
        LateralAccelerationRule(slots_per_step=2),
        LateralClearanceRule(slots_per_step=2),
        StubRule("3r6", 3, slots_per_step=2, doc="Lane intrusion / lateral TTC — multi-agent state-machine."),
        LateralComfortRule(slots_per_step=2),
    ]
    return RuleSet(levels=[safety, legal, comfort])
