"""Arc-length parameterised reference path used by the trajectory planner.

The reference is a polyline derived from concatenated lane baseline paths.
For every vertex we cache cumulative arc length ``s``, heading ``psi`` and
per-segment unit tangent. Sampling / projection are implemented with O(N) scans
because the leading window of the route handed to the MPC is always short
(N <= a few hundred vertices).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from nuplan.common.actor_state.state_representation import Point2D, StateSE2
from nuplan.common.geometry.compute import principal_value


@dataclass
class ReferenceSample:
    s: float
    x: float
    y: float
    psi: float
    v_limit: float


class ReferencePath:
    """Polyline reference parameterised by cumulative arc length."""

    def __init__(self, points: np.ndarray, speed_limit_mps: np.ndarray):
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError(f"points must have shape (N, 2), got {points.shape}")
        if points.shape[0] < 2:
            raise ValueError(f"reference path needs >= 2 points, got {points.shape[0]}")
        if speed_limit_mps.shape != (points.shape[0],):
            raise ValueError(
                f"speed_limit_mps shape {speed_limit_mps.shape} does not match points {points.shape}"
            )

        self._xy = points.astype(np.float64)
        self._v_limit = speed_limit_mps.astype(np.float64)

        deltas = np.diff(self._xy, axis=0)
        seg_len = np.hypot(deltas[:, 0], deltas[:, 1])
        # Vertices coincident with their successor would produce zero-length
        # segments and NaN headings. Filter them out.
        keep = np.concatenate(([True], seg_len > 1e-6))
        if not keep.all():
            self._xy = self._xy[keep]
            self._v_limit = self._v_limit[keep]
            deltas = np.diff(self._xy, axis=0)
            seg_len = np.hypot(deltas[:, 0], deltas[:, 1])
            if self._xy.shape[0] < 2:
                raise ValueError("reference path collapses to < 2 distinct points")

        self._s = np.concatenate(([0.0], np.cumsum(seg_len)))
        seg_psi = np.arctan2(deltas[:, 1], deltas[:, 0])
        # Per-vertex heading: average adjacent segments, with end-points taking the only neighbour.
        vertex_psi = np.empty(self._xy.shape[0])
        vertex_psi[0] = seg_psi[0]
        vertex_psi[-1] = seg_psi[-1]
        if seg_psi.shape[0] > 1:
            mid = seg_psi[:-1] + 0.5 * np.vectorize(principal_value)(seg_psi[1:] - seg_psi[:-1])
            vertex_psi[1:-1] = mid
        self._psi = np.vectorize(principal_value)(vertex_psi)
        self._seg_psi = seg_psi
        self._seg_len = seg_len

    @property
    def length(self) -> float:
        return float(self._s[-1])

    @property
    def points(self) -> np.ndarray:
        return self._xy

    def sample(self, s_query: float) -> ReferenceSample:
        s = float(np.clip(s_query, 0.0, self.length))
        idx = int(np.searchsorted(self._s, s, side="right") - 1)
        idx = max(0, min(idx, self._xy.shape[0] - 2))
        s0 = self._s[idx]
        seg = self._seg_len[idx]
        alpha = 0.0 if seg <= 0.0 else (s - s0) / seg
        x = (1.0 - alpha) * self._xy[idx, 0] + alpha * self._xy[idx + 1, 0]
        y = (1.0 - alpha) * self._xy[idx, 1] + alpha * self._xy[idx + 1, 1]
        # Headings interpolated via shortest-arc to avoid wrap discontinuities.
        d_psi = principal_value(self._psi[idx + 1] - self._psi[idx])
        psi = principal_value(self._psi[idx] + alpha * d_psi)
        v_limit = (1.0 - alpha) * self._v_limit[idx] + alpha * self._v_limit[idx + 1]
        return ReferenceSample(s=s, x=x, y=y, psi=psi, v_limit=v_limit)

    def project(self, point: Point2D) -> Tuple[float, float]:
        """Return (arc length s, signed lateral offset) of closest point on the polyline.

        Lateral offset is positive to the left of the path direction.
        """
        best_s = 0.0
        best_d2 = float("inf")
        best_lat = 0.0
        px, py = float(point.x), float(point.y)
        for idx in range(self._xy.shape[0] - 1):
            ax, ay = self._xy[idx]
            bx, by = self._xy[idx + 1]
            dx = bx - ax
            dy = by - ay
            seg_len2 = dx * dx + dy * dy
            if seg_len2 <= 0.0:
                continue
            t = ((px - ax) * dx + (py - ay) * dy) / seg_len2
            t_clamped = max(0.0, min(1.0, t))
            qx = ax + t_clamped * dx
            qy = ay + t_clamped * dy
            d2 = (px - qx) ** 2 + (py - qy) ** 2
            if d2 < best_d2:
                best_d2 = d2
                seg = np.sqrt(seg_len2)
                best_s = float(self._s[idx] + t_clamped * seg)
                # signed cross product with unit tangent → lateral offset
                tx = dx / seg
                ty = dy / seg
                best_lat = float((px - qx) * (-ty) + (py - qy) * tx)
        return best_s, best_lat


def reference_from_se2_polyline(
    discrete_path: Sequence[StateSE2],
    speed_limits_mps: Sequence[Optional[float]],
    default_speed_limit_mps: float,
    start_s: float = 0.0,
    end_s: Optional[float] = None,
) -> ReferencePath:
    """Build a :class:`ReferencePath` from a sequence of :class:`StateSE2` plus per-vertex speed limits.

    The polyline can be trimmed in arc length to [start_s, end_s] before construction.
    """
    if len(discrete_path) != len(speed_limits_mps):
        raise ValueError(
            f"discrete_path ({len(discrete_path)}) and speed_limits_mps ({len(speed_limits_mps)}) length mismatch"
        )
    xy = np.array([[s.x, s.y] for s in discrete_path], dtype=np.float64)
    vlim = np.array(
        [default_speed_limit_mps if v is None else float(v) for v in speed_limits_mps],
        dtype=np.float64,
    )

    if start_s > 0.0 or end_s is not None:
        deltas = np.diff(xy, axis=0)
        cum = np.concatenate(([0.0], np.cumsum(np.hypot(deltas[:, 0], deltas[:, 1]))))
        lo_idx = int(np.searchsorted(cum, start_s, side="left"))
        if end_s is not None:
            hi_idx = int(np.searchsorted(cum, end_s, side="right"))
        else:
            hi_idx = xy.shape[0]
        lo_idx = max(0, min(lo_idx, xy.shape[0] - 2))
        hi_idx = max(lo_idx + 2, min(hi_idx, xy.shape[0]))
        xy = xy[lo_idx:hi_idx]
        vlim = vlim[lo_idx:hi_idx]

    return ReferencePath(xy, vlim)


def straight_reference(
    origin: StateSE2,
    length_m: float,
    speed_limit_mps: float,
    num_points: int = 50,
) -> ReferencePath:
    """Build a straight reference extending forward from ``origin`` along its heading.

    Used as a fallback when the global planner cannot produce a route yet.
    """
    ts = np.linspace(0.0, length_m, num_points)
    xs = origin.x + ts * np.cos(origin.heading)
    ys = origin.y + ts * np.sin(origin.heading)
    xy = np.column_stack([xs, ys])
    vlim = np.full(num_points, speed_limit_mps, dtype=np.float64)
    return ReferencePath(xy, vlim)
