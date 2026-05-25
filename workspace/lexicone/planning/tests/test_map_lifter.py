"""Unit tests for the per-tick map lifter.

These tests do not touch nuplan-devkit at all — they exercise the half-plane
reduction and polygon→local-frame transform on synthetic polygons.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from lexicone.planning.map_lifter import (
    H_PER_POLY,
    HalfPlane,
    LocalPolygon,
    _distance_from_origin_to_polygon,
    _polygon_to_half_planes,
)


def _unit_square_ccw() -> np.ndarray:
    """A 2 m × 2 m square centred at the origin, vertices in CCW order."""
    return np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]], dtype=np.float64)


def test_half_plane_reduction_square_outward_normals():
    """For a CCW unit square at origin, the four outward normals must point
    along ±x and ±y, with offset 1.0 each."""
    halves = _polygon_to_half_planes(_unit_square_ccw(), max_planes=H_PER_POLY)
    assert len(halves) == 4
    # All offsets equal 1 (each edge is distance 1 from origin).
    for hp in halves:
        assert hp.d == pytest.approx(1.0, abs=1e-9)
    # Outward normals (modulo ordering): {(0,-1), (1,0), (0,1), (-1,0)}.
    seen_normals = sorted(
        [(round(hp.n_x, 6), round(hp.n_y, 6)) for hp in halves]
    )
    expected = sorted([(0.0, -1.0), (1.0, 0.0), (0.0, 1.0), (-1.0, 0.0)])
    assert seen_normals == expected


def test_half_plane_reduction_square_cw_winding():
    """Polygons with CW winding (mirror of the CCW case) must still produce
    outward normals — i.e., normals pointing AWAY from the polygon interior."""
    cw = _unit_square_ccw()[::-1].copy()
    halves = _polygon_to_half_planes(cw, max_planes=H_PER_POLY)
    assert len(halves) == 4
    # Every half-plane's normal should point away from the interior. For each
    # half-plane n·x <= d, the origin (interior) should satisfy n·0 == 0 <= 1.
    for hp in halves:
        assert hp.d == pytest.approx(1.0, abs=1e-9)
        # The origin is inside the polygon, so 0 <= d must hold.
        assert hp.d >= 0


def test_half_plane_reduction_respects_max_planes():
    """A polygon with more vertices than ``max_planes`` is reduced to exactly
    ``max_planes`` half-planes — the ones closest to the origin."""
    # Regular 12-gon of radius 5 around (10, 0): the origin lies outside, but
    # this still exercises the reduction logic.
    n_vertices = 12
    angles = np.linspace(0, 2 * math.pi, n_vertices, endpoint=False)
    pts = np.column_stack([10.0 + 5.0 * np.cos(angles), 5.0 * np.sin(angles)])
    halves = _polygon_to_half_planes(pts, max_planes=4)
    assert len(halves) == 4
    # The four kept half-planes should be the four edges closest to the origin —
    # i.e., the edges on the polygon's left side (smallest x). Each closest-d
    # value should be less than the polygon's average distance from origin.
    avg_d_to_centre = 10.0
    assert all(hp.d < avg_d_to_centre for hp in halves)


def test_half_plane_origin_inside_polygon_satisfies_all():
    """If the origin is strictly inside the polygon, every half-plane
    ``n·x <= d`` evaluated at ``x = 0`` produces ``0 <= d`` with d > 0."""
    halves = _polygon_to_half_planes(_unit_square_ccw(), max_planes=H_PER_POLY)
    for hp in halves:
        assert 0.0 <= hp.d


def test_half_plane_origin_outside_polygon_violates_at_least_one():
    """If the origin is outside the polygon, at least one half-plane is violated
    at ``x = 0``, i.e., ``n·0 = 0 > d`` for some half-plane (d < 0)."""
    # Square shifted so origin is outside.
    shifted = _unit_square_ccw() + np.array([3.0, 0.0])
    halves = _polygon_to_half_planes(shifted, max_planes=H_PER_POLY)
    # At least one half-plane has d < 0 (origin is on the wrong side).
    assert any(hp.d < 0.0 for hp in halves)


def test_distance_from_origin_to_polygon_inside_returns_distance_to_nearest_edge():
    """For the unit square centred at origin, the closest edge is 1 m away."""
    d = _distance_from_origin_to_polygon(_unit_square_ccw())
    assert d == pytest.approx(1.0, abs=1e-9)


def test_distance_from_origin_to_polygon_outside():
    """Square shifted to (3, 0): closest edge is at x=2, so distance = 2."""
    pts = _unit_square_ccw() + np.array([3.0, 0.0])
    d = _distance_from_origin_to_polygon(pts)
    assert d == pytest.approx(2.0, abs=1e-9)


def test_degenerate_two_point_polygon_returns_empty():
    """Polygons with fewer than 3 distinct vertices are rejected."""
    halves = _polygon_to_half_planes(np.array([[0.0, 0.0], [1.0, 0.0]]), max_planes=H_PER_POLY)
    assert halves == []


def test_half_planes_are_unit_length_normals():
    """Each returned half-plane's normal vector has unit length."""
    halves = _polygon_to_half_planes(_unit_square_ccw(), max_planes=H_PER_POLY)
    for hp in halves:
        norm = math.hypot(hp.n_x, hp.n_y)
        assert norm == pytest.approx(1.0, abs=1e-9)


def test_localpolygon_construction():
    """A LocalPolygon dataclass round-trips through tuple coercion as expected."""
    halves = _polygon_to_half_planes(_unit_square_ccw(), max_planes=H_PER_POLY)
    lp = LocalPolygon(polygon_id="test", half_planes=tuple(halves), closest_distance_m=1.0)
    assert lp.polygon_id == "test"
    assert len(lp.half_planes) == 4
    assert lp.closest_distance_m == 1.0
