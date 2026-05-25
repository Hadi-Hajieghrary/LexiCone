"""Per-tick map extraction for the LCP MPC trajectory planner.

The LCP MPC needs lane, walkway, bike-lane, crosswalk, stop-line, and intersection
polygons (plus a per-step traffic-light schedule) to encode 12 of the 16 rules
that involve map data. The observer's :mod:`lexicone.observer.nuplan_adapter`
already queries these from a NuPlan ``map_api``, but it materialises them in
*world* coordinates for the rule engine; the MPC needs them in *ego-local*
coordinates (origin at the rear axle at the current tick, x-axis aligned with
ego heading) so IPOPT's internal scaling stays well-conditioned.

This module mirrors the observer's extraction pattern but produces a
:class:`MapHorizonView` whose polygons are already rotated/translated into the
ego-local frame. Polygons of arbitrary vertex count are over-approximated by a
fixed number of half-planes (the K nearest to the ego), so they fit into the
MPC's fixed-size parameter slots.

Key design choices:

- **Soft nuplan-devkit dependency.** The module imports ``nuplan`` lazily inside
  :meth:`MapLifter.__init__`. Construction without a live ``map_api`` is allowed
  for unit tests; ``view()`` is the only call that touches nuplan.
- **Per-tick freshness.** Each ``view(ego_state, traffic_lights)`` call
  re-queries the map. There is no caching across ticks because the ego moves;
  caching by spatial hash is a future optimisation.
- **Half-plane over-approximation.** Each polygon is reduced to ``H_PER_POLY``
  half-planes (default 6), keeping the ones closest to the ego footprint. Convex
  polygons round-trip exactly through this representation up to
  ``H_PER_POLY``-sided over-approximation; non-convex polygons (rare for nuPlan
  lane/walkway geometry) are conservatively over-approximated by their convex
  hull.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# Number of half-planes used to over-approximate each polygon when packed into
# the MPC's parameter slots. Convex polygons with <= H_PER_POLY edges round-trip
# exactly; larger polygons are reduced by retaining the H_PER_POLY edges whose
# supporting half-plane is closest to the ego footprint.
H_PER_POLY: int = 6


@dataclass
class HalfPlane:
    """Outward half-plane ``n_x · x + n_y · y <= d`` in ego-local frame.

    The polygon's interior is on the side where ``n_x · x + n_y · y <= d``.
    The normal ``(n_x, n_y)`` is unit-length. A point is *inside* the polygon
    iff all of its half-planes return ``<= d``.
    """

    n_x: float
    n_y: float
    d: float


@dataclass
class LocalPolygon:
    """A polygon in ego-local frame, represented as up to ``H_PER_POLY`` half-planes."""

    polygon_id: str
    half_planes: Tuple[HalfPlane, ...]
    # Closest point on the polygon to the local origin (for prioritising which
    # half-planes are retained when the polygon has more edges than H_PER_POLY).
    closest_distance_m: float


@dataclass
class LocalLane:
    """A lane segment in ego-local frame.

    The lane is summarised by its centerline polyline (in local frame) plus the
    derived per-vertex headings and the lane width. The boundary is reconstructed
    as the centerline shifted by ``±width/2`` along the normal direction.
    """

    lane_id: str
    centerline_xy: np.ndarray  # shape (N, 2)
    headings: np.ndarray       # shape (N,) — per-vertex heading in local frame
    width_m: float
    speed_limit_mps: Optional[float]
    is_lane_connector: bool
    in_route_corridor: bool
    incoming_ids: Tuple[str, ...] = ()
    outgoing_ids: Tuple[str, ...] = ()


@dataclass
class LocalStopLine:
    """A stop line (associated with a traffic light or stop sign) in ego-local frame.

    For traffic-light rule (``7r1``), we additionally carry the connector ID so the
    rule encoder can correlate against the per-tick traffic-light status array.
    """

    stop_line_id: str
    polyline_xy: np.ndarray            # shape (M, 2)
    associated_connector_id: Optional[str]
    stop_type: str                     # "TRAFFIC_LIGHT" | "STOP_SIGN" | "YIELD_SIGN" | "GENERIC"


@dataclass
class LocalTrafficLight:
    """Per-connector traffic-light state at the current tick."""

    connector_id: str
    state: str                         # "GREEN" | "YELLOW" | "RED" | "UNKNOWN"


@dataclass
class MapHorizonView:
    """Everything the rule encoder needs from the map at this tick, in ego-local frame.

    All polygons are in the **ego rear-axle frame**: origin at the ego's rear
    axle, x-axis aligned with the ego heading. The same transform the LCP MPC
    uses internally — so the rule encoder's outputs feed straight into the
    MPC's CasADi parameter slots without an additional rotation.
    """

    # Centerline / heading data for lane orientation rules (7r2, 7r3) and for
    # the lane-boundary signed-distance constraint (7r0 partial).
    route_lanes: Tuple[LocalLane, ...] = ()         # lanes inside the planned route corridor
    nonroute_lanes: Tuple[LocalLane, ...] = ()      # lanes outside the route corridor but inside ROI
    lane_connectors: Tuple[LocalLane, ...] = ()
    # Drivable-surface approximation (union of lane polygons is what we use, since
    # NuPlanMap doesn't expose DRIVABLE_AREA as a queryable map object).
    drivable_polygons: Tuple[LocalPolygon, ...] = ()
    walkways: Tuple[LocalPolygon, ...] = ()         # for 7r5 sidewalk-drive
    bike_lanes: Tuple[LocalPolygon, ...] = ()       # for 10r5 (NuPlan generally does not expose these)
    crosswalks: Tuple[LocalPolygon, ...] = ()       # for 7r4 stop-in-crosswalk
    intersections: Tuple[LocalPolygon, ...] = ()
    stop_lines: Tuple[LocalStopLine, ...] = ()
    traffic_lights: Tuple[LocalTrafficLight, ...] = ()
    # Frame metadata so consumers can sanity-check the transform.
    anchor_xy_world: np.ndarray = field(default_factory=lambda: np.zeros(2))
    anchor_heading_world: float = 0.0


class MapLifter:
    """Per-tick map extraction in ego-local frame.

    Construction stores a reference to the live nuPlan ``map_api`` and the
    scenario's route-roadblock identifiers (used to mark which lanes lie inside
    the planned corridor — the ``RouteAdherenceRule`` ``2r2`` and the lane
    half-plane construction rely on this flag).

    Per-tick, :meth:`view` queries the map within ``radius_m`` of the ego and
    builds a :class:`MapHorizonView` whose polygons are all rotated/translated
    into the ego-local frame.
    """

    def __init__(
        self,
        map_api: Any,
        route_roadblock_ids: Sequence[str],
        radius_m: float = 80.0,
        include_lane_connectors: bool = True,
    ) -> None:
        self._map_api = map_api
        self._route_roadblock_ids = list(route_roadblock_ids)
        self._radius_m = float(radius_m)
        self._include_lane_connectors = include_lane_connectors

        # Lazy nuplan-devkit import: only needed when we actually query the map.
        from nuplan.common.maps.maps_datatypes import SemanticMapLayer  # type: ignore

        self._SL = SemanticMapLayer
        self._route_lane_ids: set[str] = self._collect_route_lane_ids()

    def _collect_route_lane_ids(self) -> set[str]:
        """Up-front: pull every lane edge inside the route corridor into a set.

        Used at tick-time to label each proximal lane as ``in_route_corridor``
        without a second graph traversal.
        """
        lane_ids: set[str] = set()
        if self._map_api is None:
            return lane_ids
        for rb_id in self._route_roadblock_ids:
            block = self._map_api.get_map_object(rb_id, self._SL.ROADBLOCK)
            if block is None:
                block = self._map_api.get_map_object(rb_id, self._SL.ROADBLOCK_CONNECTOR)
            if block is None:
                continue
            for edge in getattr(block, "interior_edges", []) or []:
                edge_id = getattr(edge, "id", None)
                if edge_id is not None:
                    lane_ids.add(str(edge_id))
        return lane_ids

    def view(
        self,
        anchor_xy_world: np.ndarray,
        anchor_heading_world: float,
        traffic_light_data: Optional[Iterable[Any]] = None,
    ) -> MapHorizonView:
        """Build a per-tick :class:`MapHorizonView` in ego-local frame.

        Parameters
        ----------
        anchor_xy_world:
            ``(x, y)`` position of the ego rear axle in world coordinates. The
            local frame is defined as the rotation+translation that places this
            point at the origin and aligns the ego heading with ``+x``.
        anchor_heading_world:
            Ego heading in world frame, in radians.
        traffic_light_data:
            Optional iterable of nuPlan's ``TrafficLightStatusData`` (or
            compatible duck-typed objects exposing ``lane_connector_id`` and
            ``status``). Used to populate :attr:`MapHorizonView.traffic_lights`.
        """
        from nuplan.common.actor_state.state_representation import Point2D  # type: ignore

        point = Point2D(float(anchor_xy_world[0]), float(anchor_xy_world[1]))
        requested = [
            self._SL.LANE,
            self._SL.INTERSECTION,
            self._SL.STOP_LINE,
            self._SL.CROSSWALK,
            self._SL.WALKWAYS,
            self._SL.DRIVABLE_AREA,
        ]
        if self._include_lane_connectors:
            requested.append(self._SL.LANE_CONNECTOR)

        # NuPlanMap exposes only a subset of layers as queryable map objects;
        # DRIVABLE_AREA in particular is raster-only on most map releases. Filter
        # to whatever the map advertises.
        try:
            available = set(self._map_api.get_available_map_objects())
        except Exception:
            available = set(requested)
        layers = [layer for layer in requested if layer in available]
        proximal = self._map_api.get_proximal_map_objects(point, self._radius_m, layers)

        anchor = np.asarray(anchor_xy_world, dtype=np.float64).reshape(2)
        cos_h, sin_h = math.cos(-anchor_heading_world), math.sin(-anchor_heading_world)
        rot = np.array([[cos_h, -sin_h], [sin_h, cos_h]])

        def to_local(xy: np.ndarray) -> np.ndarray:
            return (rot @ (xy.T - anchor[:, None])).T

        # Lanes — split into route / non-route bins.
        route_lanes: List[LocalLane] = []
        nonroute_lanes: List[LocalLane] = []
        for lane in proximal.get(self._SL.LANE, []):
            local = self._convert_lane(lane, to_local, is_lane_connector=False)
            if local is None:
                continue
            (route_lanes if local.in_route_corridor else nonroute_lanes).append(local)

        lane_connectors: List[LocalLane] = []
        if self._include_lane_connectors:
            for lc in proximal.get(self._SL.LANE_CONNECTOR, []):
                local = self._convert_lane(lc, to_local, is_lane_connector=True)
                if local is not None:
                    lane_connectors.append(local)

        # Drivable: use union of lane polygons as our drivable surface (NuPlan's
        # DRIVABLE_AREA is raster-only and not directly queryable).
        drivable: List[LocalPolygon] = []
        for lane in proximal.get(self._SL.LANE, []) + (
            proximal.get(self._SL.LANE_CONNECTOR, []) if self._include_lane_connectors else []
        ):
            poly = self._polygon_to_local(lane, to_local)
            if poly is not None:
                drivable.append(poly)

        walkways = [
            p for p in (self._polygon_to_local(o, to_local) for o in proximal.get(self._SL.WALKWAYS, []))
            if p is not None
        ]
        crosswalks = [
            p for p in (self._polygon_to_local(o, to_local) for o in proximal.get(self._SL.CROSSWALK, []))
            if p is not None
        ]
        intersections = [
            p for p in (self._polygon_to_local(o, to_local) for o in proximal.get(self._SL.INTERSECTION, []))
            if p is not None
        ]

        stop_lines: List[LocalStopLine] = []
        for sl in proximal.get(self._SL.STOP_LINE, []):
            converted = self._convert_stop_line(sl, to_local)
            if converted is not None:
                stop_lines.append(converted)

        traffic_lights = self._convert_traffic_lights(traffic_light_data)

        return MapHorizonView(
            route_lanes=tuple(route_lanes),
            nonroute_lanes=tuple(nonroute_lanes),
            lane_connectors=tuple(lane_connectors),
            drivable_polygons=tuple(drivable),
            walkways=tuple(walkways),
            bike_lanes=tuple(),  # NuPlan does not currently expose dedicated bike-lane semantics.
            crosswalks=tuple(crosswalks),
            intersections=tuple(intersections),
            stop_lines=tuple(stop_lines),
            traffic_lights=traffic_lights,
            anchor_xy_world=anchor,
            anchor_heading_world=float(anchor_heading_world),
        )

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------

    def _convert_lane(self, lane: Any, to_local, is_lane_connector: bool) -> Optional[LocalLane]:
        baseline = getattr(lane, "baseline_path", None)
        discrete = getattr(baseline, "discrete_path", None) if baseline is not None else None
        if not discrete:
            return None
        xy_world = np.array([[s.x, s.y] for s in discrete], dtype=np.float64)
        if xy_world.shape[0] < 2:
            return None
        xy_local = to_local(xy_world)
        # Per-vertex headings via finite differences of consecutive segments.
        deltas = np.diff(xy_local, axis=0)
        seg_psi = np.arctan2(deltas[:, 1], deltas[:, 0])
        per_vertex = np.empty(xy_local.shape[0])
        per_vertex[0] = seg_psi[0]
        per_vertex[-1] = seg_psi[-1]
        if seg_psi.shape[0] > 1:
            per_vertex[1:-1] = 0.5 * (seg_psi[:-1] + seg_psi[1:])
        # Polygon to extract width if available; otherwise default 3.5 m.
        polygon = getattr(lane, "polygon", None)
        width = 3.5
        if polygon is not None:
            try:
                width = float(polygon.bounds[3] - polygon.bounds[1])
                # Approximate width: assume rectangular bounding box of the lane
                # polygon when projected onto the centerline-perpendicular axis.
                # For curved lanes this overestimates; for tracking-style use
                # this is adequate, refined further in the rule encoder.
                width = max(2.5, min(width, 6.0))
            except Exception:
                pass
        lane_id = str(getattr(lane, "id", "unknown"))
        return LocalLane(
            lane_id=lane_id,
            centerline_xy=xy_local,
            headings=per_vertex,
            width_m=width,
            speed_limit_mps=getattr(lane, "speed_limit_mps", None),
            is_lane_connector=is_lane_connector,
            in_route_corridor=lane_id in self._route_lane_ids,
            incoming_ids=tuple(str(getattr(l, "id", l)) for l in (getattr(lane, "incoming_edges", []) or [])),
            outgoing_ids=tuple(str(getattr(l, "id", l)) for l in (getattr(lane, "outgoing_edges", []) or [])),
        )

    def _polygon_to_local(self, obj: Any, to_local) -> Optional[LocalPolygon]:
        polygon = getattr(obj, "polygon", None)
        if polygon is None:
            return None
        try:
            exterior = polygon.exterior.coords
        except Exception:
            return None
        xy_world = np.array([(c[0], c[1]) for c in exterior], dtype=np.float64)
        if xy_world.shape[0] < 3:
            return None
        xy_local = to_local(xy_world)
        half_planes = _polygon_to_half_planes(xy_local, max_planes=H_PER_POLY)
        if not half_planes:
            return None
        closest = float(min(_distance_from_origin_to_polygon(xy_local), 1e6))
        return LocalPolygon(
            polygon_id=str(getattr(obj, "id", "polygon")),
            half_planes=tuple(half_planes),
            closest_distance_m=closest,
        )

    def _convert_stop_line(self, sl: Any, to_local) -> Optional[LocalStopLine]:
        polygon = getattr(sl, "polygon", None)
        if polygon is None:
            return None
        try:
            xy_world = np.array(list(polygon.exterior.coords), dtype=np.float64)
        except Exception:
            return None
        if xy_world.shape[0] < 2:
            return None
        xy_local = to_local(xy_world)
        stop_type_name = getattr(sl, "stop_line_type", None)
        stype = "GENERIC"
        if stop_type_name is not None:
            n = getattr(stop_type_name, "name", str(stop_type_name)).upper()
            if "TRAFFIC" in n or "LIGHT" in n:
                stype = "TRAFFIC_LIGHT"
            elif "STOP" in n:
                stype = "STOP_SIGN"
            elif "YIELD" in n:
                stype = "YIELD_SIGN"
        # Try to associate stop line with a lane connector via the map's
        # cross-references. NuPlanMap exposes this on some releases via the
        # ``associated_lane_connector_id`` attribute; we fall back to None
        # otherwise and the traffic-light rule then matches on geometric
        # proximity at constraint-encoding time.
        associated = getattr(sl, "associated_lane_connector_id", None)
        return LocalStopLine(
            stop_line_id=str(getattr(sl, "id", "stop_line")),
            polyline_xy=xy_local,
            associated_connector_id=str(associated) if associated is not None else None,
            stop_type=stype,
        )

    def _convert_traffic_lights(self, data: Optional[Iterable[Any]]) -> Tuple[LocalTrafficLight, ...]:
        if data is None:
            return ()
        lights: List[LocalTrafficLight] = []
        for tl in data:
            try:
                connector_id = str(getattr(tl, "lane_connector_id"))
                status = getattr(tl, "status", None)
                if hasattr(status, "name"):
                    state = str(status.name)
                else:
                    state = str(status).upper()
                lights.append(LocalTrafficLight(connector_id=connector_id, state=state))
            except Exception as exc:
                logger.debug("MapLifter: skipping malformed traffic-light record %s (%s)", tl, exc)
        return tuple(lights)


# ----------------------------------------------------------------------
# Half-plane reduction helpers
# ----------------------------------------------------------------------


def _polygon_to_half_planes(xy_local: np.ndarray, max_planes: int) -> List[HalfPlane]:
    """Reduce a polygon (in local frame) to at most ``max_planes`` outward half-planes.

    Each polygon edge ``(p_i, p_{i+1})`` yields a half-plane
    ``n_x · x + n_y · y <= d`` where ``(n_x, n_y)`` is the outward normal
    (right-hand side of the edge when traversed in vertex order) and ``d`` is
    the offset. Edges with the smallest ``d`` (i.e., whose supporting half-plane
    passes closest to the local origin) are kept; the rest are discarded.

    For convex polygons this yields an exact under-approximation of the polygon
    interior using the ``max_planes`` most-binding edges; for non-convex
    polygons the result is the convex-hull edges, which is a conservative
    over-approximation of the polygon (the constraint set the MPC sees is
    *smaller* than the actual polygon, which is the right direction for safety
    constraints like 'stay outside this region').
    """
    if xy_local.shape[0] < 3:
        return []
    # If the polygon is closed (first==last), drop the duplicate.
    pts = xy_local
    if np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    n_edges = pts.shape[0]
    halves: List[Tuple[float, HalfPlane]] = []
    # Determine winding direction (CCW => signed area > 0); we want OUTWARD
    # normals, so we flip when the winding is CCW (interior is on the left of
    # each edge => outward normal points right). For CW winding, outward
    # normal points left.
    signed_area = 0.0
    for i in range(n_edges):
        j = (i + 1) % n_edges
        signed_area += pts[i, 0] * pts[j, 1] - pts[j, 0] * pts[i, 1]
    ccw = signed_area > 0
    for i in range(n_edges):
        j = (i + 1) % n_edges
        dx = pts[j, 0] - pts[i, 0]
        dy = pts[j, 1] - pts[i, 1]
        length = math.hypot(dx, dy)
        if length < 1e-9:
            continue
        # Right-perpendicular gives outward normal for CCW winding;
        # left-perpendicular gives outward normal for CW winding.
        if ccw:
            nx, ny = dy / length, -dx / length
        else:
            nx, ny = -dy / length, dx / length
        d = nx * pts[i, 0] + ny * pts[i, 1]
        halves.append((d, HalfPlane(n_x=float(nx), n_y=float(ny), d=float(d))))
    if not halves:
        return []
    # Keep the planes with smallest signed offset from the local origin (i.e.,
    # the ones currently binding or near-binding for an ego at origin).
    halves.sort(key=lambda hd: hd[0])
    return [hp for _, hp in halves[:max_planes]]


def _distance_from_origin_to_polygon(xy_local: np.ndarray) -> float:
    """Closest distance from the local origin to the polygon's boundary.

    Used only as a metadata hint for the rule encoder (so it can prioritise
    which polygons matter the most at this tick); not used for the constraint
    formulation itself.
    """
    pts = xy_local
    if pts.shape[0] < 2:
        return 1e6
    best = float("inf")
    for i in range(pts.shape[0]):
        j = (i + 1) % pts.shape[0]
        ax, ay = pts[i]
        bx, by = pts[j]
        dx, dy = bx - ax, by - ay
        seg_len2 = dx * dx + dy * dy
        if seg_len2 <= 0.0:
            best = min(best, math.hypot(ax, ay))
            continue
        t = -((ax * dx + ay * dy) / seg_len2)
        t = max(0.0, min(1.0, t))
        cx = ax + t * dx
        cy = ay + t * dy
        best = min(best, math.hypot(cx, cy))
    return best
