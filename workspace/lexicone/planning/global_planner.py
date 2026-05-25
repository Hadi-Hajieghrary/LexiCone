"""Global route planner.

Periodically extracts a route along the lane graph from the current ego pose to
the scenario's goal roadblock and converts it into a continuous reference path
for the trajectory planner. The strategy mirrors
``nuplan.planning.simulation.planner.idm_planner.IDMPlanner._initialize_ego_path``:

1. Resolve the scenario's ``route_roadblock_ids`` into roadblock map objects.
2. Find the starting lane edge that contains (or is nearest to) the ego.
3. Run breadth-first search over lane successors restricted to lanes inside the
   route corridor.
4. Concatenate the baseline-path samples of the resulting edges, trim the prefix
   that lies behind the ego, and return only the leading ``lookahead_m`` window.
"""

from __future__ import annotations

import logging
import math
from typing import List, Optional

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.abstract_map_objects import LaneGraphEdgeMapObject, RoadBlockGraphEdgeMapObject
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.planning.simulation.planner.utils.breadth_first_search import BreadthFirstSearch

from .reference_path import ReferencePath, reference_from_se2_polyline

logger = logging.getLogger(__name__)


class GlobalRoutePlanner:
    """Stateless per-call route extractor."""

    def __init__(
        self,
        map_api: AbstractMap,
        route_roadblock_ids: List[str],
        mission_goal: Optional[StateSE2],
        default_speed_limit_mps: float,
        lookahead_m: float = 200.0,
    ):
        self._map_api = map_api
        self._mission_goal = mission_goal
        self._default_speed_limit_mps = default_speed_limit_mps
        self._lookahead_m = lookahead_m
        self._route_roadblocks: List[RoadBlockGraphEdgeMapObject] = []
        self._candidate_lane_edge_ids: List[str] = []
        self._load_route(route_roadblock_ids)

    def _load_route(self, route_roadblock_ids: List[str]) -> None:
        for id_ in route_roadblock_ids:
            block = self._map_api.get_map_object(id_, SemanticMapLayer.ROADBLOCK)
            if block is None:
                block = self._map_api.get_map_object(id_, SemanticMapLayer.ROADBLOCK_CONNECTOR)
            if block is not None:
                self._route_roadblocks.append(block)
        if not self._route_roadblocks:
            raise RuntimeError("GlobalRoutePlanner: route_roadblock_ids resolved to zero map objects")
        self._candidate_lane_edge_ids = [
            edge.id for block in self._route_roadblocks for edge in block.interior_edges
        ]

    def plan(self, ego_state: EgoState) -> Optional[ReferencePath]:
        """Return a reference path for the leading ``lookahead_m`` of the route, or None on failure."""
        starting_edge = self._get_starting_edge(ego_state)
        if starting_edge is None:
            logger.warning("GlobalRoutePlanner: failed to find a starting lane edge for ego")
            return None

        route_edges, path_found = self._search(starting_edge)
        if not path_found:
            logger.warning(
                "GlobalRoutePlanner: BFS did not reach the goal roadblock; using longest partial route"
            )
        if not route_edges:
            return None

        # Concatenate baseline path samples and per-vertex speed limits.
        discrete: List[StateSE2] = []
        speed_limits: List[Optional[float]] = []
        for edge in route_edges:
            samples = list(edge.baseline_path.discrete_path)
            v = edge.speed_limit_mps
            for s in samples:
                # Drop near-duplicate join vertices between consecutive edges.
                if discrete and math.hypot(s.x - discrete[-1].x, s.y - discrete[-1].y) < 1e-3:
                    continue
                discrete.append(s)
                speed_limits.append(v)

        if len(discrete) < 2:
            return None

        # Trim everything before the ego's projection so the reference starts at the car.
        rear = ego_state.rear_axle
        ref_full = reference_from_se2_polyline(
            discrete, speed_limits, self._default_speed_limit_mps
        )
        s_ego, _ = ref_full.project(rear.point)
        end_s = min(ref_full.length, s_ego + self._lookahead_m)

        # If the ego is already at the end of the available route the next replan should pick up
        # additional successors; for now just return what we have so the MPC can coast.
        if end_s - s_ego < 1.0:
            return ref_full

        return reference_from_se2_polyline(
            discrete,
            speed_limits,
            self._default_speed_limit_mps,
            start_s=s_ego,
            end_s=end_s,
        )

    def _get_starting_edge(self, ego_state: EgoState) -> Optional[LaneGraphEdgeMapObject]:
        """Find the lane edge the ego currently occupies (or is closest to).

        Unlike :class:`~nuplan.planning.simulation.planner.idm_planner.IDMPlanner`,
        which only ever runs this at init time, we re-plan periodically — so we
        search **every** roadblock in the route, not just the first two. As the
        ego progresses past the early roadblocks we still find a valid starting
        edge ahead in the corridor.
        """
        # Prefer an edge that actually contains the ego: that's an unambiguous, lane-aware match.
        for block in self._route_roadblocks:
            for edge in block.interior_edges:
                if edge.contains_point(ego_state.center):
                    return edge
        # Fall back to the nearest edge to the ego's footprint.
        starting_edge: Optional[LaneGraphEdgeMapObject] = None
        closest_distance = math.inf
        for block in self._route_roadblocks:
            for edge in block.interior_edges:
                distance = edge.polygon.distance(ego_state.car_footprint.geometry)
                if distance < closest_distance:
                    starting_edge = edge
                    closest_distance = distance
        return starting_edge

    def _search(self, starting_edge: LaneGraphEdgeMapObject):
        graph_search = BreadthFirstSearch(starting_edge, self._candidate_lane_edge_ids)
        # BFS depth = how many roadblocks remain between the starting edge and the goal.
        starting_block_id = starting_edge.get_roadblock_id()
        try:
            start_index = next(
                i for i, block in enumerate(self._route_roadblocks) if block.id == starting_block_id
            )
        except StopIteration:
            start_index = 0
        remaining = len(self._route_roadblocks) - start_index
        target_depth = max(1, remaining)
        return graph_search.search(self._route_roadblocks[-1], target_depth)
