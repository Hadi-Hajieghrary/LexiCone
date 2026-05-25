"""Adapter from a NuPlan :class:`SimulationLog` to :class:`SceneSnapshot`.

A simulation log is the artefact produced by nuPlan's closed-loop simulation
pipeline (``SimulationLogCallback``). It contains:

- ``simulation_log.scenario`` — the :class:`AbstractScenario` the planner was
  driven over (used here for the map and recorded traffic lights);
- ``simulation_log.simulation_history.data`` — one
  :class:`SimulationHistorySample` per simulator iteration, with
  ``ego_state`` (the planner's chosen ego state), ``observation``
  (:class:`DetectionsTracks`), ``trajectory`` (the planner's planned future),
  and ``iteration`` (timestamp + index).

This module reuses the helper functions in :mod:`nuplan_adapter` for the
mechanical conversions (ego → :class:`EgoSnapshot`, detections →
:class:`AgentSnapshot`, ``map_api`` proximal query → :class:`MapSnapshot`).

Usage::

    from lexicone.observer.simulation_log_adapter import NuPlanSimulationLogSource
    source = NuPlanSimulationLogSource.from_path(Path("/path/to/log.msgpack.xz"), radius_m=80.0)
    engine = RuleEngine()
    for snap in source:
        engine.step(snap)
    print(engine.summary())

The nuplan-devkit import is lazy — importing this module does not require
nuplan to be installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Iterator, List, Optional

from .nuplan_adapter import (
    _detections_to_agents,
    _ego_to_snapshot,
    _map_to_snapshot,
    _tls_to_status,
)
from .types import EgoSnapshot, Pose2D, SceneSnapshot


class NuPlanSimulationLogSource:
    """Iterate :class:`SceneSnapshot`s from a nuPlan ``SimulationLog``.

    Parameters
    ----------
    simulation_log:
        A pre-loaded ``SimulationLog``. Use :meth:`from_path` to load from
        disk.
    radius_m:
        Region-of-interest radius around the ego for proximal map queries.
    route_lane_ids:
        Optional planner route (constant across the log). If left ``None``,
        the source tries ``scenario.get_route_roadblock_ids()`` and uses
        whatever it returns.
    include_lane_connectors:
        Pass-through to the map adapter.
    """

    def __init__(
        self,
        simulation_log: Any,
        *,
        radius_m: float = 80.0,
        route_lane_ids: Optional[Iterable[str]] = None,
        include_lane_connectors: bool = True,
    ) -> None:
        self.simulation_log = simulation_log
        self.radius_m = float(radius_m)
        self.include_lane_connectors = include_lane_connectors

        # Lazy nuplan import (just the SemanticMapLayer enum we need below).
        from nuplan.common.maps.maps_datatypes import SemanticMapLayer  # type: ignore

        self._SL = SemanticMapLayer

        if route_lane_ids is not None:
            self._route_lane_ids: Optional[List[str]] = [str(x) for x in route_lane_ids]
        else:
            try:
                ids = simulation_log.scenario.get_route_roadblock_ids() or []
                self._route_lane_ids = [str(x) for x in ids] if ids else None
            except Exception:
                self._route_lane_ids = None

    @classmethod
    def from_path(
        cls,
        log_path: Path,
        *,
        radius_m: float = 80.0,
        route_lane_ids: Optional[Iterable[str]] = None,
        include_lane_connectors: bool = True,
    ) -> "NuPlanSimulationLogSource":
        """Load a saved ``SimulationLog`` and return a source iterator."""
        from nuplan.planning.simulation.simulation_log import SimulationLog  # type: ignore

        log = SimulationLog.load_data(log_path)
        return cls(
            log,
            radius_m=radius_m,
            route_lane_ids=route_lane_ids,
            include_lane_connectors=include_lane_connectors,
        )

    # ----- iteration -----

    def __iter__(self) -> Iterator[SceneSnapshot]:
        for sample in self.simulation_log.simulation_history.data:
            yield self.snapshot_of(sample)

    def __len__(self) -> int:
        return len(self.simulation_log.simulation_history.data)

    def snapshot_of(self, sample: Any) -> SceneSnapshot:
        """Convert one ``SimulationHistorySample`` to a :class:`SceneSnapshot`."""
        scenario = self.simulation_log.scenario
        ego = _ego_to_snapshot(sample.ego_state)
        agents = _detections_to_agents(sample.observation)

        map_api = scenario.map_api
        map_snap = _map_to_snapshot(map_api, ego, self.radius_m, self._SL, self.include_lane_connectors)

        # Traffic-light status: try the sample first (some simulations attach it),
        # otherwise fall back to the scenario at this iteration index.
        tls = None
        for attr in ("traffic_light_status", "traffic_light_data"):
            tls = getattr(sample, attr, None)
            if tls is not None:
                break
        if tls is None:
            iter_idx = getattr(getattr(sample, "iteration", None), "index", None)
            if iter_idx is not None:
                try:
                    tls = scenario.get_traffic_light_status_at_iteration(int(iter_idx))
                except Exception:
                    tls = None
        traffic_lights = _tls_to_status(tls)

        planned_trajectory = _planned_trajectory(sample)

        timestamp_us = int(getattr(sample.ego_state.time_point, "time_us", 0))

        return SceneSnapshot(
            timestamp_us=timestamp_us,
            ego=ego,
            agents=agents,
            map=map_snap,
            traffic_lights=traffic_lights,
            planned_trajectory=planned_trajectory,
            route_lane_ids=self._route_lane_ids,
        )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _planned_trajectory(sample: Any) -> Optional[List[EgoSnapshot]]:
    """Extract the planner's planned future as a list of :class:`EgoSnapshot`.

    Returns ``None`` if the sample has no trajectory or it has no states.
    """
    trajectory = getattr(sample, "trajectory", None)
    if trajectory is None:
        return None
    try:
        states = trajectory.get_sampled_trajectory()
    except Exception:
        return None
    if not states:
        return None
    planned: List[EgoSnapshot] = []
    for state in states:
        center = getattr(state, "center", None) or getattr(state, "rear_axle", None)
        if center is None:
            continue
        time_us = int(getattr(getattr(state, "time_point", None), "time_us", 0))
        planned.append(
            EgoSnapshot(
                timestamp_us=time_us,
                pose=Pose2D(x=float(center.x), y=float(center.y), heading=float(center.heading)),
                vx=0.0,
                vy=0.0,
                ax=0.0,
                ay=0.0,
                yaw_rate=0.0,
                length=4.7,
                width=1.85,
                rear_axle_to_center=0.0,
                pose_at_center=True,
            )
        )
    return planned if planned else None
