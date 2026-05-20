"""Dynamic-safety rules for spacing.

- 3r3 — Maintain safe following headway (THW & TTC).
- 3r5 — Maintain lateral clearance.
- 3r6 — Manage lane intrusions from adjacent vehicles.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Tuple

from ..rule import ObserverRule
from ..types import SceneSnapshot
from ._common import lateral_neighbors, lead_agent_in_lane, relative_velocity


class SafeHeadwayRule(ObserverRule):
    id = "3r3"
    level = 3
    name = "Maintain safe following headway"
    description = (
        "Penalises shortfalls in time headway (THW) and time-to-collision "
        "(TTC) below safe minimums when an in-lane lead vehicle is present."
    )

    def __init__(
        self,
        min_thw_s: float = 1.5,
        min_ttc_s: float = 3.0,
        min_distance_m: float = 2.0,
    ):
        self.min_thw_s = min_thw_s
        self.min_ttc_s = min_ttc_s
        self.min_distance_m = min_distance_m

    def applies_at(self, snap: SceneSnapshot) -> Tuple[bool, Mapping[str, Any]]:
        lead, lon, lane = lead_agent_in_lane(snap)
        if lead is None:
            return False, {}
        return True, {"lead_track_id": lead.track_id, "longitudinal_distance_m": lon}

    def violation_at(self, snap: SceneSnapshot) -> Tuple[float, Mapping[str, Any]]:
        lead, lon, lane = lead_agent_in_lane(snap)
        if lead is None or lon is None:
            return 0.0, {}
        v = max(snap.ego.speed, 1e-3)
        # Use bumper-to-bumper distance.
        gap = max(0.0, lon - (snap.ego.length + lead.length) / 2.0)
        thw = gap / v
        rel_lon, _ = relative_velocity(snap.ego, lead)
        closing = -rel_lon  # positive when ego is closing on lead
        ttc = gap / closing if closing > 0.1 else math.inf
        thw_shortfall = max(0.0, self.min_thw_s - thw)
        ttc_shortfall = max(0.0, self.min_ttc_s - ttc) if ttc != math.inf else 0.0
        too_close = max(0.0, self.min_distance_m - gap) * 2.0
        rate = thw_shortfall + ttc_shortfall + too_close
        return rate, {
            "lead_track_id": lead.track_id,
            "gap_m": gap,
            "thw_s": thw,
            "ttc_s": None if ttc == math.inf else ttc,
            "closing_speed_mps": closing,
        }


class LateralClearanceRule(ObserverRule):
    id = "3r5"
    level = 3
    name = "Maintain lateral clearance"
    description = (
        "Penalises lateral distance to adjacent agents below a dynamic safe "
        "minimum that grows with relative lateral velocity."
    )

    def __init__(self, min_lateral_m: float = 1.0, vrel_coef_s: float = 0.5):
        self.min_lateral_m = min_lateral_m
        self.vrel_coef_s = vrel_coef_s

    def applies_at(self, snap: SceneSnapshot) -> Tuple[bool, Mapping[str, Any]]:
        neighbors = lateral_neighbors(snap, max_long_m=8.0, lateral_band_m=4.0)
        return bool(neighbors), {"n_neighbors": len(neighbors)}

    def violation_at(self, snap: SceneSnapshot) -> Tuple[float, Mapping[str, Any]]:
        worst = 0.0
        worst_track = None
        for agent, lat, lon in lateral_neighbors(snap, max_long_m=8.0, lateral_band_m=4.0):
            # Bumper-to-bumper lateral gap.
            lat_gap = max(0.0, abs(lat) - (snap.ego.width + agent.width) / 2.0)
            _, lat_rel = relative_velocity(snap.ego, agent)
            v_close = max(0.0, -lat_rel if lat > 0 else lat_rel)
            d_safe = self.min_lateral_m + self.vrel_coef_s * v_close
            shortfall = max(0.0, d_safe - lat_gap)
            if shortfall > worst:
                worst = shortfall
                worst_track = agent.track_id
        return worst, {"worst_track_id": worst_track, "min_lateral_threshold_m": self.min_lateral_m}


class LaneIntrusionRule(ObserverRule):
    id = "3r6"
    level = 3
    name = "Manage lane intrusions from adjacent vehicles"
    description = (
        "Penalises low lateral time-to-collision with adjacent vehicles, "
        "encouraging early gap creation."
    )

    def __init__(self, min_lat_ttc_s: float = 2.0):
        self.min_lat_ttc_s = min_lat_ttc_s

    def applies_at(self, snap: SceneSnapshot) -> Tuple[bool, Mapping[str, Any]]:
        # Any neighbor moving laterally toward the ego.
        for agent, lat, _ in lateral_neighbors(snap, max_long_m=10.0, lateral_band_m=5.0):
            _, lat_rel = relative_velocity(snap.ego, agent)
            v_close = -lat_rel if lat > 0 else lat_rel
            if v_close > 0.2:
                return True, {"intruding_track_id": agent.track_id}
        return False, {}

    def violation_at(self, snap: SceneSnapshot) -> Tuple[float, Mapping[str, Any]]:
        worst = 0.0
        worst_track = None
        for agent, lat, _ in lateral_neighbors(snap, max_long_m=10.0, lateral_band_m=5.0):
            lat_gap = max(0.0, abs(lat) - (snap.ego.width + agent.width) / 2.0)
            _, lat_rel = relative_velocity(snap.ego, agent)
            v_close = -lat_rel if lat > 0 else lat_rel
            if v_close <= 0.05:
                continue
            ttc = lat_gap / v_close
            shortfall = max(0.0, self.min_lat_ttc_s - ttc)
            if shortfall > worst:
                worst = shortfall
                worst_track = agent.track_id
        return worst, {"worst_track_id": worst_track, "min_lat_ttc_s": self.min_lat_ttc_s}
