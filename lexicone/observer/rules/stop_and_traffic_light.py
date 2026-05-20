"""Regulatory stop and signal compliance rules.

- 8r0 — Comply with mandatory stops (stop signs, red-before-turn).
- 7r1 — Obey traffic-light states inside intersections.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional, Tuple

from ..geometry import ego_footprint, find_ego_lane, polygon_from_points
from ..rule import ObserverRule
from ..types import SceneSnapshot, StopType, TrafficLightState
from ._common import (
    ego_in_intersection,
    red_or_yellow_for_lane,
    stop_line_for_ego,
)


class MandatoryStopRule(ObserverRule):
    id = "8r0"
    level = 8
    name = "Comply with mandatory stops"
    description = (
        "At a stop sign or red-light controlled stop line, the ego must come "
        "to a complete stop at/before the line. Penalises: (i) crossing while "
        "moving above stop tolerance, (ii) failing to stop within a short "
        "approach window."
    )

    def __init__(self, stop_speed_mps: float = 0.3, approach_dist_m: float = 8.0):
        self.stop_speed_mps = stop_speed_mps
        self.approach_dist_m = approach_dist_m
        # Track per-stop-line minimum speed seen during approach (id -> v_min).
        self._approach_state: dict[str, float] = {}
        self._last_stop_id: Optional[str] = None

    def _relevant_stop_line(self, snap: SceneSnapshot):
        ego_lane = find_ego_lane(
            snap.ego, list(snap.map.lanes) + list(snap.map.lane_connectors)
        )
        if ego_lane is None:
            return None, None
        result = stop_line_for_ego(snap, ego_lane)
        if result is None:
            return ego_lane, None
        return ego_lane, result

    def applies_at(self, snap: SceneSnapshot) -> Tuple[bool, Mapping[str, Any]]:
        ego_lane, result = self._relevant_stop_line(snap)
        if result is None:
            return False, {}
        poly, signed_dist = result
        # Identify the controlling stop_line (by associated lane or proximity).
        stop_id = None
        controlling_type: Optional[str] = None
        for sl in snap.map.stop_lines:
            if sl.associated_lane_id == ego_lane.lane_id:
                stop_id = sl.stop_line_id
                controlling_type = (
                    sl.stop_type.value if isinstance(sl.stop_type, StopType) else sl.stop_type
                )
                break
        # Rule only applies for STOP_SIGN. TL-controlled stop is handled by 7r1.
        if controlling_type not in (StopType.STOP_SIGN.value, StopType.GENERIC.value):
            return False, {"stop_type": controlling_type}
        if abs(signed_dist) > self.approach_dist_m:
            return False, {"distance_to_stop_m": signed_dist}
        return True, {
            "stop_line_id": stop_id,
            "distance_to_stop_m": signed_dist,
            "stop_type": controlling_type,
        }

    def violation_at(self, snap: SceneSnapshot) -> Tuple[float, Mapping[str, Any]]:
        ego_lane, result = self._relevant_stop_line(snap)
        if result is None:
            return 0.0, {}
        poly, signed_dist = result
        fp = ego_footprint(snap.ego)
        # Crossing the stop polygon while moving above tolerance is a violation.
        v = snap.ego.speed
        rate = 0.0
        details: dict[str, Any] = {"distance_to_stop_m": signed_dist, "ego_speed_mps": v}
        if fp.intersects(poly) and v > self.stop_speed_mps:
            rate = max(rate, v - self.stop_speed_mps)
            details["overrunning_stop_line"] = True
        # Track minimum speed seen during approach to flag "rolling stops"
        # after the ego has crossed (signed_dist < 0).
        # Use stop_line id (or lane id) as a key for state tracking.
        stop_id = ego_lane.lane_id
        prev_min = self._approach_state.get(stop_id, math.inf)
        new_min = min(prev_min, v)
        self._approach_state[stop_id] = new_min
        if signed_dist < -1.0 and prev_min == math.inf:
            # Crossed without ever entering approach — already accounted for above.
            pass
        if signed_dist < -1.0 and new_min > self.stop_speed_mps:
            # Past the line but never came to a stop.
            rate = max(rate, new_min - self.stop_speed_mps)
            details["min_approach_speed_mps"] = new_min
            details["rolling_stop"] = True
        return rate, details


class TrafficLightComplianceRule(ObserverRule):
    id = "7r1"
    level = 7
    name = "Obey traffic-light states in intersections"
    description = (
        "Counts time inside (or entering) an intersection while the associated "
        "lane-connector traffic light is RED (or YELLOW with safe stopping "
        "feasible)."
    )

    def __init__(self, safe_stop_decel_mps2: float = 3.0):
        self.safe_stop_decel_mps2 = safe_stop_decel_mps2

    def _relevant_connector(self, snap: SceneSnapshot):
        # Use the lane connector containing the ego, or the next one along the
        # planned route. Fallback: any lane connector whose polygon overlaps the
        # ego footprint.
        from ..geometry import ego_footprint as _ef

        fp = _ef(snap.ego)
        for lc in snap.map.lane_connectors:
            poly = polygon_from_points(lc.polygon)
            if poly is not None and fp.intersects(poly):
                return lc
        return None

    def applies_at(self, snap: SceneSnapshot) -> Tuple[bool, Mapping[str, Any]]:
        lc = self._relevant_connector(snap)
        if lc is None:
            return False, {}
        active, state = red_or_yellow_for_lane(snap, lc.lane_id)
        if not active:
            return False, {"lc_id": lc.lane_id, "state": state}
        return True, {"lc_id": lc.lane_id, "state": state}

    def violation_at(self, snap: SceneSnapshot) -> Tuple[float, Mapping[str, Any]]:
        lc = self._relevant_connector(snap)
        if lc is None:
            return 0.0, {}
        active, state = red_or_yellow_for_lane(snap, lc.lane_id)
        in_int, _ = ego_in_intersection(snap)
        v = snap.ego.speed
        rate = 0.0
        details: dict[str, Any] = {"state": state, "in_intersection": in_int, "ego_speed_mps": v}
        if state == TrafficLightState.RED.value and in_int and v > 0.3:
            # Violation = dwell-speed in intersection while red.
            rate = v
            details["red_in_intersection"] = True
        elif state == TrafficLightState.YELLOW.value and not in_int and v > 0.3:
            # Yellow-light violation only if a safe stop was feasible:
            # v^2 / (2 * a_safe) <= distance-to-stop. We approximate distance
            # using the nearest stop line; if none is associated, skip.
            ego_lane = find_ego_lane(snap.ego, list(snap.map.lanes) + list(snap.map.lane_connectors))
            sl = stop_line_for_ego(snap, ego_lane) if ego_lane is not None else None
            if sl is not None:
                _, d = sl
                d_to_stop = d
                if d_to_stop > 0:
                    stop_dist = (v * v) / (2.0 * self.safe_stop_decel_mps2)
                    if stop_dist <= d_to_stop:
                        rate = v - 0.3
                        details["yellow_runnable"] = True
        return max(0.0, rate), details
