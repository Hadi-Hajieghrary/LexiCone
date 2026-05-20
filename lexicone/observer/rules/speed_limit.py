"""3r0 — Obey posted speed limits."""

from __future__ import annotations

from typing import Any, Mapping, Tuple

from ..rule import ObserverRule
from ..types import SceneSnapshot
from ._common import speed_limit_for_ego


class SpeedLimitRule(ObserverRule):
    id = "3r0"
    level = 3
    name = "Obey posted speed limits"
    description = (
        "Penalises (v - v_lim)^2 whenever the ego speed exceeds the lane's "
        "posted speed limit plus a tolerance."
    )

    def __init__(self, tolerance_mps: float = 1.0, exponent: float = 2.0):
        self.tolerance_mps = tolerance_mps
        self.exponent = exponent

    def applies_at(self, snap: SceneSnapshot) -> Tuple[bool, Mapping[str, Any]]:
        v_lim = speed_limit_for_ego(snap)
        return v_lim is not None, {"speed_limit_mps": v_lim}

    def violation_at(self, snap: SceneSnapshot) -> Tuple[float, Mapping[str, Any]]:
        v_lim = speed_limit_for_ego(snap)
        if v_lim is None:
            return 0.0, {}
        overshoot = max(0.0, snap.ego.speed - (v_lim + self.tolerance_mps))
        return overshoot ** self.exponent, {
            "speed_limit_mps": v_lim,
            "ego_speed_mps": snap.ego.speed,
            "tolerance_mps": self.tolerance_mps,
            "overshoot_mps": overshoot,
        }
