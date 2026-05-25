"""3r0 — Obey posted speed limits."""

from __future__ import annotations

from typing import Any, Mapping, Tuple

from ..context import SceneContext
from ..rule import ObserverRule


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

    def applies(self, ctx: SceneContext) -> Tuple[bool, Mapping[str, Any]]:
        v_lim = ctx.ego_speed_limit_mps
        return v_lim is not None, {"speed_limit_mps": v_lim}

    def violation(self, ctx: SceneContext) -> Tuple[float, Mapping[str, Any]]:
        v_lim = ctx.ego_speed_limit_mps
        if v_lim is None:
            return 0.0, {}
        overshoot = max(0.0, ctx.ego.speed - (v_lim + self.tolerance_mps))
        return overshoot ** self.exponent, {
            "speed_limit_mps": v_lim,
            "ego_speed_mps": ctx.ego.speed,
            "tolerance_mps": self.tolerance_mps,
            "overshoot_mps": overshoot,
        }
