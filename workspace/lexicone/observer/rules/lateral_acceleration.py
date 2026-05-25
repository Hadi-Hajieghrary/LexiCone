"""1r11 — Limit lateral acceleration for comfort."""

from __future__ import annotations

from typing import Any, Mapping, Tuple

from ..context import SceneContext
from ..rule import ObserverRule


class LateralAccelerationRule(ObserverRule):
    id = "1r11"
    level = 1
    name = "Limit lateral acceleration for comfort"
    description = (
        "Penalises lateral acceleration above a comfort threshold. Uses |ay| "
        "if provided, otherwise approximates ay = v * yaw_rate."
    )

    def __init__(self, comfort_lat_mps2: float = 2.0, exponent: float = 2.0):
        self.comfort_lat_mps2 = comfort_lat_mps2
        self.exponent = exponent

    def applies(self, ctx: SceneContext) -> Tuple[bool, Mapping[str, Any]]:
        return ctx.ego.speed > 0.1, {"ego_speed_mps": ctx.ego.speed}

    def violation(self, ctx: SceneContext) -> Tuple[float, Mapping[str, Any]]:
        ay = ctx.ego.ay if ctx.ego.ay != 0.0 else ctx.ego.speed * ctx.ego.yaw_rate
        excess = max(0.0, abs(ay) - self.comfort_lat_mps2)
        return excess ** self.exponent, {
            "ay_mps2": ay,
            "comfort_lat_mps2": self.comfort_lat_mps2,
        }
