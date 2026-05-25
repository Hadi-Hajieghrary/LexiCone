"""0r3 — Limit uncomfortable lateral maneuvers (smoother complement to 1r11)."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Tuple

from ..context import SceneContext
from ..rule import ObserverRule


class LateralComfortRule(ObserverRule):
    id = "0r3"
    level = 0
    name = "Limit uncomfortable lateral maneuvers"
    description = (
        "Penalises |ay| and lateral jerk above comfort thresholds (smoother "
        "complement to 1r11). Jerk is estimated by finite differences."
    )

    def __init__(
        self,
        comfort_ay_mps2: float = 1.5,
        comfort_jerk_mps3: float = 1.5,
        weight_a: float = 1.0,
        weight_j: float = 0.5,
    ):
        self.comfort_ay_mps2 = comfort_ay_mps2
        self.comfort_jerk_mps3 = comfort_jerk_mps3
        self.weight_a = weight_a
        self.weight_j = weight_j
        self._last_a: Optional[float] = None
        self._last_ts_us: Optional[int] = None

    def _jerk_estimate(self, current_a: float, ts_us: int) -> float:
        if self._last_a is None or self._last_ts_us is None:
            self._last_a = current_a
            self._last_ts_us = ts_us
            return 0.0
        dt = (ts_us - self._last_ts_us) * 1e-6
        if dt <= 1e-3:
            return 0.0
        j = (current_a - self._last_a) / dt
        self._last_a = current_a
        self._last_ts_us = ts_us
        return j

    def applies(self, ctx: SceneContext) -> Tuple[bool, Mapping[str, Any]]:
        return ctx.ego.speed > 0.1, {"ego_speed_mps": ctx.ego.speed}

    def violation(self, ctx: SceneContext) -> Tuple[float, Mapping[str, Any]]:
        ay = ctx.ego.ay if ctx.ego.ay != 0.0 else ctx.ego.speed * ctx.ego.yaw_rate
        jerk = self._jerk_estimate(ay, ctx.timestamp_us)
        a_excess = max(0.0, abs(ay) - self.comfort_ay_mps2)
        j_excess = max(0.0, abs(jerk) - self.comfort_jerk_mps3)
        rate = self.weight_a * a_excess + self.weight_j * j_excess
        return rate, {
            "ay_mps2": ay,
            "jerk_mps3": jerk,
            "comfort_ay_mps2": self.comfort_ay_mps2,
            "comfort_jerk_mps3": self.comfort_jerk_mps3,
        }
