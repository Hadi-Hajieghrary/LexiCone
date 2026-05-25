"""0r2 — Limit uncomfortable longitudinal maneuvers."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Tuple

from ..context import SceneContext
from ..rule import ObserverRule


class LongitudinalComfortRule(ObserverRule):
    id = "0r2"
    level = 0
    name = "Limit uncomfortable longitudinal maneuvers"
    description = (
        "Penalises |ax| and longitudinal jerk above comfort thresholds. Jerk "
        "is estimated by finite differences across consecutive ticks."
    )

    def __init__(
        self,
        comfort_ax_mps2: float = 2.0,
        comfort_jerk_mps3: float = 2.0,
        weight_a: float = 1.0,
        weight_j: float = 0.5,
    ):
        self.comfort_ax_mps2 = comfort_ax_mps2
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
        return True, {}

    def violation(self, ctx: SceneContext) -> Tuple[float, Mapping[str, Any]]:
        ax = ctx.ego.ax
        jerk = self._jerk_estimate(ax, ctx.timestamp_us)
        a_excess = max(0.0, abs(ax) - self.comfort_ax_mps2)
        j_excess = max(0.0, abs(jerk) - self.comfort_jerk_mps3)
        rate = self.weight_a * a_excess + self.weight_j * j_excess
        return rate, {
            "ax_mps2": ax,
            "jerk_mps3": jerk,
            "comfort_ax_mps2": self.comfort_ax_mps2,
            "comfort_jerk_mps3": self.comfort_jerk_mps3,
        }
