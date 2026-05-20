"""Comfort rules.

- 1r11 — Limit lateral acceleration.
- 0r2 — Limit uncomfortable longitudinal maneuvers (a + jerk).
- 0r3 — Limit uncomfortable lateral maneuvers (a + jerk).

For 0r2/0r3 we approximate jerk using finite differences across consecutive
snapshots that the observer feeds in. The rule keeps a small state to compute
this.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Tuple

from ..rule import ObserverRule
from ..types import SceneSnapshot


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

    def applies_at(self, snap: SceneSnapshot) -> Tuple[bool, Mapping[str, Any]]:
        return snap.ego.speed > 0.1, {"ego_speed_mps": snap.ego.speed}

    def violation_at(self, snap: SceneSnapshot) -> Tuple[float, Mapping[str, Any]]:
        if snap.ego.ay != 0.0:
            ay = snap.ego.ay
        else:
            ay = snap.ego.speed * snap.ego.yaw_rate
        excess = max(0.0, abs(ay) - self.comfort_lat_mps2)
        return excess ** self.exponent, {
            "ay_mps2": ay,
            "comfort_lat_mps2": self.comfort_lat_mps2,
        }


class _ComfortBase(ObserverRule):
    """Common scaffold for 0r2/0r3 — tracks last value to compute finite-diff jerk."""

    def __init__(self) -> None:
        self._last_a: float | None = None
        self._last_ts_us: int | None = None

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


class LongitudinalComfortRule(_ComfortBase):
    id = "0r2"
    level = 0
    name = "Limit uncomfortable longitudinal maneuvers"
    description = (
        "Penalises |ax| and longitudinal jerk above comfort thresholds."
    )

    def __init__(
        self,
        comfort_ax_mps2: float = 2.0,
        comfort_jerk_mps3: float = 2.0,
        weight_a: float = 1.0,
        weight_j: float = 0.5,
    ):
        super().__init__()
        self.comfort_ax_mps2 = comfort_ax_mps2
        self.comfort_jerk_mps3 = comfort_jerk_mps3
        self.weight_a = weight_a
        self.weight_j = weight_j

    def applies_at(self, snap: SceneSnapshot) -> Tuple[bool, Mapping[str, Any]]:
        return True, {}

    def violation_at(self, snap: SceneSnapshot) -> Tuple[float, Mapping[str, Any]]:
        ax = snap.ego.ax
        jerk = self._jerk_estimate(ax, snap.timestamp_us)
        a_excess = max(0.0, abs(ax) - self.comfort_ax_mps2)
        j_excess = max(0.0, abs(jerk) - self.comfort_jerk_mps3)
        rate = self.weight_a * a_excess + self.weight_j * j_excess
        return rate, {
            "ax_mps2": ax,
            "jerk_mps3": jerk,
            "comfort_ax_mps2": self.comfort_ax_mps2,
            "comfort_jerk_mps3": self.comfort_jerk_mps3,
        }


class LateralComfortRule(_ComfortBase):
    id = "0r3"
    level = 0
    name = "Limit uncomfortable lateral maneuvers"
    description = (
        "Penalises |ay| and lateral jerk above comfort thresholds (smoother "
        "complement to 1r11)."
    )

    def __init__(
        self,
        comfort_ay_mps2: float = 1.5,
        comfort_jerk_mps3: float = 1.5,
        weight_a: float = 1.0,
        weight_j: float = 0.5,
    ):
        super().__init__()
        self.comfort_ay_mps2 = comfort_ay_mps2
        self.comfort_jerk_mps3 = comfort_jerk_mps3
        self.weight_a = weight_a
        self.weight_j = weight_j

    def applies_at(self, snap: SceneSnapshot) -> Tuple[bool, Mapping[str, Any]]:
        return snap.ego.speed > 0.1, {"ego_speed_mps": snap.ego.speed}

    def violation_at(self, snap: SceneSnapshot) -> Tuple[float, Mapping[str, Any]]:
        ay = snap.ego.ay if snap.ego.ay != 0.0 else snap.ego.speed * snap.ego.yaw_rate
        jerk = self._jerk_estimate(ay, snap.timestamp_us)
        a_excess = max(0.0, abs(ay) - self.comfort_ay_mps2)
        j_excess = max(0.0, abs(jerk) - self.comfort_jerk_mps3)
        rate = self.weight_a * a_excess + self.weight_j * j_excess
        return rate, {
            "ay_mps2": ay,
            "jerk_mps3": jerk,
            "comfort_ay_mps2": self.comfort_ay_mps2,
            "comfort_jerk_mps3": self.comfort_jerk_mps3,
        }
