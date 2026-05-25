"""1r0 — Yield to higher-priority road users."""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Tuple

from ..context import VEHICLE_LIKE_TYPES, SceneContext, _type_value, relative_velocity
from ..geometry import is_same_direction, planar_distance
from ..rule import ObserverRule
from ..types import AgentSnapshot, AgentType


class YieldPriorityRule(ObserverRule):
    id = "1r0"
    level = 1
    name = "Yield to higher-priority road users"
    description = (
        "Penalises encroachment that forces a prioritized agent (e.g. a "
        "through vehicle on a major road, a pedestrian with right-of-way) to "
        "brake beyond comfort or face critically low TTC."
    )

    def __init__(self, comfort_decel_mps2: float = 2.0, ttc_critical_s: float = 2.0):
        self.comfort_decel_mps2 = comfort_decel_mps2
        self.ttc_critical_s = ttc_critical_s

    def _priority_agents(self, ctx: SceneContext) -> Iterable[AgentSnapshot]:
        ec = ctx.ego_center
        cos_h = math.cos(ctx.ego.pose.heading)
        sin_h = math.sin(ctx.ego.pose.heading)
        for a in ctx.snapshot.agents:
            ot = _type_value(a.object_type)
            if ot == AgentType.PEDESTRIAN.value:
                if planar_distance(ec, (a.pose.x, a.pose.y)) <= 15.0:
                    yield a
            elif ot in VEHICLE_LIKE_TYPES:
                dx = a.pose.x - ec[0]
                dy = a.pose.y - ec[1]
                lon = dx * cos_h + dy * sin_h
                lat = -dx * sin_h + dy * cos_h
                if 0 < lon < 25.0 and abs(lat) < 15.0 and not is_same_direction(
                    a.pose.heading, ctx.ego.pose.heading, math.radians(45)
                ):
                    yield a

    def applies(self, ctx: SceneContext) -> Tuple[bool, Mapping[str, Any]]:
        peers = list(self._priority_agents(ctx))
        return bool(peers), {"n_priority_agents": len(peers)}

    def violation(self, ctx: SceneContext) -> Tuple[float, Mapping[str, Any]]:
        worst = 0.0
        worst_track = None
        ec = ctx.ego_center
        for a in self._priority_agents(ctx):
            d = planar_distance(ec, (a.pose.x, a.pose.y))
            rel_lon, _ = relative_velocity(ctx.ego, a)
            closing = -rel_lon
            ttc = d / closing if closing > 0.3 else math.inf
            forced_decel = (closing * closing) / (2.0 * max(d, 0.3)) if closing > 0 else 0.0
            score = max(0.0, forced_decel - self.comfort_decel_mps2)
            if ttc != math.inf and ttc < self.ttc_critical_s:
                score += (self.ttc_critical_s - ttc)
            if score > worst:
                worst = score
                worst_track = a.track_id
        return worst, {
            "worst_track_id": worst_track,
            "comfort_decel_mps2": self.comfort_decel_mps2,
            "ttc_critical_s": self.ttc_critical_s,
        }
