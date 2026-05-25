"""Abstract base class for observer rules.

A :class:`ObserverRule` is queried by the :class:`RuleEngine` once per tick:
the engine builds a :class:`SceneContext` from the raw snapshot and asks the
rule whether it applies in this situation, then — only if it does —
computes the per-tick violation rate. The framework integrates that rate
across applicable ticks to produce an episode-level total.
"""

from __future__ import annotations

from typing import Any, Mapping, Tuple

from .context import SceneContext
from .types import RuleEvaluation


class ObserverRule:
    """A single rule evaluated per scene tick.

    Subclasses must set ``id``, ``level``, ``name``, ``description`` and
    implement :meth:`applies` and :meth:`violation`.

    The violation metric is a non-negative scalar per tick. The engine
    integrates this metric over the applicable steps using the inter-tick
    time delta, producing both a per-tick "violation rate" and an
    episode-level "total violation" (∫ violation_rate · dt). A non-zero rate
    counts as a violation for the tick.
    """

    id: str = ""
    level: int = -1
    name: str = ""
    description: str = ""

    # ----- subclass hooks -----

    def applies(self, ctx: SceneContext) -> Tuple[bool, Mapping[str, Any]]:
        """Return ``(applies, details)`` for the current situation.

        ``details`` is recorded in the per-tick evaluation regardless of
        applicability and is useful for debugging gating logic.
        """
        return True, {}

    def violation(self, ctx: SceneContext) -> Tuple[float, Mapping[str, Any]]:
        """Return ``(violation_rate, details)`` for the current situation.

        Only called by the engine when :meth:`applies` returned True.
        ``violation_rate`` must be >= 0.
        """
        return 0.0, {}

    # ----- framework entrypoint -----

    def evaluate(self, ctx: SceneContext) -> RuleEvaluation:
        applies, app_details = self.applies(ctx)
        if not applies:
            return RuleEvaluation(
                rule_id=self.id,
                rule_level=self.level,
                rule_name=self.name,
                timestamp_us=ctx.timestamp_us,
                applies=False,
                violation_rate=0.0,
                is_violated=False,
                details={"applicability": dict(app_details)},
            )
        rate, viol_details = self.violation(ctx)
        rate_f = float(max(0.0, rate))
        return RuleEvaluation(
            rule_id=self.id,
            rule_level=self.level,
            rule_name=self.name,
            timestamp_us=ctx.timestamp_us,
            applies=True,
            violation_rate=rate_f,
            is_violated=rate_f > 0.0,
            details={
                "applicability": dict(app_details),
                "violation": dict(viol_details),
            },
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<{self.__class__.__name__} id={self.id!r} level={self.level}>"
