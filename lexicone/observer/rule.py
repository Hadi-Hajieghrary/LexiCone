"""Abstract base class for observer rules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Tuple

from .types import RuleEvaluation, SceneSnapshot


@dataclass
class _Applicability:
    applies: bool
    details: Mapping[str, Any]


class ObserverRule:
    """A single rule evaluated per scene snapshot.

    Subclasses must set ``id``, ``level``, ``name``, ``description`` and
    implement :meth:`applies_at` and :meth:`violation_at`.

    The violation metric is a non-negative scalar per tick. The observer
    integrates this metric over the applicable steps using the inter-tick
    time delta, producing both a per-tick "violation rate" and an episode-level
    "total violation" (∫ violation_rate · dt). A non-zero rate counts as a
    violation for the tick.
    """

    id: str = ""
    level: int = -1
    name: str = ""
    description: str = ""

    # ----- subclass hooks -----

    def applies_at(self, snap: SceneSnapshot) -> Tuple[bool, Mapping[str, Any]]:
        """Return ``(applies, details)``.

        ``details`` is recorded in the per-tick evaluation regardless of
        applicability and is useful for debugging gating logic.
        """
        return True, {}

    def violation_at(self, snap: SceneSnapshot) -> Tuple[float, Mapping[str, Any]]:
        """Return ``(violation_rate, details)``.

        Only called by the framework when :meth:`applies_at` returned True.
        ``violation_rate`` must be >= 0. ``details`` is recorded in the per-tick
        evaluation.
        """
        return 0.0, {}

    # ----- framework entrypoint -----

    def evaluate(self, snap: SceneSnapshot) -> RuleEvaluation:
        applies, app_details = self.applies_at(snap)
        if not applies:
            return RuleEvaluation(
                rule_id=self.id,
                rule_level=self.level,
                rule_name=self.name,
                timestamp_us=snap.timestamp_us,
                applies=False,
                violation_rate=0.0,
                is_violated=False,
                details={"applicability": dict(app_details)},
            )
        rate, viol_details = self.violation_at(snap)
        rate_f = float(max(0.0, rate))
        return RuleEvaluation(
            rule_id=self.id,
            rule_level=self.level,
            rule_name=self.name,
            timestamp_us=snap.timestamp_us,
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
