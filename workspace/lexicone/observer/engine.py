"""Per-tick rule engine.

The :class:`RuleEngine` is the orchestrator. For every :class:`SceneSnapshot`
fed in it:

1. Builds a :class:`SceneContext` (vicinity map, ego derivations, semantic
   relationships, neighbour lookups, …) once per tick.
2. Asks every registered :class:`ObserverRule` whether it applies in this
   situation.
3. For each rule that applies, summons its violation computation and records
   the per-tick :class:`RuleEvaluation`.

It also owns episode history and the windowed :class:`EpisodeSummary`
aggregation. The streaming API is intentionally tiny — :meth:`step` for one
tick, :meth:`run_replay` for a whole stream, :meth:`summary` for aggregation.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

from .context import SceneContext
from .registry import build_default_rules
from .rule import ObserverRule
from .types import (
    EpisodeSummary,
    RuleEvaluation,
    RuleSummary,
    SceneSnapshot,
)


class RuleEngine:
    """Build context, dispatch rules, accumulate episode history.

    Use one engine per episode (it keeps per-rule state across ticks for
    rules like 0r2/0r3 that compute finite-difference jerk, or 8r0 that
    tracks the minimum approach speed at a stop line).
    """

    def __init__(self, rules: Optional[Iterable[ObserverRule]] = None) -> None:
        self.rules: List[ObserverRule] = (
            list(rules) if rules is not None else build_default_rules()
        )
        self._snapshots: List[SceneSnapshot] = []
        self._evals: List[List[RuleEvaluation]] = []

    # ----- streaming API -----

    def step(self, snap: SceneSnapshot) -> List[RuleEvaluation]:
        """Evaluate all rules for one tick.

        Returns the per-rule evaluations for this tick. They are also stored
        in :attr:`history` for later aggregation.
        """
        ctx = SceneContext(snap)
        evals = [r.evaluate(ctx) for r in self.rules]
        self._snapshots.append(snap)
        self._evals.append(evals)
        return evals

    def run_replay(self, scenes: Iterable[SceneSnapshot]) -> List[List[RuleEvaluation]]:
        """Convenience: evaluate a whole replay in one call."""
        return [self.step(s) for s in scenes]

    # ----- introspection -----

    @property
    def history(self) -> Sequence[Sequence[RuleEvaluation]]:
        return self._evals

    @property
    def snapshots(self) -> Sequence[SceneSnapshot]:
        return self._snapshots

    def current_applicable_rules(self) -> List[RuleEvaluation]:
        """Return per-rule evaluations from the most recent tick, filtered
        to those whose :meth:`applies` returned True."""
        if not self._evals:
            return []
        return [e for e in self._evals[-1] if e.applies]

    def current_violations(self) -> List[RuleEvaluation]:
        """Most recent tick: rules that are applicable AND violated."""
        if not self._evals:
            return []
        return [e for e in self._evals[-1] if e.applies and e.is_violated]

    # ----- aggregation -----

    def summary(
        self,
        window_s: Optional[Tuple[float, float]] = None,
    ) -> EpisodeSummary:
        """Aggregate per-rule outcomes over a window.

        ``window_s`` is ``(start, end)`` in seconds, expressed as offsets
        from the first observed snapshot. ``None`` means the entire episode.
        """
        if not self._snapshots:
            return EpisodeSummary(
                rule_summaries={},
                duration_s=0.0,
                n_steps=0,
                window_start_us=0,
                window_end_us=0,
            )

        t0 = self._snapshots[0].timestamp_us
        start_us = t0
        end_us = self._snapshots[-1].timestamp_us
        if window_s is not None:
            start_us = t0 + int(window_s[0] * 1e6)
            end_us = t0 + int(window_s[1] * 1e6)

        indices = [
            i for i, s in enumerate(self._snapshots) if start_us <= s.timestamp_us <= end_us
        ]
        if not indices:
            return EpisodeSummary(
                rule_summaries={},
                duration_s=0.0,
                n_steps=0,
                window_start_us=start_us,
                window_end_us=end_us,
            )

        ts_us = [self._snapshots[i].timestamp_us for i in indices]
        n_steps = len(indices)
        dts = _compute_dts(ts_us)
        duration_s = sum(dts)

        rule_summaries: dict[str, RuleSummary] = {}
        for r in self.rules:
            per_step: List[RuleEvaluation] = [
                next(e for e in self._evals[i] if e.rule_id == r.id) for i in indices
            ]
            applicable_idx = [k for k, e in enumerate(per_step) if e.applies]
            violated_idx = [k for k, e in enumerate(per_step) if e.applies and e.is_violated]
            duration_app = sum(dts[k] for k in applicable_idx)
            integrated = sum(per_step[k].violation_rate * dts[k] for k in applicable_idx)
            max_rate = max((per_step[k].violation_rate for k in applicable_idx), default=0.0)
            first_v_t = per_step[violated_idx[0]].timestamp_s if violated_idx else None
            last_v_t = per_step[violated_idx[-1]].timestamp_s if violated_idx else None
            rule_summaries[r.id] = RuleSummary(
                rule_id=r.id,
                rule_level=r.level,
                rule_name=r.name,
                n_steps_total=n_steps,
                n_steps_applicable=len(applicable_idx),
                n_steps_violated=len(violated_idx),
                duration_applicable_s=duration_app,
                integrated_violation=integrated,
                max_violation_rate=max_rate,
                first_violation_t_s=first_v_t,
                last_violation_t_s=last_v_t,
            )

        return EpisodeSummary(
            rule_summaries=rule_summaries,
            duration_s=duration_s,
            n_steps=n_steps,
            window_start_us=ts_us[0],
            window_end_us=ts_us[-1],
        )

    def reset(self) -> None:
        """Clear history and per-rule state (rebuilds rule instances)."""
        self._snapshots.clear()
        self._evals.clear()
        self.rules = build_default_rules()


def _compute_dts(timestamps_us: Sequence[int]) -> List[float]:
    """Per-tick dt in seconds. First tick mirrors the second tick's dt."""
    if not timestamps_us:
        return []
    if len(timestamps_us) == 1:
        return [0.0]
    diffs = [
        (timestamps_us[i + 1] - timestamps_us[i]) * 1e-6
        for i in range(len(timestamps_us) - 1)
    ]
    return [diffs[0]] + diffs
