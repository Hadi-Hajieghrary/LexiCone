"""Per-tick compliance verification of the LCP MPC's trajectory.

After each WS solve, the planner produces a trajectory ``z_ws``. The LCP
framework's runtime-monitoring step (paper Section 12.4) compares the binary
*compliance vector* ``b_ε(z_ws)`` against the cached ``b_ε(z_lex*)`` from the
offline cascade. A mismatch signals that the active set has drifted from the
calibration assumption.

This module wraps the existing :mod:`lexicone.observer.RuleEngine` so the
planner does not need to reimplement rule semantics. We construct a single
:class:`SceneSnapshot` per tick from the WS trajectory's first sample, run
the engine once, and convert the engine's per-rule ``RuleEvaluation`` into a
per-level binary compliance vector keyed by the planner's level mapping.

Mismatch policy
---------------

Per the user-locked design choice, on mismatch we **log only** — no fallback
to the cascade, no recalibration trigger. The log goes to two places:

- Python ``logging`` at ``WARNING`` level for the live console.
- An optional CSV sink via :meth:`ComplianceChecker.attach_csv_sink`, which
  appends one row per mismatched tick — convenient for offline analysis.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence, TextIO, Tuple

logger = logging.getLogger(__name__)


# Mapping from the planner's lex-level index (0-based) to the observer rule
# IDs the level groups. Keep in sync with rule_encoder.make_default_ruleset().
DEFAULT_RULE_LEVEL_MAPPING: Tuple[Tuple[str, ...], ...] = (
    ("9r0", "10r0", "7r0", "7r5", "10r5"),                 # L=1 Safety
    ("3r0", "7r2", "7r3", "7r1", "7r4"),                   # L=2 Legal
    ("3r3", "3r5", "3r6", "1r11", "0r2", "0r3"),           # L=3 Comfort
)


@dataclass
class MismatchRecord:
    """One mismatch event between WS compliance and lex compliance."""

    timestamp_us: int
    scenario_class: str
    level_index: int             # 0-based planner level
    expected: bool
    actual: bool
    rule_ids_in_level: Tuple[str, ...]
    detail: str = ""


@dataclass
class ComplianceResult:
    """Outcome of one runtime compliance check."""

    b_eps_actual: Tuple[bool, ...]
    b_eps_expected: Tuple[bool, ...]
    mismatches: List[MismatchRecord] = field(default_factory=list)

    @property
    def matched(self) -> bool:
        return not self.mismatches


class ComplianceChecker:
    """Runtime ``b_ε(z_ws)`` checker.

    The observer's :class:`~lexicone.observer.RuleEngine` is the source of
    truth for rule semantics. We build a fresh engine instance for each
    planner (so the per-rule across-tick state — e.g., the mandatory-stop
    approach state machine — has a clean run).

    Parameters
    ----------
    epsilon_per_level:
        The operator-supplied tolerance vector :math:`\\boldsymbol{\\epsilon}`.
        A rule's per-tick ``violation_rate`` is compared against the
        level's :math:`\\epsilon_i`; if ``violation_rate > ε_i`` the level is
        considered violated at that tick.
    level_mapping:
        Tuple of length ``L``; entry ``i`` is the tuple of observer rule IDs
        the planner's level ``i`` groups. Defaults to the same mapping the
        rule encoder uses.
    """

    def __init__(
        self,
        epsilon_per_level: Sequence[float],
        level_mapping: Sequence[Sequence[str]] = DEFAULT_RULE_LEVEL_MAPPING,
    ) -> None:
        self.epsilon_per_level = tuple(float(e) for e in epsilon_per_level)
        self.level_mapping = tuple(tuple(level) for level in level_mapping)
        if len(self.epsilon_per_level) != len(self.level_mapping):
            raise ValueError(
                f"epsilon_per_level ({len(self.epsilon_per_level)}) must match "
                f"level_mapping ({len(self.level_mapping)})"
            )
        self._csv_sink: Optional[Path] = None
        self._csv_writer: Optional[csv.writer] = None
        self._csv_file: Optional[TextIO] = None

    @staticmethod
    def _build_engine():
        """Construct a fresh RuleEngine. Module-level method (not a lambda)
        so the ComplianceChecker can be pickled by ``SimulationLog.save_to_file()``."""
        from lexicone.observer import RuleEngine, build_default_rules

        return RuleEngine(rules=build_default_rules())

    # ------------------------------------------------------------------
    # Pickling — drop the open CSV file handle; the cache reopens it on demand.
    # ------------------------------------------------------------------

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        # File handles and CSV writers don't pickle. Keep the *path* so the
        # caller can re-attach the sink after unpickling if desired.
        state["_csv_file"] = None
        state["_csv_writer"] = None
        return state

    # ------------------------------------------------------------------
    # CSV sink (optional)
    # ------------------------------------------------------------------

    def attach_csv_sink(self, path: Path) -> None:
        """Stream every mismatch event to a CSV file. Idempotent."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._csv_sink = path
        self._csv_file = path.open("a", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        if path.stat().st_size == 0:
            self._csv_writer.writerow([
                "log_time", "tick_us", "scenario_class", "level",
                "expected", "actual", "rule_ids", "detail",
            ])
            self._csv_file.flush()

    def detach_csv_sink(self) -> None:
        if self._csv_file is not None:
            self._csv_file.close()
        self._csv_sink = None
        self._csv_writer = None
        self._csv_file = None

    # ------------------------------------------------------------------
    # Check API
    # ------------------------------------------------------------------

    def check_snapshot(
        self,
        snapshot,
        expected_b_eps: Sequence[bool],
        scenario_class: str = "unknown",
    ) -> ComplianceResult:
        """Evaluate the observer rule engine on one :class:`SceneSnapshot`,
        derive ``b_ε(z_ws)`` per level, compare against ``expected_b_eps``,
        and emit a :class:`ComplianceResult`.

        ``expected_b_eps`` is typically the ``b_eps_lex`` stored in the
        :class:`~lexicone.planning.calibration_cache.CalibrationEntry`.
        """
        engine = self._build_engine()
        evaluations = engine.step(snapshot)
        eval_by_id: Mapping[str, "RuleEvaluation"] = {ev.rule_id: ev for ev in evaluations}

        b_eps_actual: List[bool] = []
        mismatches: List[MismatchRecord] = []
        for level_idx, rule_ids in enumerate(self.level_mapping):
            eps = self.epsilon_per_level[level_idx]
            # Level is *compliant* iff every rule's violation_rate <= eps
            # (or the rule is not applicable at this tick).
            compliant = True
            culprit_detail = ""
            for rid in rule_ids:
                ev = eval_by_id.get(rid)
                if ev is None or not ev.applies:
                    continue
                if ev.violation_rate > eps:
                    compliant = False
                    culprit_detail = f"{rid} rate={ev.violation_rate:.3g} > eps={eps:.3g}"
                    break
            b_eps_actual.append(compliant)
            expected = bool(expected_b_eps[level_idx]) if level_idx < len(expected_b_eps) else True
            if compliant != expected:
                mismatch = MismatchRecord(
                    timestamp_us=int(snapshot.timestamp_us),
                    scenario_class=scenario_class,
                    level_index=level_idx,
                    expected=expected,
                    actual=compliant,
                    rule_ids_in_level=tuple(rule_ids),
                    detail=culprit_detail,
                )
                mismatches.append(mismatch)
                self._emit_mismatch(mismatch)

        return ComplianceResult(
            b_eps_actual=tuple(b_eps_actual),
            b_eps_expected=tuple(expected_b_eps),
            mismatches=mismatches,
        )

    def _emit_mismatch(self, m: MismatchRecord) -> None:
        logger.warning(
            "LCP compliance mismatch [scenario=%s level=%d expected=%s actual=%s] %s",
            m.scenario_class, m.level_index, m.expected, m.actual, m.detail,
        )
        if self._csv_writer is not None:
            self._csv_writer.writerow([
                datetime.now(timezone.utc).isoformat(),
                m.timestamp_us,
                m.scenario_class,
                m.level_index,
                int(m.expected),
                int(m.actual),
                "+".join(m.rule_ids_in_level),
                m.detail,
            ])
            assert self._csv_file is not None
            self._csv_file.flush()
