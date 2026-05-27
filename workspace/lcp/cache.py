"""JSON-backed cache for per-scenario-class LCP calibration outputs.

The LCP framework's offline calibration (Algorithm 1A for L₁ exact equivalence,
Algorithm 1B for L₂ tolerance compliance) runs the L+1-stage lex cascade and
solves a Chebyshev-centre LP to pick a robust weight vector ``w†``. This is
expensive — multiple NLP solves plus an LP — but the result only depends on
the *scenario class* (e.g., ``following_lane_with_lead``,
``traversing_intersection``), not on the per-tick details. So we calibrate
once per scenario class and cache the result.

Cache format
------------

The cache is a single JSON file at ``${NUPLAN_EXP_ROOT}/lcp_cache.json``.
Each entry is keyed by ``(scenario_class, penalty_form, epsilon_signature)``
and carries:

- ``weights``: the calibrated ``w†`` vector (one float per priority level).
- ``epsilon_per_level``: the operator-supplied tolerance vector used in
  Algorithm 1B (for L₂; null for L₁).
- ``b_eps_lex``: the binary compliance vector at the lex point — what the
  runtime compliance check compares against.
- ``computed_at``: ISO-8601 timestamp.
- ``cascade_p_star``: the ``(V_1*, ..., V_L*, J*)`` achievement point, for
  diagnostic logging.

This module is intentionally tiny — JSON I/O plus a lookup wrapper — because
the heavy lifting lives in ``weight_calibration.py`` (Phase A/B).

Sentinel handling for ``"auto"`` weights in YAML
------------------------------------------------

The YAML's ``weights_per_level: [auto, auto, auto, 1.0]`` form is resolved by
:meth:`CalibrationCache.resolve_weights` — entries equal to the string
``"auto"`` are filled in from the cache; numeric entries pass through. If the
cache misses, the resolver falls back to a documented heuristic default
(strong descending weights ``[1000, 100, 10, 1]`` for L₁; tolerance-derived
weights for L₂).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

logger = logging.getLogger(__name__)


WeightSpec = Union[float, str]  # "auto" sentinel or a literal weight


@dataclass
class CalibrationEntry:
    """One scenario-class calibration record."""

    scenario_class: str
    penalty_form: str                       # "l1" | "l2"
    weights: List[float]
    epsilon_per_level: Optional[List[float]] = None
    b_eps_lex: Optional[List[bool]] = None
    cascade_p_star: Optional[List[float]] = None
    computed_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    notes: str = ""

    def cache_key(self) -> str:
        eps_signature = (
            "_eps_" + "_".join(f"{e:.5g}" for e in self.epsilon_per_level)
            if self.epsilon_per_level is not None
            else ""
        )
        return f"{self.scenario_class}__{self.penalty_form}{eps_signature}"


HEURISTIC_DEFAULTS: Dict[str, List[float]] = {
    "l1": [1000.0, 100.0, 10.0, 1.0],
    "l2": [10000.0, 1000.0, 100.0, 1.0],
}


class CalibrationCache:
    """JSON-backed cache. Single-process safe; concurrent writes use a simple
    atomic-rename pattern.
    """

    def __init__(self, path: Optional[Path] = None) -> None:
        if path is None:
            exp_root = Path(os.environ.get("NUPLAN_EXP_ROOT", "/workspace/exp"))
            path = exp_root / "lcp_cache.json"
        self.path = Path(path)
        self._entries: Dict[str, CalibrationEntry] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            self._entries = {}
            return
        try:
            with self.path.open("r") as f:
                blob = json.load(f)
            self._entries = {
                k: CalibrationEntry(**v) for k, v in blob.items()
            }
        except (json.JSONDecodeError, TypeError, KeyError) as exc:
            logger.warning(
                "CalibrationCache: failed to load %s (%s); starting empty.",
                self.path, exc,
            )
            self._entries = {}

    def save(self) -> None:
        """Atomic rewrite of the cache file."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        with tmp.open("w") as f:
            json.dump(
                {k: asdict(v) for k, v in self._entries.items()},
                f,
                indent=2,
                sort_keys=True,
            )
        tmp.replace(self.path)

    def get(
        self,
        scenario_class: str,
        penalty_form: str,
        epsilon_per_level: Optional[Sequence[float]] = None,
    ) -> Optional[CalibrationEntry]:
        probe = CalibrationEntry(
            scenario_class=scenario_class,
            penalty_form=penalty_form,
            weights=[],
            epsilon_per_level=list(epsilon_per_level) if epsilon_per_level is not None else None,
        )
        return self._entries.get(probe.cache_key())

    def put(self, entry: CalibrationEntry) -> None:
        self._entries[entry.cache_key()] = entry
        self.save()

    def resolve_weights(
        self,
        weights_spec: Sequence[WeightSpec],
        scenario_class: str,
        penalty_form: str,
        epsilon_per_level: Optional[Sequence[float]] = None,
    ) -> Tuple[List[float], Optional[CalibrationEntry]]:
        """Substitute every ``"auto"`` entry in ``weights_spec`` with its
        cached value, falling back to heuristic defaults on cache miss.

        Returns ``(resolved_weights, cache_entry_or_None)``.
        """
        cached = self.get(scenario_class, penalty_form, epsilon_per_level)
        if cached is not None and len(cached.weights) >= len(weights_spec):
            cache_weights = cached.weights
        else:
            cache_weights = HEURISTIC_DEFAULTS.get(penalty_form, [1.0] * len(weights_spec))

        resolved: List[float] = []
        for i, spec in enumerate(weights_spec):
            if isinstance(spec, str) and spec.lower() == "auto":
                if i < len(cache_weights):
                    resolved.append(float(cache_weights[i]))
                elif i < len(HEURISTIC_DEFAULTS.get(penalty_form, [])):
                    resolved.append(float(HEURISTIC_DEFAULTS[penalty_form][i]))
                else:
                    resolved.append(1.0)
            else:
                resolved.append(float(spec))
        return resolved, cached

    def __len__(self) -> int:
        return len(self._entries)

    def keys(self) -> List[str]:
        return list(self._entries.keys())
