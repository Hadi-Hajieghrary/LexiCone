"""Tests for the JSON-backed calibration cache."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from lexicone.planning.calibration_cache import (
    HEURISTIC_DEFAULTS,
    CalibrationCache,
    CalibrationEntry,
)


def test_empty_cache_returns_none(tmp_path: Path):
    cache = CalibrationCache(path=tmp_path / "cache.json")
    assert len(cache) == 0
    assert cache.get("any", "l1") is None


def test_put_and_get_roundtrip(tmp_path: Path):
    cache = CalibrationCache(path=tmp_path / "cache.json")
    entry = CalibrationEntry(
        scenario_class="following_lane_with_lead",
        penalty_form="l1",
        weights=[100.0, 10.0, 1.0, 1.0],
        epsilon_per_level=None,
        b_eps_lex=[True, True, False],
        cascade_p_star=[0.0, 0.0, 2.25, 49.0],
    )
    cache.put(entry)
    assert len(cache) == 1
    fetched = cache.get("following_lane_with_lead", "l1")
    assert fetched is not None
    assert fetched.weights == [100.0, 10.0, 1.0, 1.0]
    assert fetched.b_eps_lex == [True, True, False]


def test_cache_persists_across_instances(tmp_path: Path):
    path = tmp_path / "cache.json"
    cache_a = CalibrationCache(path=path)
    cache_a.put(
        CalibrationEntry(scenario_class="x", penalty_form="l2", weights=[1.0], epsilon_per_level=[0.01])
    )
    cache_b = CalibrationCache(path=path)
    fetched = cache_b.get("x", "l2", epsilon_per_level=[0.01])
    assert fetched is not None
    assert fetched.scenario_class == "x"


def test_epsilon_signature_distinguishes_entries(tmp_path: Path):
    cache = CalibrationCache(path=tmp_path / "cache.json")
    cache.put(CalibrationEntry(scenario_class="x", penalty_form="l2", weights=[1.0], epsilon_per_level=[0.01]))
    cache.put(CalibrationEntry(scenario_class="x", penalty_form="l2", weights=[2.0], epsilon_per_level=[0.001]))
    assert len(cache) == 2
    e1 = cache.get("x", "l2", epsilon_per_level=[0.01])
    e2 = cache.get("x", "l2", epsilon_per_level=[0.001])
    assert e1 is not None and e2 is not None
    assert e1.weights == [1.0]
    assert e2.weights == [2.0]


def test_resolve_weights_substitutes_auto(tmp_path: Path):
    cache = CalibrationCache(path=tmp_path / "cache.json")
    cache.put(
        CalibrationEntry(
            scenario_class="x",
            penalty_form="l1",
            weights=[500.0, 50.0, 5.0, 0.5],
        )
    )
    # YAML-style spec with three "auto"s and a literal performance weight.
    resolved, entry = cache.resolve_weights(
        ["auto", "auto", "auto", 1.0],
        scenario_class="x",
        penalty_form="l1",
    )
    assert resolved == [500.0, 50.0, 5.0, 1.0]
    assert entry is not None


def test_resolve_weights_falls_back_on_cache_miss(tmp_path: Path):
    cache = CalibrationCache(path=tmp_path / "cache.json")
    resolved, entry = cache.resolve_weights(
        ["auto", "auto", "auto", "auto"],
        scenario_class="never_seen",
        penalty_form="l1",
    )
    assert entry is None
    assert resolved == HEURISTIC_DEFAULTS["l1"]


def test_resolve_weights_passes_through_numeric(tmp_path: Path):
    """Literal weights are passed through verbatim, even on cache miss."""
    cache = CalibrationCache(path=tmp_path / "cache.json")
    resolved, _ = cache.resolve_weights(
        [42.0, 13.0, 7.0, 1.0],
        scenario_class="never_seen",
        penalty_form="l1",
    )
    assert resolved == [42.0, 13.0, 7.0, 1.0]


def test_corrupt_cache_starts_empty(tmp_path: Path):
    path = tmp_path / "cache.json"
    path.write_text("this is not json")
    cache = CalibrationCache(path=path)
    assert len(cache) == 0  # gracefully resets


def test_cache_keys_are_stable(tmp_path: Path):
    """Same (class, form, eps) → same cache key."""
    e1 = CalibrationEntry(scenario_class="x", penalty_form="l2", weights=[], epsilon_per_level=[0.01, 0.04])
    e2 = CalibrationEntry(scenario_class="x", penalty_form="l2", weights=[], epsilon_per_level=[0.01, 0.04])
    assert e1.cache_key() == e2.cache_key()
    e3 = CalibrationEntry(scenario_class="x", penalty_form="l2", weights=[], epsilon_per_level=[0.01, 0.05])
    assert e1.cache_key() != e3.cache_key()
