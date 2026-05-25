"""Tests for the runtime compliance checker."""

from __future__ import annotations

from pathlib import Path

import pytest

from lexicone.observer.types import (
    EgoSnapshot,
    MapSnapshot,
    Pose2D,
    SceneSnapshot,
)
from lexicone.planning.compliance_checker import (
    DEFAULT_RULE_LEVEL_MAPPING,
    ComplianceChecker,
    ComplianceResult,
    MismatchRecord,
)


def _trivial_snapshot(timestamp_us: int = 1_000_000) -> SceneSnapshot:
    """An ego-at-rest scene with no agents, empty map. No rule should fire."""
    ego = EgoSnapshot(
        timestamp_us=timestamp_us,
        pose=Pose2D(x=0.0, y=0.0, heading=0.0),
        vx=0.0, vy=0.0,
    )
    return SceneSnapshot(
        timestamp_us=timestamp_us,
        ego=ego,
        agents=(),
        map=MapSnapshot(),
        traffic_lights=(),
    )


def test_checker_construction_validates_shape():
    with pytest.raises(ValueError):
        ComplianceChecker(epsilon_per_level=[0.01, 0.02])  # 2 vs 3 levels


def test_trivial_scene_all_levels_compliant():
    """A static scene with no rule-triggering structure: every level is
    compliant, and the expected vector matches."""
    checker = ComplianceChecker(epsilon_per_level=[1e-3, 1e-3, 1e-3])
    snap = _trivial_snapshot()
    result = checker.check_snapshot(snap, expected_b_eps=[True, True, True], scenario_class="test")
    assert result.matched
    assert result.b_eps_actual == (True, True, True)
    assert result.mismatches == []


def test_mismatch_when_expected_violates_but_actual_compliant():
    """If we tell the checker to expect a violation at level 0 but the scene
    has no violation, a mismatch is emitted."""
    checker = ComplianceChecker(epsilon_per_level=[1e-3, 1e-3, 1e-3])
    snap = _trivial_snapshot()
    result = checker.check_snapshot(snap, expected_b_eps=[False, True, True], scenario_class="test")
    assert not result.matched
    assert len(result.mismatches) == 1
    m = result.mismatches[0]
    assert m.level_index == 0
    assert m.expected is False
    assert m.actual is True


def test_csv_sink_writes_header_and_row(tmp_path: Path):
    csv_path = tmp_path / "compliance.csv"
    checker = ComplianceChecker(epsilon_per_level=[1e-3, 1e-3, 1e-3])
    checker.attach_csv_sink(csv_path)
    try:
        snap = _trivial_snapshot()
        checker.check_snapshot(snap, expected_b_eps=[False, True, True], scenario_class="demo_class")
    finally:
        checker.detach_csv_sink()
    contents = csv_path.read_text().strip().splitlines()
    assert len(contents) == 2  # header + 1 mismatch row
    assert "log_time" in contents[0]
    assert "demo_class" in contents[1]


def test_default_rule_level_mapping_covers_three_levels():
    """The default mapping should have three levels (safety / legal / comfort)
    and reference observer rule IDs that exist in the registry."""
    assert len(DEFAULT_RULE_LEVEL_MAPPING) == 3
    flat = [rid for level in DEFAULT_RULE_LEVEL_MAPPING for rid in level]
    # All entries are non-empty observer-ID-like strings.
    assert all(isinstance(r, str) and "r" in r for r in flat)


def test_checker_records_rule_id_in_mismatch_when_level_violates():
    """If a snapshot has a violating rule, the mismatch record carries the
    level's rule IDs so downstream analysis can attribute the failure."""
    checker = ComplianceChecker(epsilon_per_level=[1e-3, 1e-3, 1e-3])
    snap = _trivial_snapshot()
    # Tell the checker we EXPECT level-1 to be violated (e.g. cached b_ε_lex
    # said False). The actual scene has no violation, so a mismatch is raised
    # for level 1.
    result = checker.check_snapshot(snap, expected_b_eps=[True, False, True], scenario_class="x")
    assert len(result.mismatches) == 1
    m = result.mismatches[0]
    assert m.level_index == 1
    assert m.rule_ids_in_level == DEFAULT_RULE_LEVEL_MAPPING[1]
