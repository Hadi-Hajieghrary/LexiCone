"""Partition test for the LCP rule-level mapping.

The comparative-effectiveness protocol decomposes per-tick rule violations into
two disjoint sets:

* **MPC-controlled** — observer rules whose violations the LCP planner can
  reduce because there is an *active* (non-stub) encoder in
  :func:`lexicone.planning.rule_encoder.make_default_ruleset` that produces real
  constraints for the OCP. These are the rules the comparison's primary metric
  (lex-Pareto dominance) is evaluated over.
* **Invariant** — rules that no MPC variant can affect: observer-only state
  machines (e.g. mandatory-stop approach, yield priority, route adherence) plus
  any LCP-slot stubs that always emit inactive constraints. These are reported
  separately as a negative-control set: if they differ across planner
  conditions, the comparison is contaminated.

The two sets must partition the full 25-rule observer registry
(:data:`lexicone.observer.registry.DEFAULT_RULE_IDS`).
"""

from __future__ import annotations

from lexicone.observer.registry import DEFAULT_RULE_IDS
from lexicone.planning.rule_encoder import StubRule, make_default_ruleset


# Active encoders (non-stub) → observer rule IDs they control. The
# ``CollisionRule`` encoder is a special case: it has rule_id ``9r0_10r0``
# because it implements collision avoidance for *both* observer-side rules
# (``9r0`` vehicle collisions + ``10r0`` VRU collisions) in one OCP-side
# constraint group.
MPC_CONTROLLED_IDS = frozenset({
    # L10 + L9 (combined collision encoder)
    "10r0", "9r0",
    # L7
    "7r0", "7r1", "7r2", "7r3", "7r5",
    # L3
    "3r0", "3r3", "3r5",
    # L1 (lateral accel)
    "1r11",
    # L0
    "0r2", "0r3",
})

# Stub-rule IDs from ``make_default_ruleset`` — kept in the slot budget for
# future enforcement but emit inactive constraints today.
STUB_IDS = frozenset({"10r5", "7r4", "3r6"})

# Observer-only state-machine rules (no LCP encoder exists or is planned for
# the current convex-MPC framework — see §14.5 of the paper).
OBSERVER_ONLY_IDS = frozenset({
    "10r3", "10r4", "9r1",
    "8r0", "8r1",
    "2r2",
    "1r0", "1r2", "1r5",
})

# The "invariant" control set for the comparison is the union of stubs +
# observer-only rules: no MPC variant can affect either.
INVARIANT_IDS = STUB_IDS | OBSERVER_ONLY_IDS


def test_mpc_controlled_and_invariant_partition_the_registry():
    """The two sets must be disjoint and together cover all 25 observer rules."""
    union = MPC_CONTROLLED_IDS | INVARIANT_IDS
    assert union == frozenset(DEFAULT_RULE_IDS), (
        f"Partition mismatch:\n"
        f"  missing from union: {sorted(frozenset(DEFAULT_RULE_IDS) - union)}\n"
        f"  extra in union:     {sorted(union - frozenset(DEFAULT_RULE_IDS))}"
    )
    assert MPC_CONTROLLED_IDS.isdisjoint(INVARIANT_IDS), (
        f"MPC-controlled and invariant sets overlap on: "
        f"{sorted(MPC_CONTROLLED_IDS & INVARIANT_IDS)}"
    )


def test_make_default_ruleset_matches_declared_partition():
    """Every active encoder in ``make_default_ruleset`` controls a rule that is
    in ``MPC_CONTROLLED_IDS``; every stub is in ``STUB_IDS``."""
    ruleset = make_default_ruleset()
    active_encoder_ids: set[str] = set()
    stub_encoder_ids: set[str] = set()
    for level_encoders in ruleset.levels:
        for enc in level_encoders:
            rid = enc.rule_id
            if isinstance(enc, StubRule):
                stub_encoder_ids.add(rid)
            elif rid == "9r0_10r0":
                # Combined encoder — expands to both observer rules.
                active_encoder_ids.update({"9r0", "10r0"})
            else:
                active_encoder_ids.add(rid)

    assert active_encoder_ids == MPC_CONTROLLED_IDS, (
        f"Active encoder IDs do not match MPC_CONTROLLED_IDS:\n"
        f"  in encoders but not declared: "
        f"{sorted(active_encoder_ids - MPC_CONTROLLED_IDS)}\n"
        f"  declared but no active encoder: "
        f"{sorted(MPC_CONTROLLED_IDS - active_encoder_ids)}"
    )
    assert stub_encoder_ids == STUB_IDS, (
        f"Stub encoder IDs do not match STUB_IDS:\n"
        f"  in encoders but not declared: {sorted(stub_encoder_ids - STUB_IDS)}\n"
        f"  declared but no stub:         {sorted(STUB_IDS - stub_encoder_ids)}"
    )


def test_level_distribution_for_mpc_controlled():
    """The MPC-controlled set must span exactly the levels {10, 9, 7, 3, 1, 0}.

    This is the input to the per-priority-level decomposition: the comparison
    aggregates ``violation_rate`` by leading priority digit. If a level is
    suddenly empty (e.g. all level-7 rules disabled), the protocol's ``V_ell``
    vector becomes degenerate at that level — caught here.
    """
    levels = {int(rid.split("r")[0]) for rid in MPC_CONTROLLED_IDS}
    assert levels == {10, 9, 7, 3, 1, 0}, (
        f"MPC-controlled rules span unexpected levels: {sorted(levels)}"
    )
