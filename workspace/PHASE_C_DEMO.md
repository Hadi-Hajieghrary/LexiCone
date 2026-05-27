# Phase C — v10_2 Framework Demonstration

This document ties the dynamics-agnostic v10_2 reference implementation in
[lcp/](lcp/) to the nuPlan-simulator deployment in [lexicone/](lexicone/),
showing that the resulting closed-loop planner produces good trajectories
that compromise rules in priority order.

## What the framework guarantees

The Lexicographic Constraint Programming framework
([References/lex_constraint_programming_report_v10_2.md](References/lex_constraint_programming_report_v10_2.md))
turns an `L`-stage cascade into a single weighted-sum solve. The contract
is Theorem 4.1: at any `w ∈ Ω(p*)`, the weighted-sum minimiser equals the
lex optimum, so under the calibrated `w_dagger` the WS planner is provably
indistinguishable from the formal cascade. When constraints conflict, the
planner sacrifices the **lowest-priority active** level first; safety rules
(top of the hierarchy) are never compromised before comfort rules (bottom).

## What this phase verifies

| Layer | Artefact | What it shows |
|---|---|---|
| Worked-example regression | `python -m pytest lcp/tests/` | 47 tests pass; Algorithm 0/1A/1B and §10 relaxation reproduce the paper's published numerical answers for Examples 1 and 2 |
| End-to-end pipeline | `python examples/lcp_demo_v10_2.py` | Diagnostics → calibration → cascade-vs-WS equivalence → compliance vector → stressed-conflict compromise → §10 necessity probe, all in one script |
| nuPlan closed-loop | `examples/outputs/12_batch_two_level_mpc_planner/` (16 scenarios) | The LCP-WS-L₁ planner driving the nuPlan-mini bicycle through every dynamic mini-split scenario; per-scenario READMEs document the rule outcomes |

The 47/47 regression tests pin down that the implementation reproduces the
paper to 1e-6 precision. The end-to-end demo prints
`Cascade z_lex* = (3, 5)`, `WS z_ws* = (3, 5)`, `J = -11`, matching v10_2
§11.1 exactly. Under a stressed variant where the rule constraints conflict
the planner returns `z = (3, 0)` with both rule violations at zero — the
hierarchy is respected while `J` is compromised from `-11` to `-6`.

## Three nuPlan scenarios illustrating hierarchy-respecting compromise

The full set of 16 scenario READMEs is under
[examples/outputs/12_batch_two_level_mpc_planner/](examples/outputs/12_batch_two_level_mpc_planner/).
Three representative cases:

### 1. Overtake-style: `01_following_slow_lead`

> Top observed violation: `3r3` Safe Headway (L3, integrated 68.29).
> No L4-L7 (safety / traffic-light) MPC-controlled rule fires.

The ego approaches a slow lead in the same lane. The L₃ Safe-Headway rule
`t_hw · v + d_min - gap ≤ 0` opens repeatedly as the gap closes. **The
planner trades the lowest-priority MPC-controlled rule** to maintain
forward progress — this is exactly the lex-order compromise the framework
prescribes. Higher-priority rules (collision avoidance, drivable area,
traffic-light) are not touched.

→ [01_following_slow_lead/](examples/outputs/12_batch_two_level_mpc_planner/01_following_slow_lead__l1/)

### 2. Dynamic single-vehicle: `14_medium_magnitude_speed`

> Top observed violation: `3r0` Speed Limit (L3, integrated 48.80).
> Only 6 rules violated of 25 — among the cleanest in the batch.

The ego runs a dynamic high-speed track. Brief speed-limit overshoots
appear at the L₃ comfort layer; the L4-L7 safety hierarchy is preserved.
Again: lowest-priority MPC-controlled rule absorbs the compromise.

→ [14_medium_magnitude_speed/](examples/outputs/12_batch_two_level_mpc_planner/14_medium_magnitude_speed__l1/)

### 3. Forced relaxation by geometry: `11_protected_cross`

> Top observed violation: `7r2` Opposing Lane (L7, integrated 174.24).

A protected-cross turn forces the trajectory across the opposing-lane
half-plane: there is no feasible trajectory that completes the turn while
keeping `7r2` hard. This is the **§10.2 necessity** case in the wild — the
constraint must be relaxed, not because the planner is incompetent, but
because it is geometrically inconsistent with the route. The §10
Relaxation Decision Framework (`lcp.compute_necessary_relaxation_level`)
formalises this judgment: `i*_nec = 7` would be reported, with `7r2`
flagged as forced-relaxation rather than discretionary compromise.

→ [11_protected_cross/](examples/outputs/12_batch_two_level_mpc_planner/11_protected_cross__l1/)

## How to re-run

```bash
cd workspace

# 1. Framework regression (≈ 1 s).
python -m pytest lcp/tests/

# 2. End-to-end Example-1 demo (≈ 2 s).
python examples/lcp_demo_v10_2.py

# 3. Single nuPlan scenario (≈ 5 min per run).
python examples/11_simulated_two_level_mpc_planner.py --seed 7

# 4. Full 16-scenario batch (already cached under examples/outputs/).
python examples/12_batch_two_level_mpc_planner.py
```

## How the layers relate

```
References/lex_constraint_programming_report_v10_2.md          (theory paper)
                          │
                          ▼  (faithful implementation)
lcp/   problem.py, equivalence.py, diagnostics.py, online.py,
       relaxation.py, cache.py, compliance.py                  (47 tests)
                          │
                          ▼  (kinematic-bicycle deployment glue)
lexicone/planning/   lcp_mpc.py, lex_cascade.py,
                     weight_calibration.py, two_level_planner.py
                          │
                          ▼  (nuPlan simulator integration)
examples/11_simulated_two_level_mpc_planner.py
examples/12_batch_two_level_mpc_planner.py
examples/outputs/12_batch_two_level_mpc_planner/{01..16}/      (16 scenarios)
```

The `lcp/` package implements the v10_2 algorithms in a dynamics-agnostic
form (operating on `ConvexPriorityProblem`). The `lexicone/planning/` glue
specialises them to the kinematic bicycle and feeds them to nuPlan via the
two-level planner.
