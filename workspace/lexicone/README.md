# lexicone — lexicographic constraint programming for autonomous driving

`lexicone` is a Python package with two sibling sub-packages that share the same data model but solve very different problems:

- **[`observer/`](observer/)** — a per-tick rule engine that *judges* the ego's behaviour against a lexicographically ordered rulebook.
- **[`planning/`](planning/)** — a two-level motion planner that *drives* the ego inside the nuPlan closed-loop simulator.

The shared data model is the `SceneSnapshot` dataclass: a nuPlan-agnostic struct that any driving log (live nuPlan scenario, saved nuPlan SimulationLog, or a synthetic toy world) can be projected into.

```
                ┌─────────────────────────┐
                │   SceneSnapshot         │
                │   • ego (EgoSnapshot)   │
                │   • agents              │
                │   • map (MapSnapshot)   │
                │   • traffic_lights      │
                │   • planned_trajectory  │
                │   • route_lane_ids      │
                └─────────────────────────┘
                      ▲                ▲
                      │                │
        (consumed by) │                │ (produced by)
                      │                │
        ┌─────────────────┐    ┌────────────────────────────┐
        │ observer.engine │    │ observer.nuplan_adapter     │
        │ ObserverRule    │    │ observer.simulation_log_…   │
        │ RuleEngine      │    │ examples/simulation.py       │
        └─────────────────┘    │ examples/scenarios.py        │
                               └────────────────────────────┘
```

## Why "lexicone"?

The package name comes from the rule engine's **lexicographic** prioritisation of constraints: a level-10 rule (collision avoidance) is conceptually superior to a level-0 rule (passenger comfort) — you only optimise comfort *after* every higher-level rule is satisfied. (See [`observer/README.md`](observer/README.md) for how level is currently used in code, and how it is intended to be used by future schedulers.)

`one` from "constraint optimisation" / "lex one (priority)" — the planner is a constraint-based optimiser (an MPC) that lives inside the same package, sharing the data model.

## Top-level public surface

The two sub-packages currently export their own surfaces — the top-level package itself only carries a docstring:

```python
# lexicone/__init__.py (excerpt)
"""LexiCone: Lexicographic priority-ordered constraint programming for MPC.

This package currently exposes the trajectory rule observer used to evaluate
ego behaviour against a NuPlan-applicable subset of the rule book.
"""
```

To import:

```python
# Rule observer
from lexicone.observer import RuleEngine, SceneSnapshot, build_default_rules

# Two-level MPC planner
from lexicone.planning import TwoLevelMPCPlanner, GlobalRoutePlanner, MPCTrajectoryPlanner

# Adapters live in their own modules — they are not re-exported by
# lexicone.observer because they pull nuplan-devkit at import time.
from lexicone.observer.simulation_log_adapter import NuPlanSimulationLogSource
from lexicone.observer.nuplan_adapter import NuPlanSceneSource
```

## Design choices that span both sub-packages

### One data model, two consumers

Both halves operate on `SceneSnapshot` / `EgoSnapshot` / `MapSnapshot` / etc. (defined in [`observer/types.py`](observer/types.py)). The planner does not produce snapshots itself — that's the simulator's job — but its outputs flow through the same snapshot stream once the nuPlan `SimulationLog` is loaded.

### NuPlan dependency profile

The two sub-packages handle the nuPlan dependency differently:

- **`observer/`** treats `nuplan-devkit` as **soft**: only the two adapter files import nuPlan, and they do it lazily inside `__init__` / `from_path` (see `nuplan_adapter.py` and `simulation_log_adapter.py:99`). The rule engine, the geometry helpers, and all 25 rules can be exercised in isolation without nuPlan installed.
- **`planning/`** treats `nuplan-devkit` as a **hard, module-scope** dependency: `reference_path.py`, `global_planner.py`, `trajectory_planner.py`, and `two_level_planner.py` all `from nuplan…` at the top of the file. This is intentional — the planner is built to be loaded by nuPlan's Hydra (`_target_: lexicone.planning.two_level_planner.TwoLevelMPCPlanner`), which already requires `nuplan-devkit` to be present.

### Picklable planner state

`MPCTrajectoryPlanner` carries `casadi.Opti` and `casadi.Function` objects (SwigPyObjects) which cannot pickle. It therefore implements `__getstate__` / `__setstate__` ([`planning/trajectory_planner.py:122-131`](planning/trajectory_planner.py)) — on pickle the CasADi state is dropped, on unpickle `_build_problem()` rebuilds it from scratch. This is necessary because nuPlan's `SimulationLog.save_to_file()` pickles the *entire* `SimulationLog` (including the planner instance) at the end of every run, and `MPCTrajectoryPlanner` lives inside `TwoLevelMPCPlanner` as `self._mpc`. `TwoLevelMPCPlanner` itself does **not** override `__getstate__`/`__setstate__`; the default pickle protocol walks its `__dict__` and finds the customised hooks on the nested MPC instance.

## Read next

- [`observer/README.md`](observer/README.md) — the rule engine, the 25-rule rulebook, the per-tick evaluation model.
- [`planning/README.md`](planning/README.md) — the two-level MPC pipeline, the bicycle-model MPC, the global route planner.
