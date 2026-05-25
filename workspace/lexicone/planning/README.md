# `lexicone.planning` — two-level MPC planner with optional Lexicographic Constraint Programming

A motion-planning pipeline that drives the ego inside the official nuPlan closed-loop simulator (`nuplan/planning/script/run_simulation.py`). It is split into two layers so each can be reasoned about independently:

1. **Global planner** ([`global_planner.py`](global_planner.py)) — re-extracts a lane-graph route every `replan_period_s` seconds and turns it into a continuous reference centreline. The *where to go* layer.
2. **Trajectory planner** — solves a per-tick optimal control problem (OCP) at 10 Hz over a 3 s horizon to track the reference under actuator, rate, and obstacle constraints. The *how to move* layer.

The trajectory planner has two flavours, selected via the YAML key `penalty_form`:

- **Legacy single-tier MPC** ([`trajectory_planner.py`](trajectory_planner.py)) — when `penalty_form: null`. Flat-weight CasADi/IPOPT nonlinear MPC with a soft collision-slack penalty. The original baseline.
- **LCP MPC** ([`lcp_mpc.py`](lcp_mpc.py)) — when `penalty_form ∈ {"l1", "l2"}`. Convex linearised MPC implementing the **Lexicographic Constraint Programming (LCP)** framework from [References/lex_constraint_programming_report_v10_2.md](../../../References/lex_constraint_programming_report_v10_2.md): 4-level priority hierarchy (Safety > Legal > Comfort > Efficiency), per-level epigraph slacks, applicability-masked rule constraints, and Algorithm 1A / 1B calibrated weights. The operational LCP planner.

Both flavours are wrapped by [`two_level_planner.py`](two_level_planner.py)'s `TwoLevelMPCPlanner`, which subclasses nuPlan's `AbstractPlanner` and is selected from the simulator's command line via Hydra (`planner=two_level_mpc_planner`).

## Pipeline diagram

```text
┌───────────────────────── nuPlan simulator ─────────────────────────┐
│                                                                    │
│  every 0.1 s (10 Hz):                                              │
│                                                                    │
│   PlannerInput ──┐                                                 │
│                  ▼                                                 │
│   TwoLevelMPCPlanner.compute_planner_trajectory(input)             │
│                  │                                                 │
│           ┌──────┴───────────┐                                     │
│           │ replan needed?   │                                     │
│           │ (timer or        │                                     │
│           │  drift > 5 m)    │                                     │
│           └──────┬───────────┘                                     │
│                  │                                                 │
│        yes ──▶ GlobalRoutePlanner.plan(ego) → ReferencePath        │
│                  │                                                 │
│      every tick ▼                                                  │
│   ┌──────────────────────────────────────────────────────────┐     │
│   │ penalty_form is null?                                    │     │
│   │   yes ─▶ MPCTrajectoryPlanner.solve()                    │     │
│   │           single-tier CasADi/IPOPT OCP                   │     │
│   │   no  ─▶ LCPTrajectoryPlanner._solve_lcp_tick()          │     │
│   │           1. MapLifter.view()  → ego-local map           │     │
│   │           2. ruleset.encode_all(ctx)  → LCPRulePack       │     │
│   │           3. BicycleLinearisation.linearise_trajectory   │     │
│   │           4. SLP outer iter (default 1, up to 5):        │     │
│   │              runtime_mode = "ws"      → LCPMPC WS solve  │     │
│   │              runtime_mode = "cascade" → run_cascade()    │     │
│   │           5. ComplianceChecker.check (logs mismatch)     │     │
│   └──────────────────────────────────────────────────────────┘     │
│                  │                                                 │
│                  ▼                                                 │
│   list[EgoState] at dt = 0.1 s                                     │
│                  │                                                 │
│                  ▼                                                 │
│   InterpolatedTrajectory → nuPlan's TwoStageController             │
│   (LQR + kinematic bicycle) advances the simulator's ego state.    │
└────────────────────────────────────────────────────────────────────┘
```

## Module map

| File | Role |
|---|---|
| [`__init__.py`](__init__.py) | Public surface — re-exports `TwoLevelMPCPlanner`, `GlobalRoutePlanner`, `MPCTrajectoryPlanner`, `ReferencePath`, plus the MPC dataclasses |
| [`bicycle_model.py`](bicycle_model.py) | `continuous_dynamics`, `discrete_dynamics` — symbolic kinematic bicycle integrated via RK4 in CasADi |
| [`reference_path.py`](reference_path.py) | `ReferencePath` — arc-length-parameterised polyline with `project`, `sample`. Plus `reference_from_se2_polyline`, `straight_reference` builders |
| [`global_planner.py`](global_planner.py) | `GlobalRoutePlanner` — periodic BFS over lane-graph successors restricted to the scenario's route corridor |
| [`trajectory_planner.py`](trajectory_planner.py) | `MPCTrajectoryPlanner` (+ `MPCParameters`, `MPCWeights`, `MPCLimits`, `ObstacleSnapshot`) — **legacy** single-tier CasADi/IPOPT MPC with soft circular obstacle constraints |
| [`two_level_planner.py`](two_level_planner.py) | `TwoLevelMPCPlanner(AbstractPlanner)` — orchestrator exposed to nuPlan via Hydra. Branches on `penalty_form` to legacy or LCP path |
| [`lcp_mpc.py`](lcp_mpc.py) | `LCPTrajectoryPlanner` (+ `LCPParameters`, `LCPLevelSpec`, `LCPLimits`, `LinearisedRuleConstraint`, `LCPRulePack`) — **the LCP MPC**: convex linearised OCP with per-level epigraph slacks |
| [`lex_cascade.py`](lex_cascade.py) | `run_cascade(...)` — the full L+1-stage lex cascade per tick. Selected via `runtime_mode: cascade`. Formally lex-optimal at the linearised problem |
| [`weight_calibration.py`](weight_calibration.py) | `algorithm_1a` (L₁ exact equivalence, paper §9.3) + `algorithm_1b` (L₂ tolerance compliance, paper §9.4). Computes $w^\dagger$ from lex-cascade gradients via Fourier–Motzkin + Chebyshev-centre LP |
| [`slp_linearisation.py`](slp_linearisation.py) | `BicycleLinearisation` — affine step maps around the warm-start, plus `sqp_convergence_metric` for the SLP outer loop |
| [`rule_encoder.py`](rule_encoder.py) | The **16 active rule encoders** plus 3 `StubRule` placeholders. `make_default_ruleset()` returns a `RuleSet` with three priority-level lists. Each encoder translates one observer rule into per-tick affine constraints `(a, b, e)` |
| [`compliance_checker.py`](compliance_checker.py) | `ComplianceChecker` — runtime safety net comparing $b_\epsilon(z_{\mathrm{ws}})$ against the cached $b_\epsilon(z_{\mathrm{lex}}^\star)$ |
| [`calibration_cache.py`](calibration_cache.py) | `CalibrationCache` — JSON-backed cache keyed on `(scenario_class, penalty_form, epsilon_per_level)` for per-scenario-class $w^\dagger$. Resolves to `HEURISTIC_DEFAULTS` on miss |
| [`map_lifter.py`](map_lifter.py) | `MapLifter` — per-tick lifting of nuPlan map data (lanes / walkways / crosswalks / TLs / stop lines) into the ego-local frame, producing a `MapHorizonView` consumed by the rule encoders |
| [`config/planner/two_level_mpc_planner.yaml`](config/planner/two_level_mpc_planner.yaml) | Hydra config (`_target_: lexicone.planning.two_level_planner.TwoLevelMPCPlanner`) |
| [`docs/full_rule_wiring_plan.md`](docs/full_rule_wiring_plan.md) | Design doc for the rule-encoder phase — the source of truth for the 16-rule LCP wiring decisions |
| [`tests/`](tests/) | **12 test files / 98 tests**. See §Tests below |

## Theory — the legacy MPC

### Kinematic bicycle model

State at the **rear axle**: $x = [p_x, p_y, \psi, v]^T$. Control: $u = [a, \delta]^T$ (longitudinal acceleration, tire steering angle). With the rear-axle convention the slip angle vanishes ($\beta = 0$), giving

```
ṗx  = v cos ψ
ṗy  = v sin ψ
ψ̇   = (v / L) tan δ        ← L = vehicle wheel base
v̇   = a
```

Integrated by classical RK4 with $dt = 0.1$ s. The CasADi `Function` `discrete_dynamics(wheel_base, dt) -> ca.Function(x, u) -> x_next` is built once at construction and reused for every dynamics-constraint instantiation in the OCP.

### Legacy OCP

The legacy [`MPCTrajectoryPlanner.solve`](trajectory_planner.py) solves a single flat-weight nonlinear MPC:

```
minimise   Σₖ  w_pos · ‖pₖ − p_ref,k‖²
         + Σₖ  w_psi · (1 − cos(ψₖ − ψ_ref,k))
         + Σₖ  w_speed · (vₖ − v_ref,k)²
         + Σₖ  w_u · (aₖ² + δₖ²)
         + Σₖ  w_du · (Δaₖ² + Δδₖ²)
         + Σⱼ,ₖ w_slack · slackⱼ,ₖ²
subject to
   X[0] = x₀
   X[k+1] = f_RK4(X[k], U[k], dt)
   0 ≤ vₖ ≤ v_cap[k], a_min ≤ aₖ ≤ a_max, |δₖ| ≤ δ_max
   |Δaₖ| ≤ jerk_max · dt, |Δδₖ| ≤ steer_rate_max · dt
   (xₖ − oⱼ,x)² + (yₖ − oⱼ,y)² + slackⱼ,k² ≥ (rⱼ + r_ego)²   (soft collision)
   slackⱼ,k ≥ 0
```

Horizon $N = 30$ steps over 3 s. Solver: IPOPT through CasADi's `Opti`, warm-started each tick from the previous solution shifted by one step.

Three design choices reappear in the LCP variant: heading via $1 - \cos(\Delta\psi)$ (smooth across $\pm\pi$ wrap), soft circular collision constraints (avoids instantaneous infeasibility in dense traffic), and a per-step velocity cap that decays at $a_{\min}$ so the LQR tracker downstream actually slows the ego when our `v` field drops.

## Theory — the LCP MPC

The LCP framework's full specification is in [References/lex_constraint_programming_report_v10_2.md](../../../References/lex_constraint_programming_report_v10_2.md). We summarise the load-bearing claims:

**Setting.** A convex constrained problem with $L$ priority-ordered constraint groups ($i = 1, \ldots, L$), each encoding a violation functional $V_i(z) = \sum_{j,k} \phi(g_{i,j,k}(z))$ where $\phi = [\cdot]_+$ (L₁) or $[\cdot]_+^2$ (L₂), plus a performance objective $J(z)$.

**Lex cascade.** $L+1$ sequential NLP solves: $V_1^\star = \min V_1$, $V_2^\star = \min V_2 \text{ s.t. } V_1 \le V_1^\star$, etc. Lex-optimal by construction.

**Weighted sum.** One NLP: $z_{\mathrm{ws}}^\star(w) = \arg\min \sum_i w_i V_i(z) + J(z)$. Fast but priority-blind unless $w$ is calibrated.

**Equivalence (paper Theorem 4.1).** $z_{\mathrm{lex}}^\star \in \arg\min$ WS$(w)$ iff $w$ lies in the normal cone $\widehat{\Omega}(p^\star)$ of the upper image at the lex point $p^\star$. Equivalently, $w$ lies in the unit-performance slice $\Omega(p^\star)$.

### How the LCP MPC realises this

[`LCPTrajectoryPlanner`](lcp_mpc.py) builds a convex linearised MPC whose decision variables include per-level slack matrices $T_i \in \mathbb{R}^{\mathrm{slots}_i \times N}$ for $i = 1, \ldots, L$. Per-level slack inequalities (epigraph lift) are

$$
a_{i,j,k}^T X[:, k] + b_{i,j,k}^T U[:, k] + e_{i,j,k} \le T_i[j, k], \qquad T_i[j, k] \ge 0
$$

where $(a_{i,j,k}, b_{i,j,k}, e_{i,j,k})$ are the per-tick affine coefficients supplied by the rule encoder for level $i$, slot $j$, step $k$. The mask $\mu_{i,j,k} \in \{0, 1\}$ enters by setting $e_{i,j,k} = -M$ for inactive slots, making the constraint trivially satisfied at $T_{i,j,k} = 0$.

The cost is

$$
J(z) + \sum_i w_i \, V_i(T_i) + \eta \sum_{i,j,k} T_{i,j,k}^2
$$

with $V_i(T_i) = \sum_{j,k} T_{i,j,k}$ under L₁ and $V_i(T_i) = \sum_{j,k} T_{i,j,k}^2$ under L₂. The Tikhonov regulariser $\eta = 10^{-6}$ on the L₁ slacks keeps the reduced Hessian positive-definite near the solution (see paper §14.6); it leaves the L₁ semantics unbiased to first order.

### The 16 rule encoders

[`make_default_ruleset()`](rule_encoder.py) returns three priority-level lists wrapping 16 active encoders + 3 stubs. Active set:

| Level | Encoder | Observer rule(s) | Convex form |
|---|---|---|---|
| L₁ (Safety) | `CollisionRule` (slots = 8) | `9r0` + `10r0` | Per-agent inflated-circle keep-out |
| L₁ | `LaneCorridorRule` (slots = 2) | `7r0` | Two linear half-planes around the route centreline |
| L₁ | `SidewalkDriveRule` (slots = 4) | `7r5` | Closest-face half-plane of nearest walkway polygon |
| L₂ (Legal) | `SpeedLimitRule` (slots = 1) | `3r0` | $v_k - v_{\lim}(s_k) \le 0$ |
| L₂ | `OpposingLaneRule` (slots = 2) | `7r2` | Half-plane vs nearest opposing-direction lane |
| L₂ | `OneWayDirectionRule` (slots = 2) | `7r3` | Half-plane variant with wider tolerance |
| L₂ | `TrafficLightRule` (slots = 1) | `7r1` | $x_{\mathrm{ego},k} - (x_{\mathrm{stop}} - \mathrm{buf}) \le 0$ when controlling TL is RED/YELLOW |
| L₃ (Comfort) | `SafeHeadwayRule` (slots = 1) | `3r3` | $t_{\mathrm{hw}} v_k + d_{\min} - \mathrm{gap}_k \le 0$ |
| L₃ | `LongitudinalComfortRule` (slots = 2) | `0r2` | $\|a_{x,k}\| \le a_{x,\max}^{\mathrm{comf}}$ |
| L₃ | `LateralAccelerationRule` (slots = 2) | `1r11` | $\|a_{y,k}\| \le a_{y,\max}^{\mathrm{comf}}$ |
| L₃ | `LateralClearanceRule` (slots = 2) | `3r5` | Y-band keep-out vs adjacent agents |
| L₃ | `LateralComfortRule` (slots = 2) | `0r3` | Soft cap on $|\delta|$ as cross-step jerk proxy |

Per-level slot budgets: **16 / 7 / 11** for L₁ / L₂ / L₃, so the LCP MPC has 34 epigraph slack variables per step × 30 steps $\approx 1020$ extra decision variables over the legacy MPC.

Three rule IDs are present as `StubRule` placeholders (slot budget reserved but masks always 0): `10r5` (bike-lane encroachment — nuPlan-mini has no bike lanes), `7r4` (stop-in-crosswalk — multi-tick), `3r6` (lane intrusion — multi-agent state machine). The remaining **9 observer rules** (`10r3`, `10r4`, `9r1`, `8r0`, `8r1`, `2r2`, `1r0`, `1r2`, `1r5`) are inherently multi-tick state machines (paper §14.5) and have no LCP encoder — they stay observer-only.

The partition (16 MPC-controlled rules vs 12 invariant rules = 3 stubs + 9 observer-only) is asserted at import time by [`tests/test_rule_level_mapping.py`](tests/test_rule_level_mapping.py); a drift fails the test.

### Runtime modes and SLP outer iteration

The YAML key `runtime_mode` chooses:

- `"ws"` — single weighted-sum solve at the calibrated $w^\dagger$. ~3 min per 16-scenario batch cell. Operational mode.
- `"cascade"` — full $L+1$-stage lex cascade per tick via [`run_cascade`](lex_cascade.py). Formally lex-optimal at the linearised problem. ~7× slower than WS (≈16 min/cell). Recommended for offline simulation per paper §14.8.

The kinematic-bicycle dynamics being nonlinear, the orchestrator's `_solve_lcp_tick` runs an **SLP outer iteration**: at each iter, linearise via `BicycleLinearisation` around the warm-start, solve the inner OCP, re-warm-start with the new solution, re-linearise. Convergence is monitored by the trajectory residual

$$\rho = \max_k \|z_k^{(j+1)} - z_k^{(j)}\|_2 \quad \text{against } \rho_{\mathrm{tol}} = 5 \times 10^{-2}\,\mathrm{m}.$$

The iteration budget `slp_max_iterations` is 1 by default (suffices at runtime for most ticks) and can be raised to 3–5 for offline calibration.

### Algorithm 1A and 1B — weight calibration

[`weight_calibration.py`](weight_calibration.py) implements the paper's calibration LPs:

- **`algorithm_1a(WeightCalibrationInputs) -> WeightCalibrationResult`** — L₁ exact equivalence. Lex KKT system → Fourier–Motzkin elimination of lex multipliers → box-bounded Chebyshev-centre LP via `scipy.optimize.linprog`. Returns $w^\dagger$ and the inscribed-ball radius $r^\dagger$.

- **`algorithm_1b(L2SensitivityInputs) -> WeightCalibrationResult`** — L₂ tolerance compliance. Reduced-KKT sensitivity → pointwise threshold $W_i(\epsilon_i, w_{-i})$ → coupled-linear Chebyshev LP. WS-verification step at the returned $w^\dagger$.

Both modules are unit-tested against paper Examples 1 + 2: Example 1 reproduces $w^\dagger = (5.5, 5.5)$, $r^\dagger = 4.5$; Example 2 reproduces $W_1(0.01, 1) \approx 77.5$.

### Why solve in an ego-local frame

nuPlan maps use UTM coordinates in the $10^5$–$10^6$ m range. Squared-distance penalties at those magnitudes destroy IPOPT's internal scaling. Both the legacy and LCP planners transform $x_0$, $X_{\mathrm{ref}}$, obstacles, and rule constraints into a frame where the ego rear axle sits at the origin facing along $+x$ before solving, and transform the solution back into world coordinates after.

### The global planner

[`GlobalRoutePlanner`](global_planner.py) mirrors the route-extraction pattern from nuPlan's `IDMPlanner` but generalises it to handle periodic re-planning. Per call:

1. Find the starting edge: scan **every** route roadblock for an interior edge whose polygon contains the ego centre; otherwise, return the nearest edge across every route roadblock.
2. Breadth-first search over lane successors restricted to the route's interior edges; target depth = remaining roadblocks.
3. Concatenate each edge's baseline polyline (skipping near-duplicate join vertices); trim to `lookahead_m = 200 m` ahead of the ego's projection.

A `straight_reference` fallback handles the first tick before the global planner has produced a real route.

## Configuration

The Hydra config at [`config/planner/two_level_mpc_planner.yaml`](config/planner/two_level_mpc_planner.yaml) exposes every tunable. Defaults:

### Shared (both legacy and LCP modes)

| Key | Default | Meaning |
|---|---|---|
| `mpc_horizon_s` | 3.0 | Prediction horizon. Longer = smoother anticipation through curves |
| `mpc_dt_s` | 0.1 | Step size; must match the simulator's tick |
| `replan_period_s` | 8.0 | Global re-extraction cadence (lateral drift > 5 m also triggers replan) |
| `desired_speed_mps` | 12.0 | Cruise speed; capped by each lane's `speed_limit_mps` |
| `occupancy_map_radius_m` | 40.0 | Filter agents to within this radius before passing to MPC |
| `global_lookahead_m` | 200.0 | Leading window of the route handed to the MPC each tick |
| `obstacle_slot_count` | 6 | Number of obstacle constraints baked into the OCP |
| `collision_buffer_m` | 0.4 | Added to each agent's circumscribed radius |
| `max_accel_mps2` | 2.5 | $a_{\max}$ |
| `max_decel_mps2` | 3.5 | $-a_{\min}$ |
| `max_speed_mps` | 25.0 | Upper bound on $v$ |
| `max_steer_rad` | 0.5 | $\|\delta\| \le$ this |
| `max_steer_rate_radps` | 0.7 | Slew rate on $\delta$ |
| `max_jerk_mps3` | 12.0 | Slew rate on $a$ |
| `weight_pos` / `weight_heading` / `weight_speed` | 4 / 10 / 1 | Tracking weights |
| `weight_control` / `weight_control_rate` | 0.05 / 0.2 | Control regularisation |
| `weight_slack` | 500.0 | Soft-collision penalty (legacy mode only; LCP mode ignores) |

### LCP-mode keys (only active when `penalty_form` is set)

| Key | Default | Meaning |
|---|---|---|
| `penalty_form` | `null` | `null` → legacy MPC. `"l1"` → LCP with $L_1$ epigraph slacks (paper Algorithm 1A). `"l2"` → LCP with $L_2$ slacks (Algorithm 1B) |
| `runtime_mode` | `"ws"` | `"ws"` → single WS solve at $w^\dagger$. `"cascade"` → full $L+1$-stage cascade per tick |
| `weights_per_level` | `null` | Per-level weights $[w_1, w_2, w_3]$. Strings of `"auto"` resolve through the cache to calibrated or heuristic-default values |
| `epsilon_per_level` | `null` | Per-level tolerances $[\epsilon_1, \epsilon_2, \epsilon_3]$ for the compliance vector $b_\epsilon$. Required for L₂; optional for L₁ |
| `scenario_class_hint` | `""` | Cache key (typically the nuPlan scenario-type string). If empty, every cell is a fresh calibration |
| `lcp_map_radius_m` | 80.0 | Map-lifting radius around the ego (lanes / walkways / crosswalks / TLs filtered to this radius before encoding) |
| `slp_max_iterations` | 1 | SLP outer-iteration budget per tick. 1 at runtime; 3–5 for offline calibration |
| `slp_residual_tol_m` | 0.05 | SLP convergence tolerance on the trajectory residual $\rho$ |

The config is discovered by nuPlan's Hydra search-path machinery via `hydra.searchpath=[file://…]` injected by the run scripts ([`examples/12_batch_two_level_mpc_planner.py`](../../examples/12_batch_two_level_mpc_planner.py) and [`examples/13_run_protocol.py`](../../examples/13_run_protocol.py)).

## Pickling

`SimulationLog.save_to_file()` pickles the entire `SimulationLog`, including the planner instance, at the end of each simulator run. Since `casadi.Opti`, `casadi.Function`, and the symbolic decision variables are SwigPyObjects that cannot pickle, both `MPCTrajectoryPlanner` and `LCPTrajectoryPlanner` implement `__getstate__` / `__setstate__`:

- `__getstate__` returns a dict with every CasADi attribute dropped.
- `__setstate__` restores the rest and runs `_build_problem()` to rebuild the CasADi state from scratch.

A planner loaded from a saved log is fully usable again (e.g. for offline `solve()` calls), with the trade-off that warm-start state is lost on load.

## Failure modes and how they're handled

- **IPOPT solve fails (timeout / max-iter)** — the planner catches the `RuntimeError`, logs a warning, and falls back to a "coast" trajectory (zero controls forward-rolled from $x_0$).
- **BFS doesn't reach the goal** — the longest partial chain is used; warning logged.
- **No reference at the first tick** — orchestrator falls back to `straight_reference` along the ego's current heading.
- **Reference projection unstable** — orchestrator's replan condition `|lat| > 5.0 m` triggers a fresh route extraction.
- **Obstacle slots all parked at infinity** — by construction unused slots are set so their `(dx² + dy²) >= …` constraints are trivially satisfied.
- **Compliance mismatch at runtime** — `ComplianceChecker` logs a structured record; current policy is "log only" (paper §14.7). Optional fall back to cascade.

## Quick examples

### Solve one OCP standalone (no simulator) — legacy MPC

```python
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from lexicone.planning.trajectory_planner import MPCParameters, MPCTrajectoryPlanner
from lexicone.planning.reference_path import straight_reference

vp = get_pacifica_parameters()
ego = EgoState.build_from_rear_axle(
    rear_axle_pose=StateSE2(0, 0, 0),
    rear_axle_velocity_2d=StateVector2D(5.0, 0.0),
    rear_axle_acceleration_2d=StateVector2D(0.0, 0.0),
    tire_steering_angle=0.0,
    time_point=TimePoint(0),
    vehicle_parameters=vp,
)
ref = straight_reference(StateSE2(0, 0, 0), length_m=100.0, speed_limit_mps=12.0)
mpc = MPCTrajectoryPlanner(vp, MPCParameters(horizon_s=2.0, dt_s=0.1))
states = mpc.solve(ego, ref, obstacles=[])
assert len(states) == mpc.horizon_steps + 1
```

### Drive the planner in the nuPlan simulator

See [`examples/12_batch_two_level_mpc_planner.py`](../../examples/12_batch_two_level_mpc_planner.py) for batch invocation:

```bash
# Legacy MPC
python examples/12_batch_two_level_mpc_planner.py

# LCP L₁ weighted-sum
python examples/12_batch_two_level_mpc_planner.py --penalty-form l1 --runtime-mode ws

# LCP L₁ cascade (offline-grade)
python examples/12_batch_two_level_mpc_planner.py --penalty-form l1 --runtime-mode cascade
```

For the multi-seed comparative-effectiveness protocol over 5 conditions × 5 seeds × 16 scenarios, see [`examples/13_run_protocol.py`](../../examples/13_run_protocol.py).

## Tests

```bash
PYTHONPATH=/workspace/nuplan-project/workspace pytest workspace/lexicone/planning/tests/
```

**12 test files, 98 tests total**:

| File | Tests |
|---|---|
| `test_planning.py` | 12 — RK4 bicycle dynamics, ReferencePath arc-length math, legacy MPC convergence, MPC obstacle response, `TwoLevelMPCPlanner` constructibility |
| `test_rule_encoder.py` | 18 — per-encoder constraint generation, slot budgets, applicability masks |
| `test_map_lifter.py` | 10 — per-tick map lifting into ego-local frame |
| `test_calibration_cache.py` | 9 — JSON cache round-trip, key derivation, heuristic-default fallback |
| `test_slp_linearisation.py` | 8 — affine step maps, SQP convergence metric, finite-difference jacobians |
| `test_weight_calibration_l2.py` | 8 — Algorithm 1B; reproduces paper Example 2 |
| `test_weight_calibration_l1.py` | 7 — Algorithm 1A; reproduces paper Example 1 ($w^\dagger = (5.5, 5.5)$, $r^\dagger = 4.5$) |
| `test_lcp_mpc.py` | 6 — `LCPTrajectoryPlanner` OCP construction and per-step solves |
| `test_lcp_orchestrator.py` | 6 — end-to-end LCP path through `TwoLevelMPCPlanner._solve_lcp_tick` |
| `test_compliance_checker.py` | 6 — runtime $b_\epsilon$ comparison and mismatch logging |
| `test_lex_cascade.py` | 4 — `run_cascade` per-stage solves and lex-optimality |
| `test_integration_lcp.py` | 4 — full integration over a single-scenario nuPlan log |
| `test_rule_level_mapping.py` | 3 — asserts the 16 MPC-controlled IDs + 9 observer-only IDs partition the 25-rule registry |
