"""Unit tests for the lexicone.planning package.

Every test here is fully standalone — none of them hit a live nuplan-devkit
map. The dynamics, reference-path math, MPC convergence (straight road, with
an obstacle), and the orchestrator's constructibility / observation-type
contract are all checked against synthetic inputs. No skip markers are used.
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import (
    Point2D,
    StateSE2,
    StateVector2D,
    TimePoint,
)
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters

from lexicone.planning.bicycle_model import discrete_dynamics
from lexicone.planning.reference_path import (
    ReferencePath,
    reference_from_se2_polyline,
    straight_reference,
)
from lexicone.planning.trajectory_planner import (
    MPCParameters,
    MPCTrajectoryPlanner,
    ObstacleSnapshot,
)
from lexicone.planning.two_level_planner import TwoLevelMPCPlanner


# ---------------------------------------------------------------------------
# bicycle_model
# ---------------------------------------------------------------------------


def test_bicycle_rk4_zero_control_constant_velocity():
    fn = discrete_dynamics(wheel_base=3.0, dt=0.1)
    x0 = np.array([0.0, 0.0, 0.0, 5.0])
    u0 = np.array([0.0, 0.0])
    x1 = np.asarray(fn(x0, u0)).flatten()
    # Pure forward motion, no steering, no acceleration.
    assert x1[0] == pytest.approx(0.5, abs=1e-9)
    assert x1[1] == pytest.approx(0.0, abs=1e-9)
    assert x1[2] == pytest.approx(0.0, abs=1e-9)
    assert x1[3] == pytest.approx(5.0, abs=1e-9)


def test_bicycle_rk4_constant_acceleration():
    fn = discrete_dynamics(wheel_base=3.0, dt=0.1)
    x = np.array([0.0, 0.0, 0.0, 0.0])
    u = np.array([1.0, 0.0])
    for _ in range(10):
        x = np.asarray(fn(x, u)).flatten()
    # After 1.0 s of 1 m/s^2 from rest: v == 1, displacement == 0.5 m.
    assert x[3] == pytest.approx(1.0, abs=1e-9)
    assert x[0] == pytest.approx(0.5, abs=1e-9)


# ---------------------------------------------------------------------------
# reference_path
# ---------------------------------------------------------------------------


def test_reference_path_straight_line_arc_length_and_sample():
    xy = np.column_stack([np.linspace(0.0, 10.0, 11), np.zeros(11)])
    vlim = np.full(11, 12.0)
    ref = ReferencePath(xy, vlim)
    assert ref.length == pytest.approx(10.0, abs=1e-9)
    sample = ref.sample(5.5)
    assert sample.x == pytest.approx(5.5, abs=1e-9)
    assert sample.y == pytest.approx(0.0, abs=1e-9)
    assert sample.psi == pytest.approx(0.0, abs=1e-9)
    # Out-of-range queries clamp.
    assert ref.sample(-1.0).x == pytest.approx(0.0, abs=1e-9)
    assert ref.sample(100.0).x == pytest.approx(10.0, abs=1e-9)


def test_reference_path_projection_lateral_offset_sign():
    xy = np.column_stack([np.linspace(0.0, 10.0, 11), np.zeros(11)])
    vlim = np.full(11, 12.0)
    ref = ReferencePath(xy, vlim)
    s, lat = ref.project(Point2D(3.0, 1.5))
    assert s == pytest.approx(3.0, abs=1e-9)
    # Path direction = +x, so a point at +y is to the LEFT → positive lateral.
    assert lat == pytest.approx(1.5, abs=1e-9)
    s2, lat2 = ref.project(Point2D(3.0, -1.5))
    assert s2 == pytest.approx(3.0, abs=1e-9)
    assert lat2 == pytest.approx(-1.5, abs=1e-9)


def test_reference_path_filters_duplicate_points():
    xy = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    vlim = np.array([12.0, 12.0, 12.0, 12.0, 12.0])
    ref = ReferencePath(xy, vlim)
    # Length must equal the sum of distinct-segment lengths, not zero.
    assert ref.length == pytest.approx(2.0, abs=1e-9)


def test_reference_from_se2_polyline_respects_default_speed_limit():
    poly = [StateSE2(0.0, 0.0, 0.0), StateSE2(1.0, 0.0, 0.0), StateSE2(2.0, 0.0, 0.0)]
    ref = reference_from_se2_polyline(poly, [None, None, None], default_speed_limit_mps=8.0)
    assert ref.sample(1.0).v_limit == pytest.approx(8.0, abs=1e-9)


def test_straight_reference_extends_along_heading():
    origin = StateSE2(2.0, 3.0, math.pi / 2)
    ref = straight_reference(origin, length_m=10.0, speed_limit_mps=8.0, num_points=21)
    sample = ref.sample(5.0)
    assert sample.x == pytest.approx(2.0, abs=1e-9)
    assert sample.y == pytest.approx(8.0, abs=1e-9)
    assert sample.psi == pytest.approx(math.pi / 2, abs=1e-6)


# ---------------------------------------------------------------------------
# MPC trajectory planner
# ---------------------------------------------------------------------------


def _build_mpc(**overrides) -> MPCTrajectoryPlanner:
    params = MPCParameters(
        horizon_s=overrides.pop("horizon_s", 1.5),
        dt_s=overrides.pop("dt_s", 0.1),
        desired_speed_mps=overrides.pop("desired_speed_mps", 8.0),
        obstacle_slot_count=overrides.pop("obstacle_slot_count", 2),
    )
    return MPCTrajectoryPlanner(vehicle_parameters=get_pacifica_parameters(), params=params)


def _build_ego(x: float = 0.0, y: float = 0.0, heading: float = 0.0, v: float = 0.0) -> EgoState:
    return EgoState.build_from_rear_axle(
        rear_axle_pose=StateSE2(x, y, heading),
        rear_axle_velocity_2d=StateVector2D(v, 0.0),
        rear_axle_acceleration_2d=StateVector2D(0.0, 0.0),
        tire_steering_angle=0.0,
        time_point=TimePoint(1_000_000),
        vehicle_parameters=get_pacifica_parameters(),
    )


def _straight_reference() -> ReferencePath:
    return straight_reference(StateSE2(0.0, 0.0, 0.0), length_m=200.0, speed_limit_mps=8.0)


def test_mpc_returns_trajectory_of_expected_length():
    mpc = _build_mpc()
    states = mpc.solve(_build_ego(v=5.0), _straight_reference(), obstacles=[])
    assert len(states) == mpc.horizon_steps + 1
    dt_us = int(mpc.dt * 1e6)
    times = [s.time_point.time_us for s in states]
    assert all(t1 - t0 == dt_us for t0, t1 in zip(times[:-1], times[1:]))


def test_mpc_straight_road_tracks_reference():
    mpc = _build_mpc()
    states = mpc.solve(_build_ego(v=5.0), _straight_reference(), obstacles=[])
    # Lateral deviation from y=0 should stay tiny and forward speed should approach the target.
    max_y = max(abs(s.rear_axle.y) for s in states)
    assert max_y < 0.2, f"lateral deviation {max_y:.3f} m exceeded threshold"
    terminal_v = float(states[-1].dynamic_car_state.rear_axle_velocity_2d.magnitude())
    assert terminal_v > 5.0  # MPC should accelerate toward desired_speed=8 m/s
    assert terminal_v <= 8.1


def test_mpc_decelerates_when_obstacle_is_ahead():
    mpc = _build_mpc(desired_speed_mps=8.0)
    free_states = mpc.solve(_build_ego(v=6.0), _straight_reference(), obstacles=[])
    mpc.reset()
    obstacle = ObstacleSnapshot(x=12.0, y=0.0, radius=2.0)
    blocked_states = mpc.solve(_build_ego(v=6.0), _straight_reference(), obstacles=[obstacle])
    # With an obstacle 12 m ahead the MPC must not roll forward as far as in the unobstructed case.
    free_terminal_x = free_states[-1].rear_axle.x
    blocked_terminal_x = blocked_states[-1].rear_axle.x
    assert blocked_terminal_x < free_terminal_x


# ---------------------------------------------------------------------------
# Orchestrator construction / fallback path
# ---------------------------------------------------------------------------


def test_two_level_planner_name_and_observation_type():
    planner = TwoLevelMPCPlanner(mpc_horizon_s=1.0, mpc_dt_s=0.1, replan_period_s=5.0)
    assert planner.name() == "TwoLevelMPCPlanner"
    from nuplan.planning.simulation.observation.observation_type import DetectionsTracks

    assert planner.observation_type() is DetectionsTracks


def test_two_level_planner_constructible_from_yaml_keys():
    # The Hydra YAML must instantiate without errors via the exposed keys.
    kwargs = dict(
        mpc_horizon_s=1.0,
        mpc_dt_s=0.1,
        replan_period_s=5.0,
        desired_speed_mps=10.0,
        occupancy_map_radius_m=30.0,
        global_lookahead_m=120.0,
        obstacle_slot_count=4,
        collision_buffer_m=0.4,
        max_accel_mps2=2.0,
        max_decel_mps2=3.0,
        max_speed_mps=20.0,
        max_steer_rad=0.5,
        max_steer_rate_radps=0.5,
        max_jerk_mps3=4.0,
        weight_pos=1.0,
        weight_heading=4.0,
        weight_speed=0.5,
        weight_control=0.1,
        weight_control_rate=0.5,
        weight_slack=1000.0,
    )
    TwoLevelMPCPlanner(**kwargs)
