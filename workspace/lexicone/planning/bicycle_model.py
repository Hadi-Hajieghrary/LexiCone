"""Symbolic kinematic bicycle dynamics for the MPC.

State :math:`x = [p_x, p_y, \\psi, v]` is expressed at the **rear axle**, to match
:meth:`nuplan.common.actor_state.ego_state.EgoState.build_from_rear_axle`.
Control :math:`u = [a, \\delta]` is longitudinal acceleration and steering angle.
With the rear-axle convention the slip angle ``beta`` vanishes, so the
continuous dynamics are::

    px_dot  = v * cos(psi)
    py_dot  = v * sin(psi)
    psi_dot = v / L * tan(delta)
    v_dot   = a

The discrete step is integrated with classical RK4.
"""

from __future__ import annotations

import casadi as ca


NX = 4
NU = 2


def continuous_dynamics(x: ca.SX, u: ca.SX, wheel_base: float) -> ca.SX:
    psi = x[2]
    v = x[3]
    a = u[0]
    delta = u[1]
    return ca.vertcat(
        v * ca.cos(psi),
        v * ca.sin(psi),
        v / wheel_base * ca.tan(delta),
        a,
    )


def discrete_dynamics(wheel_base: float, dt: float) -> ca.Function:
    """Returns a CasADi Function ``f(x, u) -> x_next`` integrated via RK4."""
    x = ca.SX.sym("x", NX)
    u = ca.SX.sym("u", NU)

    k1 = continuous_dynamics(x, u, wheel_base)
    k2 = continuous_dynamics(x + 0.5 * dt * k1, u, wheel_base)
    k3 = continuous_dynamics(x + 0.5 * dt * k2, u, wheel_base)
    k4 = continuous_dynamics(x + dt * k3, u, wheel_base)
    x_next = x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

    return ca.Function("bicycle_rk4", [x, u], [x_next], ["x", "u"], ["x_next"])
