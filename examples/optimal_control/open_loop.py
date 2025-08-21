r"""
Simple open-loop MPC controller
===============================

This example demos how an MPC controller can be built via :class:`csnlp.wrappers.Mpc`
and how it can be solved in an open-loop fashion.

We'll skip over some of the formalities of the library, as we assume you are already a
bit familiar with it. If this is not the case, for more introductory examples on
:mod:`csnlp`, see the other :ref:`introductory_examples`.

The problem is inspired by the example from
`this CasADi video <https://www.youtube.com/watch?v=JI-AyLv68Xs&t=918s>`_, where an
optimal control problem for a 2-dimensional task is tackled. The goal is to regulate the
state and action towards the origin.In mathematical terms, the state at time :math:`k`
is indicated by :math:`x_k`, and evolves according to the continuous-time dynamics

.. math::
    \dot{x} = f(x, u) = \begin{bmatrix}
        (1 - x_2^2) x_1 - x_2 + u \\ x_1
    \end{bmatrix},

where :math:`u` is the control action. We discretize the continuous-time dynamics using
a Runge-Kutta 4th order integrator, and we refer to the discrete-time dynamics as
:math:`f_d(x_k,u_k)`. Lastly, we formulate the optimal control problem

.. math::
    \begin{aligned}
        \min_{\substack{u_0,\dots,u_{N-1} \\ x_0,\dots,x_N}} \quad
            & \sum_{i=0}^{N} x_i^2 + \sum_{i=0}^{N-1} u_i^2 \\
        \text{s.t.} \quad
            & x_0 = [0, 1]^\top \\
            & x_{i+1} = f_d(x_i, u_i), \quad i = 0,\ldots,N-1.
    \end{aligned}
"""

# %%
# Multiple shooting MPC
# ----------------------
# In this section we'll first solve the problem in a multiple shooting formulation (the
# one proposed above). We start with the usual imports.

import casadi as cs
import matplotlib.pyplot as plt
import numpy as np

from csnlp import Nlp, wrappers

# %%
# We then define the continuous-time dynamics and discretize them via RK4. The final
# dynamics are conveniently packed in the function :math:`F`.

x = cs.MX.sym("x", 2)
u = cs.MX.sym("u")
ode = cs.vertcat((1 - x[1] ** 2) * x[0] - x[1] + u, x[0])
f = cs.Function("f", [x, u], [ode], ["x", "u"], ["ode"])
T = 15
N = 20
intg_options = {"simplify": True, "number_of_finite_elements": 4}
dae = {"x": x, "p": u, "ode": f(x, u)}
intg = cs.integrator("intg", "rk", dae, 0, T / N, intg_options)
res = intg(x0=x, p=u)
x_next = res["xf"]
F = cs.Function("F", [x, u], [x_next], ["x", "u"], ["x_next"])

# %%
# Then, we can move to building the MPC. Since it is a non-retroactive wrapper, an
# fresh :class:`csnlp.Nlp` instance must be passed to it. Instead of using
# :meth:`csnlp.Nlp.variable`, the wrapper exposes two convienient methods to define
# states and actions: :meth:`csnlp.wrappers.Mpc.state` and
# :meth:`csnlp.wrappers.Mpc.action`. We'll use those, and also pass some lower- and
# upper-bounds to them. After creation of states and actions, we can set the dynamics
# via a convenient method :meth:`csnlp.wrappers.Mpc.set_nonlinear_dynamics` that will
# automatically add the initial state constraint and the dynamics constraints. Lastly,
# an appropriate cost function is defined and the solver is initialized.

mpc = wrappers.Mpc[cs.SX](
    nlp=Nlp[cs.SX](sym_type="SX"), prediction_horizon=N, shooting="multi"
)
u, _ = mpc.action("u", lb=-1, ub=+1)
x = mpc.state("x", 2, lb=-0.2)  # must be created before dynamics
mpc.set_nonlinear_dynamics(F)
mpc.minimize(cs.sumsqr(x) + cs.sumsqr(u))

# %%
# Before solving, the solver is initialized, and then called with the value of the
# initial state ``x`` as initial condition.

opts = {"print_time": False, "ipopt": {"sb": "yes", "print_level": 5}}
mpc.init_solver(opts)
sol = mpc.solve_ocp([0, 1])

t = np.linspace(0, T, N + 1)
plt.plot(t, sol.value(x).T)
plt.step(t[:-1], sol.vals["u"].T.full(), "-.", where="post")
plt.legend(["x1", "x2", "u"])
plt.xlim(t[0], t[-1])
plt.xlabel("t [s]")
plt.show()

# %%
# Single shooting MPC
# -------------------
# :class:`csnlp.wrappers.Mpc` supports also single shooting. The process remains more or
# less the same, aside from the states. In this case, no state is returned by
# :meth:`csnlp.wrappers.Mpc.state`, but it is instead created only after the dynamics
# are specified via :meth:`csnlp.wrappers.Mpc.set_nonlinear_dynamics`.

mpc = wrappers.Mpc[cs.SX](
    nlp=Nlp[cs.SX](sym_type="SX"), prediction_horizon=N, shooting="single"
)
u, _ = mpc.action("u", lb=-1, ub=+1)
mpc.state("x", 2)  # does not return a symbolic variable
mpc.set_nonlinear_dynamics(F)
x = mpc.states["x"]  # only accessible after dynamics have been set
mpc.constraint("x_lb", x[:, 1:], ">=", -0.2)  # equivalent to a lb on x
mpc.minimize(cs.sumsqr(x) + cs.sumsqr(u))
mpc.init_solver(opts)
sol = mpc.solve_ocp([0, 1])

# %%
# We can again plot the solution. It should look somewhat similar to the one obtained
# with multiple shooting.

plt.plot(t, sol.value(x).T)
plt.step(t[:-1], sol.vals["u"].T.full(), "-.", where="post")
plt.legend(["x1", "x2", "u"])
plt.xlim(t[0], t[-1])
plt.xlabel("t [s]")
plt.show()
