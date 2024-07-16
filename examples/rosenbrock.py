r"""
A simple optimization problem: Rosenbrock function
==================================================

This example illustrates the basic usage of :class:`csnlp.Nlp` in solving a simple
optimization problem, taken from
`this CasADi blog post <https://web.casadi.org/blog/opti/>`_

Consider the parametric (in :math:`r`) constrained Rosenbrock function

.. math::

    \min_{x}{ (1 - x_1)^2 + (x_2 - x_1^2)^2 } \text{ s.t. } x_1^2 + x_2^2 \leq r.
"""

# %%
# Creating and solving the problem
# --------------------------------
# The imports are pretty standard, as if we were using ``casadi`` alone. We just need to
# additionally import the :class:`csnlp.Nlp` class.

import casadi as cs
import matplotlib.pyplot as plt
import numpy as np

from csnlp import Nlp

# %%
# In order to build the optimization problem, we first create an instance of
# :class:`csnlp.Nlp` (instructed to use :class:`casadi.MX` under the hood) and
# primal variable :math:`x` and parameter :math:`r`.

nlp = Nlp[cs.MX](sym_type="MX")
x = nlp.variable("x", (2, 1))[0]
r = nlp.parameter("r")

# %%
# Then, formulate the objective and the constraint, and initialize the solver (this is
# mandatory before any solving run). Under the hood, by default, the solver is set to
# IPOPT.
#
# Note that for the constraint we save the corresponding Lagrange multiplier
# :math:`\lambda` in the variable `lam` to be used later.


def rosenbrock(x):
    return (1 - x[0]) ** 2 + (x[1] - x[0] ** 2) ** 2


nlp.minimize(rosenbrock(x))
_, lam = nlp.constraint("con1", cs.sumsqr(x), "<=", r)
nlp.init_solver()

# %%
# For a given value of :math:`r`, e.g., :math:`r=1`, we can solve the problem as
# follows

r_value = 1
sol = nlp.solve(pars={"r": r_value})

# %%
# And we can also visualize the solution

_, ax = plt.subplots(constrained_layout=True)

X, Y = np.meshgrid(np.linspace(0, 1.5, 100), np.linspace(-0.5, 1.5, 100))
F = rosenbrock((X, Y))
contour = ax.contour(X, Y, F, levels=100, cmap="viridis")

theta = np.linspace(0, 2 * np.pi, 100)
x_circle = np.cos(theta)
y_circle = np.sin(theta)
ax.plot(x_circle, y_circle, "k-")

x_opt = sol.value(x)
ax.plot(x_opt[0], x_opt[1], "r*", markersize=20)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_aspect("equal")
ax.set_xlim(0, 1.5)
ax.set_ylim(-0.5, 1.5)
plt.show()

# %%
# Computing multipliers and sensitivities
# ---------------------------------------
# What's cooler, we can set the value of :math:`r` at runtime to any valid value, run
# the solver, and compute the sensitivity of the objective with respect to :math:`r`
# as the value of the mutiplier :math:`\lambda` at the optimal point.

r_values = np.linspace(1, 3, 25)
f_values = []
lam_values = []
for r_value in r_values:
    sol = nlp.solve(pars={"r": r_value})
    f_values.append(sol.f)
    lam_values.append(sol.value(lam))

# %%
# In the figure we plot the value of the optimal value :math:`f(x^\star; r)` as a
# function of the parameter :math:`r`, and the tangent line at each point is computed
# via the corresponding optimal multiplier :math:`\lambda^\star(r)`.

ts = np.linspace(-0.02, 0.02)

fig, ax = plt.subplots(constrained_layout=True)
ax.plot(r_values, f_values, "o")
for r_value, f_value, lam_value in zip(r_values, f_values, lam_values):
    ax.plot(r_value + ts, -lam_value * ts + f_value, "r-")

ax.set_xlabel("Value of r")
ax.set_ylabel("Objective value at solution")
plt.show()
