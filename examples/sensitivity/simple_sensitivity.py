r"""
.. _simple_sensitivity_example:

A simple example of sensitivity analysis
========================================

In this example we explore the parametric sensitivity of an NLP problem. We will use the
:mod:`csnlp.wrappers.NlpSensitivity` wrapper to achieve so, but we'll skip over some of
the formalities of the library, as we assume you are already a bit familiar with it. If
this is not the case, for more introductory examples on :mod:`csnlp`, see the other
:ref:`introductory_examples`.

The problem is inspired the example from
`this CasADi blog post <https://web.casadi.org/blog/nlp_sens/>`_. The following problem
is considered

.. math::
    \begin{aligned}
        \min_{x} \quad & (1 - x_1)^2 + p_1 (x_2 - x_1^2)^2 \\
        \text{s.t.} \quad & x_1 \ge 0 \\
            & \frac{p_2^2}{4} \le (x_1 + 0.5)^2 + x_2^2 \le p_2^2.
    \end{aligned}

where :math:`p = [p_1, p_2]^\top` are the parameters of the problem.
"""

# %%
# We start with the usual imports.

import casadi as cs
import matplotlib.pyplot as plt
import numpy as np

from csnlp import Nlp, wrappers

# %%
# Building the NLP
# ----------------
# We then build the NLP in the usual way. We create the variables and the parameters of
# the problem, and then we add the objective function and the constraints. Lastly, we
# initialize the solver.

nlp = Nlp[cs.MX](sym_type="MX")
x = nlp.variable("x", (2, 1), lb=[[0], [-np.inf]])[0]
p = nlp.parameter("p", (2, 1))

nlp.minimize((1 - x[0]) ** 2 + p[0] * (x[1] - x[0] ** 2) ** 2)
g = (x[0] + 0.5) ** 2 + x[1] ** 2
nlp.constraint("c1", (p[1] / 2) ** 2, "<=", g)
_, lam = nlp.constraint("c2", g, "<=", p[1] ** 2)
nlp.init_solver({"print_time": False, "ipopt": {"sb": "yes", "print_level": 0}})

# %%
# To spice things up, we'll convert the NLP to a :class:`casadi.Function` via the method
# :meth:`csnlp.Nlp.to_function`. This streamlines the process of evaluating the NLP
# multiple times. The function takes as input the parametrization and the initial guess
# for the primal variables, and returns the optimal primal solution.

M = nlp.to_function("M", [p, x], [x], ["p", "x0"], ["x"])

# %%
# We can call ``M`` to solve the optimization for different values of :math:`p`.

p_values = [[0.2, 1.2], [0.2, 1.45], [0.2, 1.9]]
X_opt = [M(p, 0.0).full() for p in p_values]

# %%
# A nice visualization follows. In red we have the infeasible region, while the star
# marks the optimal solution. The contours represent the objective function's levels.

_, axs = plt.subplots(1, 3, sharey=True, constrained_layout=True)

ts = np.linspace(0, 2 * np.pi, 100)
cos_ts, sin_ts = np.cos(ts), np.sin(ts)
X, Y = np.meshgrid(*[np.linspace(-1.5, 2, 100)] * 2)

for ax, (p1, p2), (x1_opt, x2_opt) in zip(axs, p_values, X_opt):
    ax.contour(X, Y, (1 - X) ** 2 + p1 * (Y - X**2) ** 2, 100, alpha=0.5)
    ax.fill(
        p2 / 2 * cos_ts - 0.5, p2 / 2 * sin_ts, facecolor="r", alpha=0.7, edgecolor="k"
    )
    ax.fill(
        np.concatenate([p2 * np.cos(ts - np.pi) - 0.5, 10 * np.cos(np.pi - ts) - 0.5]),
        np.concatenate([p2 * np.sin(ts - np.pi), 10 * sin_ts]),
        facecolor="r",
        alpha=0.7,
        edgecolor="k",
    )
    ax.fill(
        [-1.5, 0, 0, -1.5], [-1.5, -1.5, 2, 2], facecolor="r", alpha=0.7, edgecolor="k"
    )
    ax.plot(x1_opt, x2_opt, "*", markersize=12, linewidth=2)
    ax.set_aspect("equal", "box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(-1.5, 2)
    ax.set_ylim(-1.5, 2)
    ax.set_title(rf"$p_1={p1:.2f}, \ p_2={p2:.2f}$")

plt.show()

# %%
# From this plot, we can appreciate how the optimal solution changes as we vary the
# parametrization of the problem, especially when the constraint becomes inactive. This
# begs the question: how does the optimal solution vary along :math:`p_2`? We can answer
# this question via sensitivity analysis.

# %%
# Sensitivity analysis
# --------------------
# To be as general as possible, consider a generic function that maps the optimal primal
# and dual variables and parameters to a vector of expressions. We call this mapping
# :math:`Z(x(p), \lambda(p), p)`. Note that :math:`x(p)` and :math:`\lambda(p)` are the
# functions of the parameters :math:`p`, as we just saw, but this relationship is
# implicit and most of the time difficult to express analytically. Our goal is to
# numerically compute its sensitivity w.r.t. :math:`p_2`, i.e.,
#
# .. math::
#   \frac{\partial Z}{\partial p_2} \quad \text{and} \quad
#   \frac{\partial^2 Z}{\partial p_2^2}.
#
# In order to to do, let us build a very generic function ``Zfcn`` (note that its
# entries have no clear meaning, are just arbritary continuous functions). Again, we
# can use the :meth:`csnlp.Nlp.to_function` method to convert it to a CasADi function
# for ease of evaluations.

z1 = lambda x: x[1, :] - x[0, :]
z2 = lambda x, lam, p: (x[1, :] ** p[1] - x[0, :]) * cs.exp(-10 * lam + p[0]) / p[1]
z3 = lambda x, p: x[1, :] ** (1 / p[1]) - x[0, :]
z4 = lambda lam, p: cs.exp(-10 * lam + p[0]) / p[1]
Z = cs.vertcat(
    z1(x),
    z2(x, lam, p) ** 2,
    1 / (1 + z3(x, p)),
    z4(lam, p),
    z3(x, p) * (-1 - 10 * z2(x, lam, p)),
    z4(lam, p) / (1 + z1(x)),
)
Zfcn = nlp.to_function("Z", [p, x], [Z], ["p", "x0"], ["z"])

# %%
# To enable the computation of the parametric sensitivities, we need to wrap the NLP
# in a :class:`csnlp.wrappers.NlpSensitivity` object. This takes in the target
# parameter :math:`p_2`, because that's the parameter we are interested in.

nlp_ = wrappers.NlpSensitivity[cs.MX](nlp, target_parameters=p[1])

# %%
# The symbolic representation of the parametric sensitivities (jacobian and hessian) is
# obtained via the :meth:`csnlp.wrappers.NlpSensitivity.parametric_sensitivity` method
# as follows. By default, the method does not compute second order information, because
# it can be very costly, and, if no expression is passed, will compute the sensitivity
# of the primal variables w.r.t. the ``target_parameters``.

J, H = nlp_.parametric_sensitivity(expr=Z, second_order=True)

# %%
# Once the symbols are available, we can evaluate them numerically after a solution is
# computed. We do so for different values of :math:`p_2`, and store the results in some
# lists for later plotting.

Z_values, J_values, H_values = [], [], []
for p_value in p_values:
    sol = nlp.solve(pars={"p": p_value})
    Z_values.append(sol.value(Z).full().flatten())  # also evaluate the actual function
    J_values.append(sol.value(J).full().flatten())
    H_values.append(sol.value(H).full().flatten())

# %%
# Visualization of the sensitivities
# ----------------------------------
# Before jumping into the visualization of the sensitivities themselves, let's compute
# the function :math:`Z` for a range of realizations of :math:`p_2` (with
# :math:`p_1=0.2`). This will also serve as a baseline to understand whether the
# sensitivities follow the actual plot.

N = 300
p_values_all = np.row_stack((np.full(N, 0.2), np.linspace(1, 2, N)))
z_values_all = np.concatenate([Zfcn(p, 0.0) for p in p_values_all.T], axis=1)

# %%
# We can not pot the function :math:`Z` for the different values of :math:`p_2`. After
# that, we can plot the parametric sensitivities. To do so, we plot the second order
# Taylor expansion that approximates the function :math:`Z` around the optimal solution
# for different values of :math:`p_2`. This approximation obviously makes use of the
# gradient and hessian information of the function at that point.

fig, axs = plt.subplots(3, 2, sharex=True, constrained_layout=True)
for i, ax in enumerate(axs.flat):
    ax.plot(p_values_all[1], z_values_all[i], "k-", lw=4)
    ax.set_ylabel(f"$Z_{i}$")
    if i in (4, 5):
        ax.set_xlabel("$p_2$")
    ax.set_ylim(ax.get_ylim() + np.asarray([-0.1, 0.1]))

t = np.linspace(1, 2, 100)
for i in range(len(p_values)):
    p = p_values[i][1]
    Z_value = Z_values[i]
    J_value = J_values[i]
    H_value = H_values[i]
    for ax, z, j, h in zip(axs.flat, Z_value, J_value, H_value):
        ax.plot(p, z, "o", color=f"C{i}", markersize=6)
        ax.plot(t, z + j * (t - p) + 0.5 * h * (t - p) ** 2, color=f"C{i}", ls="--")

plt.show()
