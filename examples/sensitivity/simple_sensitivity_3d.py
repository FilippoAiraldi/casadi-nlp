r"""
A simple example of sensitivity analysis (3d version)
=====================================================

This example builts on top of the example :ref:`simple_sensitivity_example` and extends
it to a 3d setting, i.e., we consider the sensitivities w.r.t. both parameters. Not
much else changes, so comments are sparse for brevity. For referece, the NLP here
considered is

.. math::
    \begin{aligned}
        \min_{x} \quad & (1 - x_1)^2 + p_1 (x_2 - x_1^2)^2 \\
        \text{s.t.} \quad & x_1 \ge 0 \\
            & \frac{p_2^2}{4} \le (x_1 + 0.5)^2 + x_2^2 \le p_2^2 \\
            & x_1 + x_2 \le p_1.
    \end{aligned}
"""

import casadi as cs
import matplotlib.pyplot as plt
import numpy as np

from csnlp import Nlp, wrappers

# %%
# Building the NLP
# ----------------
# The NLP is very similar, but there is an additional constraint ``"c3"`` that depends
# on the parameter :math:`p_1`. The objective function is also slightly different.

nlp = Nlp[cs.MX](sym_type="MX")
x = nlp.variable("x", (2, 1), lb=[[0], [-np.inf]])[0]
p = nlp.parameter("p", (2, 1))

nlp.minimize((1 - x[0]) ** 2 + p[0] * (x[1] - x[0] ** 2) ** 2)
g = (x[0] + 0.5) ** 2 + x[1] ** 2
nlp.constraint("c1", (p[1] / 2) ** 2, "<=", g)
_, lam2 = nlp.constraint("c2", g, "<=", p[1] ** 2)
_, lam3 = nlp.constraint("c3", cs.sum1(x), "<=", p[0])
lam = cs.vertcat(lam2, lam3)
opts = {"print_time": False, "ipopt": {"sb": "yes", "print_level": 0}}
nlp.init_solver(opts)

# %%
# Solving it for different values of :math:`p` and plotting how the optimal solution is
# affected is similar to before.

M = nlp.to_function("M", [p, x], [x], ["p", "x0"], ["x"])
p_values = [[1.4, 1.9], [2.2, 1.9], [2.2, 1.1]]
X_opt = [M(p, 0.0).full() for p in p_values]

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
        [2, 2, 0, p1 - 2], [p1 - 2, 2, 2, 2], facecolor="r", alpha=0.7, edgecolor="k"
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
    ax.set_title(rf"$p_1={p1:.1f}, \ p_2={p2:.1f}$")

plt.show()

# %%
# Sensitivity analysis
# --------------------
# For simplicity, this time we'll consider the function :math:`Z(x(p), \lambda(p), p)`
# to output scalar quantities.


def z(x, lam, p):
    return (x[1, :] ** p[0] - x[0, :]) * cs.exp(-10 * lam[0] - 2 * lam[1]) / p[1]


Z = z(x, lam, p)
Zfun = nlp.to_function("Z", [p, x], [Z], ["p", "x0"], ["z"])

# %%
# We can compute the parametric sensitivity as usual.

nlp_ = wrappers.NlpSensitivity[cs.MX](nlp)
J, H = nlp_.parametric_sensitivity(expr=Z, second_order=True)
Z_values, J_values, H_values = [], [], []
for p_value in p_values:
    sol = nlp.solve(pars={"p": p_value})
    Z_values.append(sol.value(Z).full().flatten())
    J_values.append(sol.value(J).full().flatten())
    H_values.append(sol.value(H).full())

# %%
# Visualization of the sensitivities
# ----------------------------------
# This time the visualization is a bit more complex, as we have to plot the function
# in 3d. Depending on ``N``, the plot data might be slow to compute.

N = 50  # NOTE: increase at leasure
P = np.meshgrid(np.linspace(1, 2.5, N), np.linspace(1, 2, N))
P_flattened = np.row_stack([P[0].flat, P[1].flat])
Z_grid = Zfun(P_flattened, 0).full().reshape(N, N)

fig, ax = plt.subplots(
    constrained_layout=True, subplot_kw={"projection": "3d", "computed_zorder": False}
)
ax.plot_wireframe(
    P[0], P[1], Z_grid, color="k", antialiased=False, rstride=5, cstride=5, zorder=0
)

for i in range(len(p_values)):
    p = p_values[i]
    Z_value = Z_values[i]
    J_value = J_values[i]
    H_value = H_values[i]
    clr = ax.scatter(p[0], p[1], Z_value, s=100, zorder=2).get_facecolor()
    deltaP = np.subtract(P_flattened.T, p).T
    S = Z_value + J_value @ deltaP + 0.5 * np.diag(deltaP.T @ H_value @ deltaP)
    ax.plot_surface(P[0], P[1], S.reshape(N, N), color=clr, alpha=0.5, lw=0, zorder=1)

ax.set_xlabel("$p_0$")
ax.set_ylabel("$p_1$")
ax.set_zlabel("$z$")
ax.set_xlim(1, 2.5)
ax.set_ylim(1, 2)
ax.set_zlim(-0.25, 0.03)
plt.show()
