# Inspired by https://web.casadi.org/blog/nlp_sens/


import casadi as cs
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from csnlp import Nlp, wrappers


def plot_nlp(ax: Axes, a: float, b: float, x: float, y: float) -> None:
    X, Y = np.meshgrid(*[np.linspace(-1.5, 2, 100)] * 2)
    ax.contour(X, Y, (1 - X) ** 2 + a * (Y - X**2) ** 2, 100)

    ts = np.linspace(0, 2 * np.pi, 100)
    ax.fill(
        b / 2 * np.cos(ts) - 0.5,
        b / 2 * np.sin(ts),
        facecolor="r",
        alpha=0.3,
        edgecolor="k",
    )
    cat = np.concatenate
    ax.fill(
        cat([b * np.cos(ts - np.pi) - 0.5, 10 * np.cos(np.pi - ts) - 0.5]),
        cat([b * np.sin(ts - np.pi), 10 * np.sin(np.pi - ts)]),
        facecolor="r",
        alpha=0.3,
        edgecolor="k",
    )
    ax.fill(
        [-1.5, 0, 0, -1.5], [-1.5, -1.5, 2, 2], facecolor="r", alpha=0.3, edgecolor="k"
    )
    ax.plot(x, y, "*", markersize=12, linewidth=2)
    ax.set_aspect("equal", "box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(-1.5, 2)
    ax.set_ylim(-1.5, 2)
    ax.set_title(f"a={a:.1f}, b={b:.1f}")


def z1(x, lam, p):
    return (x[1, :] ** p[1] - x[0, :]) * cs.exp(-10 * lam + p[0]) / p[1]


def z2(x):
    return x[1, :] - x[0, :]


def z3(x, p):
    return x[1, :] ** (1 / p[1]) - x[0, :]


def z4(lam, p):
    return cs.exp(-10 * lam + p[0]) / p[1]


# build the NLP
nlp = Nlp(sym_type="MX")
x = nlp.variable("x", (2, 1), lb=[[0], [-np.inf]])[0]
p = nlp.parameter("p", (2, 1))

nlp.minimize((1 - x[0]) ** 2 + p[0] * (x[1] - x[0] ** 2) ** 2)
g = (x[0] + 0.5) ** 2 + x[1] ** 2
nlp.constraint("c1", (p[1] / 2) ** 2, "<=", g)
_, lam = nlp.constraint("c2", g, "<=", p[1] ** 2)
opts = {"print_time": False, "ipopt": {"sb": "yes", "print_level": 0}}
nlp.init_solver(opts)


# Use IPOPT to solve the nonlinear optimization
p_values = (1.2, 1.45, 1.9)
fig, axs = plt.subplots(1, 3, sharey=True, constrained_layout=True, figsize=(8, 3))
M = nlp.to_function("M", [p, x], [x], ["p", "x0"], ["x"])
for p0, ax in zip(p_values, axs):
    x_ = M([0.2, p0], 0).full()
    plot_nlp(ax, 0.2, p0, x_[0], x_[1])

# How does the optimal solution vary along p?
nlp = wrappers.NlpSensitivity(nlp)
# a bunch of strange equations we want to compute sensitivity of w.r.t. p[1]
Z = cs.blockcat(
    [
        [z1(x, lam, p) ** 2, z2(x)],
        [1 / (1 + z3(x, p)), z4(lam, p)],
        [z3(x, p) * (-1 - 10 * z1(x, lam, p)), z4(lam, p) / (1 + z2(x))],
    ]
)
Zfcn = nlp.to_function("Z", [p, x], [Z], ["p", "x0"], ["z"])

fig, axs = plt.subplots(*Z.shape, constrained_layout=True)
N = 300
Ps = np.row_stack((np.full(N, 0.2), np.linspace(1, 2, N)))
Zs = np.stack([Zfcn(Ps[:, i], 0) for i in range(Ps.shape[1])], axis=-1)
for i in np.ndindex(axs.shape):
    axs[i].plot(Ps[1].flat, Zs[i].flat, "k-", lw=4)
    axs[i].set_ylim(axs[i].get_ylim() + np.array([-0.1, 0.1]))

# Parametric sensitivities of function z(x(p), lam(p))
t = np.linspace(1, 2, 1000)
J, H = nlp.parametric_sensitivity(expr=Z)
for p0, clr in zip(p_values, ["r", "g", "b"]):
    sol = nlp.solve(pars={"p": [0.2, p0]})
    z0 = sol.value(Z)
    j0 = sol.value(J)
    h0 = sol.value(H)
    for i in np.ndindex(axs.shape):
        axs[i].plot(p0, float(z0[i]), "o", color=clr, markersize=6)
        axs[i].plot(
            t,
            z0[i] + j0[i][1] * (t - p0) + 0.5 * h0[i][1, 1] * (t - p0) ** 2,
            color=clr,
        )

for i in np.ndindex(axs.shape):
    axs[i].set_xlabel("p_1")
    axs[i].set_ylabel(rf"$z_{{{str(i)[1:-1]}}}(x(p), \lambda(p), p)$")
    axs[i].set_xlim(t[0], t[-1])
plt.show()
