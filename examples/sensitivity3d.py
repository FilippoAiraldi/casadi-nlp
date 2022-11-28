# Inspired by https://web.casadi.org/blog/nlp_sens/


import casadi as cs
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from scipy import io

try:
    from csnlp import Nlp, wrappers
except ImportError:
    import sys

    sys.path.insert(1, "src")
    from csnlp import Nlp, wrappers


def plot_nlp(ax: Axes, a: float, b: float, x: float, y: float):
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
    C = np.concatenate
    ax.fill(
        C([b * np.cos(ts - np.pi) - 0.5, 10 * np.cos(np.pi - ts) - 0.5]),
        C([b * np.sin(ts - np.pi), 10 * np.sin(np.pi - ts)]),
        facecolor="r",
        alpha=0.3,
        edgecolor="k",
    )
    ax.fill([2, 2, 0, a - 2], [a - 2, 2, 2, 2], facecolor="r", alpha=0.3, edgecolor="k")
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


def z(x, lam, p):
    # in the CasADi blog, z is only a function of x, i.e., z(x)
    return (x[1, :] ** p[0] - x[0, :]) * cs.exp(-10 * lam[0] - 2 * lam[1]) / p[1]


# build the NLP
sym_type = "MX"
nlp = Nlp(sym_type=sym_type)
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


# Use IPOPT to solve the nonlinear optimization
M = nlp.to_function("M", [p, x], [x], ["p", "x0"], ["x"])

p_values = ((1.4, 1.9), (2.2, 1.9), (2.2, 1.1))
fig, axs = plt.subplots(
    1, 3, sharex=True, sharey=True, constrained_layout=True, figsize=(5, 5)
)
for p_, ax in zip(p_values, axs.flat):
    x_ = M(p_, 0).full()
    plot_nlp(ax, *p_, x_[0], x_[1])


# How does the optimal solution vary along p?
Z = nlp.to_function("Z", [p, x], [z(x, lam, p)], ["p", "x0"], ["z"])
fig, ax = plt.subplots(
    constrained_layout=True, subplot_kw={"projection": "3d", "computed_zorder": False}
)
N = 150
P = np.meshgrid(np.linspace(1, 2.5, N), np.linspace(1, 2, N))
try:
    Z_values = io.loadmat("Z_values.mat")["z"]
except Exception:
    Z_values = Z(np.row_stack((P[0].flat, P[1].flat)), 0).full().reshape(N, N)
    io.savemat("Z_values.mat", {"p0": P[0], "p1": P[1], "z": Z_values})
ax.plot_wireframe(
    P[0], P[1], Z_values, color="k", antialiased=False, rstride=5, cstride=5, zorder=0
)


# Parametric sensitivities of function z(x(p), lam(p))
nlp = wrappers.NlpSensitivity(nlp)
Z = z(x, lam, p)
J, H = (o.squeeze() for o in nlp.parametric_sensitivity(expr=Z))

P_flat = np.row_stack([o.flatten() for o in P])
for p_ in p_values:
    sol = nlp.solve(pars={"p": p_})
    Z_ = sol.value(Z).full()
    J_ = sol.value(J)
    H_ = sol.value(H)
    c = ax.scatter(p_[0], p_[1], Z_, s=100, zorder=2)
    deltaP = P_flat - np.array(p_)[:, None]
    S = Z_ + J_.T @ deltaP + 0.5 * np.diag(deltaP.T @ H_ @ deltaP)
    ax.plot_surface(
        P[0], P[1], S.reshape(N, N), color=c.get_facecolor(), alpha=0.3, lw=0, zorder=1
    )

ax.set_xlabel("$p_0$")
ax.set_ylabel("$p_1$")
ax.set_zlabel("z")
ax.set_xlim(1, 2.5)
ax.set_ylim(1, 2)
ax.set_zlim(-0.25, 0.03)
plt.show()
