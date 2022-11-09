# Reproduces https://web.casadi.org/blog/nlp_sens/code_1d.m and
# https://web.casadi.org/blog/nlp_sens/plot_nlp.m


import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import casadi as cs
import numpy as np
from casadi_nlp import Nlp, wrappers


def plot_nlp(ax: Axes, a: float, b: float, x: float, y: float):
    X, Y = np.meshgrid(*[np.linspace(-1.5, 2, 100)] * 2)
    ax.contour(X, Y, (1 - X)**2 + a * (Y - X**2)**2, 100)

    ts = np.linspace(0, 2 * np.pi, 100)
    ax.fill(b / 2 * np.cos(ts) - 0.5, b / 2 * np.sin(ts),
            facecolor='r', alpha=0.3, edgecolor='k')
    cat = np.concatenate
    ax.fill(
        cat([b * np.cos(ts - np.pi) - 0.5, 10 * np.cos(np.pi - ts) - 0.5]),
        cat([b * np.sin(ts - np.pi), 10 * np.sin(np.pi - ts)]),
        facecolor='r', alpha=0.3, edgecolor='k'
    )
    ax.fill([-1.5, 0, 0, -1.5], [-1.5, -1.5, 2, 2],
            facecolor='r', alpha=0.3, edgecolor='k')
    ax.plot(x, y, '*', markersize=12, linewidth=2)
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim(-1.5, 2)
    ax.set_ylim(-1.5, 2)


def z(xy):
    return xy[1, :] - xy[0, :]


# build the NLP
nlp = Nlp(sym_type='SX')

x = nlp.variable('x', lb=0)[0]
y = nlp.variable('y')[0]
xy = cs.vertcat(x, y)
a = 0.2
p = nlp.parameter('p')

nlp.minimize((1 - x)**2 + a * (y - x**2)**2)
g = (x + 0.5)**2 + y**2
nlp.constraint('c1', (p / 2)**2, '<=', g)
nlp.constraint('c2', g, '<=', p**2)
opts = {'print_time': False, 'ipopt': {'sb': 'yes', 'print_level': 0}}
nlp.init_solver(opts)


# Use IPOPT to solve the nonlinear optimization
M = nlp.to_function('M', [p, xy], [xy], ['p', 'xy'], ['xy'])

p_values = (1.25, 1.4, 2)
fig, axs = plt.subplots(
    1, 3, sharey=True, constrained_layout=True, figsize=(8, 3))
for p0, ax in zip(p_values, axs):
    xy_ = M(p0, 0)
    plot_nlp(ax, a, p0, float(xy_[0]), float(xy_[1]))


# How does the optimal solution vary along p?
pv = np.linspace(1, 2, 100).reshape(1, -1)
S = M(pv, 0).full()

fig, ax = plt.subplots(constrained_layout=True)
ax.plot(pv.flat, z(S).flat, 'k.', markersize=8)


# Parametric sensitivities
nlp = wrappers.DifferentiableNlp(nlp)
dxydp = nlp.parametric_sensitivity()[:nlp.nx]
# dxydp is the sensitivity of primal vars w.r.t. parameters

t = np.linspace(1, 2, 1000)
for p0, clr in zip(p_values, ['r', 'g', 'b']):
    sol = nlp.solve(pars={'p': p0})
    F = sol.value(z(xy))
    J = sol.value(cs.jacobian(z(xy), xy) @ dxydp)
    H = 0

    ax.plot(p0, float(F), 'x', color=clr, markersize=16)
    ax.plot(t, F + J * (t - p0) + 0.5 * H * (t - p0)**2, lw=2, color=clr)

plt.show()
