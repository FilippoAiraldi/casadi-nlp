# Inspired by https://web.casadi.org/blog/nlp_sens/code_1d.m and
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


def z(x, lam):
    # in the CasADi blog, z is only a function of x, i.e., z(x)
    return (x[1, :] - x[0, :]) * cs.exp(-10 * lam)


# build the NLP
nlp = Nlp(sym_type='SX')

x = nlp.variable('x', (2, 1), lb=[[0], [-np.inf]])[0]
p = nlp.parameter('p', (2, 1))

nlp.minimize((1 - x[0])**2 + p[0] * (x[1] - x[0]**2)**2)
g = (x[0] + 0.5)**2 + x[1]**2
nlp.constraint('c1', (p[1] / 2)**2, '<=', g)
_, lam = nlp.constraint('c2', g, '<=', p[1]**2)
opts = {'print_time': False, 'ipopt': {'sb': 'yes', 'print_level': 0}}
nlp.init_solver(opts)


# Use IPOPT to solve the nonlinear optimization
M = nlp.to_function('M', [p, x], [x], ['p', 'x0'], ['x'])

p_values = (1.2, 1.5, 1.9)
fig, axs = plt.subplots(
    1, 3, sharey=True, constrained_layout=True, figsize=(8, 3))
for p0, ax in zip(p_values, axs):
    x_ = M([0.2, p0], 0).full()
    plot_nlp(ax, 0.2, p0, x_[0], x_[1])


# How does the optimal solution vary along p?
Z = nlp.to_function('Z', [p, x], [z(x, lam)], ['p', 'x0'], ['z'])

fig, ax = plt.subplots(constrained_layout=True)
pv = np.row_stack((np.full(100, 0.2), np.linspace(1, 2, 100)))
ax.plot(pv[1].flat, Z(pv, 0).full().flat, 'k.', markersize=8)


# Parametric sensitivities
nlp = wrappers.DifferentiableNlp(nlp)
y = nlp.primal_dual_vars
# dydp is the sensitivity of primal-dual vars w w.r.t. parameters p
dydp = nlp.parametric_sensitivity()
dzdy = cs.jacobian(z(x, lam), y).T

t = np.linspace(1, 2, 1000)
for p0, clr in zip(p_values, ['r', 'g', 'b']):
    sol = nlp.solve(pars={'p': [0.2, p0]})
    F = sol.value(z(x, lam))
    J = sol.value(dzdy.T @ dydp)[1]  # take jacobian of only the 2nd parameter
    H = 0  # 3.5169, 1.4018, 0

    ax.plot(p0, float(F), 'x', color=clr, markersize=16)
    ax.plot(t, F + J * (t - p0) + 0.5 * H * (t - p0)**2, lw=2, color=clr)

ax.set_xlim(1, 2)
ax.set_ylim(-0.17, 0.03)
plt.show()
