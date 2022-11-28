# Reproduces https://web.casadi.org/blog/opti/


import casadi as cs
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from csnlp import Nlp


def plot_chain(ax: Axes, x: cs.DM, y: cs.DM) -> None:
    ax.plot(x.full().flat, y.full().flat, "o-")


def plot_ground(ax: Axes) -> None:
    xs = np.linspace(-2, 2, 100)
    ax.plot(xs, np.cos(0.1 * xs) - 0.5, "r--")


N = 40
m = 40 / N
D = 70 * N / 2
g = 9.81
L = 1
nlp = Nlp()
fig, axs = plt.subplots(1, 3, sharey=True, constrained_layout=True, figsize=(7, 2))


# Problem 1: Simple hanging chain
p = nlp.variable("p", (2, N))[0]
x, y = p[0, :], p[1, :]
V = D * (cs.sumsqr(cs.diff(x)) + cs.sumsqr(cs.diff(y))) + g * m * cs.sum2(y)
nlp.minimize(V)
nlp.constraint("c1", p[:, 0], "==", [-2, 1])
nlp.constraint("c2", p[:, -1], "==", [2, 1])
nlp.init_solver()
sol = nlp.solve()
plot_chain(axs[0], sol.value(x), sol.value(y))


# Problem 2: adding ground constraints
nlp.constraint("c3", y, ">=", cs.cos(0.1 * x) - 0.5)
sol = nlp.solve(vals0={"p": sol.vals["p"]})  # warm-starts the new solver run
plot_chain(axs[1], sol.value(x), sol.value(y))
plot_ground(axs[1])


# Problem 3: Rest Length
V = D * cs.sum2(
    (cs.sqrt(cs.diff(x) ** 2 + cs.diff(y) ** 2) - L / N) ** 2
) + g * m * cs.sum2(y)
nlp.minimize(V)
sol = nlp.solve(vals0={"p": np.row_stack((np.linspace(-2, 2, N), np.ones(y.shape)))})
plot_chain(axs[2], sol.value(x), sol.value(y))
plot_ground(axs[2])

plt.show()
