r"""
A simple optimization problem: hanging chain
============================================

Again, this example illustrates another simple use of :class:`csnlp.Nlp` in solving an
optimization problem.

The problem reproduces the example from
`this CasADi blog post <https://web.casadi.org/blog/opti/>`_, where a chain of :math:`N`
point masses (of mass :math:`m`) connected by springs is hanging. We are interested in
finding the rest position of each of the masses, i.e.,
:math:`(x_i, y_i), \ i = 1, \ldots, N`. To do so, we minimize the potential energy of
the system, which is given by the gravitational potential energy and the elastic energy
of the springs

.. math::
    V = \frac{1}{2} D \sum_{i=1}^{N-1}{
            \left( (x_{i+1} - x_i)^2 + (y_{i+1} - y_i)^2 \right)
    } + g m \sum_{i=1}^N y_i.
"""

# %%
# Creating and solving the problem
# --------------------------------
# Again, the only novel import is :class:`csnlp.Nlp`. The rest should be familiar.

import casadi as cs
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from csnlp import Nlp

# %%
# As last setup step, we define the constant parameters of the problem and create a
# :class:`matplotlib.figure.Figure` with three subplots (which will be populated with
# three variants of the problem).

N = 40
m = 40 / N
D = 70 * N / 2
g = 9.81
L = 1

# %%
# In order to build the optimization problem, we first create an instance of
# :class:`csnlp.Nlp` and define the position variables :math:`p=(x,y)`. Then, we can
# formulate an adequate objective function, i.e., the total energy of the system, and
# pass it to the :meth:`csnlp.Nlp.minimize` method.

nlp = Nlp[cs.SX]()
p = nlp.variable("p", (2, N))[0]
x, y = p[0, :], p[1, :]
V = D * (cs.sumsqr(cs.diff(x)) + cs.sumsqr(cs.diff(y))) + g * m * cs.sum2(y)
nlp.minimize(V)

# %%
# To have the problem well-posed, we need to add some constraints. We fix the first and
# last points of the chain to be at :math:`(-2, 1)` and :math:`(2, 1)`, respectively.
nlp.constraint("c1", p[:, 0], "==", [-2, 1])
_ = nlp.constraint("c2", p[:, -1], "==", [2, 1])

# %%
# We can now solve the problem and plot the chain. We first initialize the solver with
# the default options (uses IPOPT) and then call the :meth:`csnlp.Nlp.solve` method.
# The solution allows us to compute the numerical value of the optimal positions via
# :meth:`csnlp.Solution.value`.

nlp.init_solver()
sol1 = nlp.solve()

# %%
# Adding ground constraints
# -------------------------
# We can further improve the "simulation" of the chain in case it lays on the
# hypothetical ground. We can add a new constraint modelling the ground, without the
# need to recreate the :class:`csnlp.Nlp` instance or reinitialize the solver (this is
# done manually under the hood). We can also warm-start the new solver's run with the
# previous solution.

nlp.constraint("c3", y, ">=", cs.cos(0.1 * x) - 0.5)
sol2 = nlp.solve(vals0={"p": sol1.vals["p"]})

# %%
# Non-zero rest length
# -------------------------
# The problem can be further extended by considering that the springs have a non-zero
# rest length. This can be done by modifying the spring energy contributions. We can
# then pass the new objective to the :meth:`csnlp.Nlp.minimize` method and solve the
# problem again.

# Problem 3: Rest Length
V = D * cs.sum2(
    (cs.sqrt(cs.diff(x) ** 2 + cs.diff(y) ** 2) - L / N) ** 2
) + g * m * cs.sum2(y)
nlp.minimize(V)
sol3 = nlp.solve(vals0={"p": np.row_stack((np.linspace(-2, 2, N), np.ones(y.shape)))})


# %%
# Plotting
# --------
# Finally, we can plot the three variants of the chain problem side by side.


def plot_chain(ax: Axes, x: cs.DM, y: cs.DM) -> None:
    ax.plot(x.full().flat, y.full().flat, "o-")


def plot_ground(ax: Axes) -> None:
    xs = np.linspace(-2, 2, 100)
    ax.plot(xs, np.cos(0.1 * xs) - 0.5, "r--")


_, axs = plt.subplots(1, 3, sharey=True, constrained_layout=True, figsize=(7, 3))
plot_chain(axs[0], sol1.value(x), sol1.value(y))
plot_ground(axs[1])
plot_chain(axs[1], sol2.value(x), sol2.value(y))
plot_ground(axs[2])
plot_chain(axs[2], sol3.value(x), sol3.value(y))
axs[0].set_ylabel("y [m]")
for ax in axs:
    ax.set_xlabel("x [m]")
plt.show()
