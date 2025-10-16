r"""
How scaling can help convergence
================================

This example showcases a optimal control task where the scaling of the NLP variables can
really impact the performance of the solver. This is due to the fact that the variables
have widely different scales, making it challenging for the solver to find a good
numerical solutions.

We'll skip over some of the formalities of the library, as we assume you are already a
bit familiar with it. If this is not the case, for more introductory examples on
:mod:`csnlp`, see the other :ref:`introductory_examples`.

The problem reproduces the example from
`this CasADi blog post <https://web.casadi.org/blog/nlp-scaling/>`_, where an optimal
control problem for a 1-dimensional rocket task is tackled. The goal is to attain a
given altitude :math:`Y_f` at time :math:`T` with minimal fuel consumption. In
mathematical terms, the rocket state at time :math:`t` is given by
:math:`x_t = [y_t, v_t, m_t]^\top`, i.e., the altitude, speed, and mass of the rocket.
The continuous-time dynamics of the rocket are given by

.. math::
    f(x_t, u_t) = \begin{bmatrix}
        v_t \\ \frac{u_t}{m_t} - g \\ -\alpha u_t
    \end{bmatrix},

where :math:`u` is the thrust, and other symbols are constants. By Euler-discretizing
the dynamics, we formulate the final optimal control problem

.. math::
    \begin{aligned}
        \min_{\substack{u_0,\dots,u_{N-1} \\ x_0,\dots,x_N}} \quad & m_0 - m_N \\
        \text{s.t.} \quad & y_N = Y_f \\
            & x_0 = [0, 0, m_0]^\top \\
            & x_{t+1} = x_t + f(x_t, u_t) \Delta t, \quad t = 0, \ldots, N-1.
    \end{aligned}
"""

# %%
# Without scaling
# ---------------
# First, we'll address the problem without scaling. As usual, we'll import the necessary
# classes and modules from :mod:`csnlp`.

import casadi as cs
import matplotlib.pyplot as plt
import numpy as np

from csnlp import Nlp, scaling, wrappers

# %%
# Let us define constants of the problem, the dynamics of the rocket, as well as the
# options for the IPOPT solver.

N = 100  # number of control intervals
T = 100  # Time horizon [s]
K = 3  # Nlp multistarts
dt = T / N
m0 = 500000  # start mass [kg]
yT = 100000  # final height [m]
g = 9.81  # gravity 9.81 [m/s^2]
alpha = 1 / (300 * g)  # kg/(N*s)
seed = 69
opts = {"print_time": False, "ipopt": {"sb": "yes", "print_level": 5}}


def get_dynamics() -> cs.Function:
    x, u = cs.SX.sym("x", 3), cs.SX.sym("u")
    x_next = x + cs.vertcat(x[1], u / x[2] - g, -alpha * u) * dt
    return cs.Function("F", [x, u], [x_next], ["x", "u"], ["x+"])


F = get_dynamics()


# %%
# Now, we can build the optimization problem and solve it.

nlp = Nlp[cs.SX]()
x, _, _ = nlp.variable("x", (3, N))
u, _, _ = nlp.variable("u", (1, N - 1), lb=0)
nlp.constraint("dynamics", x[:, 1:], "==", F(x[:, :-1], u))
nlp.constraint("initial_conditions", x[:, 0], "==", [0, 0, m0])
nlp.constraint("final_conditions", x[0, -1], "==", yT)
nlp.minimize(x[2, 0] - x[2, -1])
nlp.init_solver(opts)
x_guess = cs.repmat([0, 0, 1e5], 1, N)
sol = nlp(vals0={"x": x_guess})


# %%
# We can visualize the optimal thrust strategy and the predicted evolution of the
# states

_, axs = plt.subplots(4, 1, sharex=True, constrained_layout=True)

time = np.linspace(0, T, N)
u_opt = sol.value(u).full().flatten()
x_opt = sol.value(x).full()
axs[0].plot(time[:-1], u_opt)
for x_, ax in zip(x_opt, axs[1:]):
    ax.plot(time, x_)

axs[0].set_ylabel("Thrust [N]")
axs[1].set_ylabel("Height [m]")
axs[2].set_ylabel("Speed [m/s]")
axs[3].set_ylabel("Mass [kg]")
axs[3].set_xlabel("Time [s]")
plt.show()

# %%
# Though it is much more interesting to see how the solver behaved during its
# iterations. In particular, we can plot the convergence of primal and dual feasibility,
# which is pretty bad. This is due to the fact that the altitude and mass of the rocket
# have larger scales than the speed, which makes the solver struggle to find a good
# solution.

iter_stats = sol.stats["iterations"]
_, ax = plt.subplots(constrained_layout=True)
ax.semilogy(iter_stats["inf_pr"], label="Primal")
ax.semilogy(iter_stats["inf_du"], label="Dual")
ax.legend()
ax.set_xlabel("Iteration")
ax.set_ylabel("Feasibility")
plt.show()

# %%
# With (automatic) scaling
# ------------------------
# How can we then improve convergence? One way is to scale the variables of the NLP.
# :mod:`csnlp` provides the scaling wrapper :class:`csnlp.wrappers.NlpScaling` that can
# do just this.
#
# First, let us define the scales of each state and build an instance of
# :class:`csnlp.core.scaling.Scaler`, which contains all the information needed for
# scaling. We can do so by calling the :meth:`csnlp.core.scaling.Scaler.register`.
x_nominal = cs.DM([1e5, 2e3, 3e5])
u_nominal = 1e8
scaler = scaling.Scaler()
scaler.register("x", scale=x_nominal)
scaler.register("u", scale=u_nominal)

# %%
# Then, we wrap the NLP isntance in the scaling wrapper and solve the problem again.
# Since this wrapper is non-retroactive, we need to build the NLP again from scratch.
# The solver should now converge much faster.

nlp = Nlp[cs.SX]()
nlp = wrappers.NlpScaling[cs.SX](nlp, scaler=scaler)  # added this line!
x, _, _ = nlp.variable("x", (3, N))
u, _, _ = nlp.variable("u", (1, N - 1), lb=0)
nlp.constraint("dynamics", x[:, 1:], "==", F(x[:, :-1], u))
nlp.constraint("initial_conditions", x[:, 0], "==", [0, 0, m0])
nlp.constraint("final_conditions", x[0, -1], "==", yT)
nlp.minimize(x[2, 0] - x[2, -1])
nlp.init_solver(opts)
sol_scaled = nlp(vals0={"x": x_guess})

# %%
# We can again plot the optimal values, both for the scaled (solid, blue) and unscaled
# variables (dashed, orange). As you can see, there is no difference (aside of course
# from the scale) between the two solutions.

_, axs = plt.subplots(4, 1, sharex=True, constrained_layout=True)

u_opt_scaled = sol_scaled.value(u).full().flatten()
x_opt_scaled = sol_scaled.value(x).full()
axs[0].plot(time[:-1], u_opt_scaled)
for x_, ax in zip(x_opt_scaled, axs[1:]):
    ax.plot(time, x_)

axs_twin = [ax.twinx() for ax in axs]
u_opt_unscaled = sol_scaled.value(nlp.unscale(u)).full().flatten()
x_opt_unscaled = sol_scaled.value(nlp.unscale(x)).full()
axs_twin[0].plot(time[:-1], u_opt_unscaled, "--", color="C1")
for x_, ax in zip(x_opt_unscaled, axs_twin[1:]):
    ax.plot(time, x_, "--", color="C1")

for ax in axs:
    ax.spines["left"].set_color("C0")
    ax.tick_params(axis="y", colors="C0")
for ax in axs_twin:
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_color("C1")
    ax.tick_params(axis="y", colors="C1")
axs[0].set_ylabel("Thrust [N]")
axs[1].set_ylabel("Height [m]")
axs[2].set_ylabel("Speed [m/s]")
axs[3].set_ylabel("Mass [kg]")
axs[3].set_xlabel("Time [s]")
plt.show()

# %%
# Finally, we can plot the convergence of the solver. For comparison, we plot again the
# convergence of the unscaled problem (solid, faded), and on top of those also the ones
# of the scaled problem (dashed). As you can see, the scaled problem converges much
# faster (i.e., with less iterations and falling steeper).

iter_stats_scaled = sol_scaled.stats["iterations"]
_, ax = plt.subplots(constrained_layout=True)
ax.semilogy(iter_stats["inf_pr"], alpha=0.2, label="Primal", color="C0")
ax.semilogy(iter_stats["inf_du"], alpha=0.2, label="Dual", color="C1")
ax.semilogy(iter_stats_scaled["inf_pr"], "--", lw=2, color="C0")
ax.semilogy(iter_stats_scaled["inf_du"], "--", lw=2, color="C1")
ax.legend()
ax.set_xlabel("Iteration")
ax.set_ylabel("Feasibility")
plt.show()
