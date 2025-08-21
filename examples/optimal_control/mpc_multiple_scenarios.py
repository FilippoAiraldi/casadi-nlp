r"""
Multiple Scenario MPC
=====================

This example shows how to use the :class:`csnlp.wrappers.MultiScenarioMpc` to create a
multi-scenario MPC controller. We will only solve it once to demonstrate its open-loop
solution, but of course the controller can and should be used in a receding horizon
fashion, as done in the other examples. It is highly recommended to first go through the
example :ref:`scenario_based_mpc_example` to familiarize yourself with
:class:`csnlp.wrappers.ScenarioBasedMpc`.

Consider a prediction model

.. math::
    \dot{x} = f(x, u) = \begin{bmatrix}
        (1 - x_2^2) x_1 - x_2 + u \\ p x_1
    \end{bmatrix},

where :math:`p \sim \mathcal{U}_{[1,3]}` is an uncertain parameter distributed as a
uniform distribution, :math:`x` is the state, and :math:`u` is the control action. We
discretize the continuous-time dynamics using a Runge-Kutta 4th order integrator,
obtaining the discrete-time dynamics :math:`f_d(x_k,u_k)`. The stochastic optimal
control problem is then

.. math::
    \begin{aligned}
        \min_{\substack{u_0,\dots,u_{N-1} \\ x_0,\dots,x_N}} \quad
            & \mathbb{E} \left[ \sum_{i=0}^{N} x_i^2 + \sum_{i=0}^{N-1} u_i^2 \right] \\
        \text{s.t.} \quad
            & x_0 = [0, 1]^\top \\
            & x_{i+1} = f_d(x_i, u_i), \quad i = 0,\ldots,N-1.
    \end{aligned}

In general, solving such stochastic problmes is not easy. Here, we propose to use a
multi-scenario MPC (MSMPC) controller. In practice, given :math:`K` independent samples
of the uncertain parameter :math:`p`, we replace the expectation with the average
over :math:`K` different scenarios, each with its own value of :math:`p`, state
trajectory and optimal action sequence. Note that the first action of each scenario is
shared across all scenarios, so that once computed we can apply it to the real
system.
"""

# %%
# We start with the usual imports. Let us also fix the RNG.

from itertools import cycle

import casadi as cs
import matplotlib.pyplot as plt
import numpy as np

from csnlp import Nlp
from csnlp.wrappers import MultiScenarioMpc

np_random = np.random.default_rng(0)


# %%
# Setup
# -----
# First of all, let us define some constants.

N = 20  # MSMPC horizon
T = 15  # time horizon (for RK4 discretization and simulation)
K = 3  # MSMPC number of scenarios
nx = 2  # number of states
nu = 1  # number of actions
x0 = np.array([0, 1])  # initial state
shooting = "multi"  # "multi" or "single"
a_bound = 1.0  # upper and lower bound on the action
x_lb = -0.2  # lower bound on the states
opts = {"print_time": False, "ipopt": {"sb": "yes", "print_level": 5}}  # IPOPT options

# %%
# Then, let's define the continuous-time dynamics and discretize them via RK4. The final
# dynamics are conveniently packed in the function :math:`F`.

x = cs.SX.sym("x", nx, 1)
u = cs.SX.sym("u", nu, 1)
p = cs.SX.sym("p")
ode = cs.vertcat((1 - x[1] ** 2) * x[0] - x[1] + u, p * x[0])
dae = {"x": x, "p": cs.vertcat(u, p), "ode": ode}
intg = cs.integrator(
    "intg", "rk", dae, 0, T / N, {"simplify": True, "number_of_finite_elements": 4}
)
x_next = intg(x0=x, p=cs.vertcat(u, p))["xf"]
F = cs.Function("F", [x, u, p], [x_next], ["x", "u", "p"], ["x_next"])
del x, u, p, ode, dae, intg, x_next  # cleanup


# %%
# Building the MSMPC
# ------------------
# The interface of the :class:`csnlp.wrappers.MultiScenarioMpc` remains mostly similar
# the :class:`csnlp.wrappers.ScenarioBasedMpc`, but it also allows to define also
# parameters and actions that are shared across all scenarios. The first step is to
# create a fresh :class:`csnlp.Nlp` instance  and pass it to the
# :class:`csnlp.wrappers.MultiScenarioMpc` constructor. Alongside the number of
# scenarios and control horizon, another important argument to
# :class:`csnlp.wrappers.MultiScenarioMpc.__init__` is ``input_sharing``, which defines
# how many actions are shared across all scenarios. In this case, we leave it to the
# default value of ``1``, meaning that the first action is shared across all scenarios.

nlp = Nlp[cs.SX](sym_type="SX")
msmpc = MultiScenarioMpc[cs.SX](nlp, K, N, shooting=shooting)

# %%
# Then, we can define action :math:`u` and parameter :math:`p`. Under the hood,
# similarly to how :class:`csnlp.wrappers.ScenarioBasedMpc` handles states, one copy of
# action/parameter is created for each scenario (and can be found in the return).

u = msmpc.action("u", nu, lb=-a_bound, ub=a_bound)[0]
p, _ = msmpc.parameter("p")

# %%
# Also create the state :math:`x` which requires different handling depending on the
# shooting method. Suffice to say that in single shooting the state is first declared,
# then created when the dynamics are set, and only after that any constraint on the
# state can be created. In multi shooting, the state is created first, and
# constraints can be added right away such as the dynamics.

if shooting == "multi":
    x, _ = msmpc.state("x", nx, lb=x_lb)
    msmpc.set_nonlinear_dynamics(lambda x_, u_: F(x_, u_, p))
else:
    msmpc.state("x", nx)
    msmpc.set_nonlinear_dynamics(lambda x_, u_: F(x_, u_, p))
    x = msmpc.single_states["x"]  # only accessible after dynamics have been set
    msmpc.constraint_from_single("x_lb", x[:, 1:], ">=", x_lb)

# %%
# Finally, we set the objective of the problem as the empirical expectation (i.e.,
# average) of the original cost. The averaging is done automatically under the hood.
# Don't forget to also initialize the solver.

msmpc.minimize_from_single(cs.sumsqr(x) + cs.sumsqr(u))
msmpc.init_solver(opts)

# %%
# Solving the MSMPC and plotting
# ------------------------------
# After we have generated :math:`K` samples of the uncertain parameter :math:`p`, we can
# solve the MSMPC by passing the initial states and samples as a dictionary. If unsure
# about the names that must be contained in this dictionary, you can check the keys of
# :attr:`csnlp.wrappers.MultiScenarioMpc.parameters` (this contains all values that must
# be specified to run the NLP solver).

p_samples = np_random.uniform(1, 3, size=K)
pars = {f"p__{i}": p_samples[i] for i in range(K)}
sol = msmpc.solve_ocp(x0, pars)

# %%
# For plotting, we will plot the results in a two axes, one for the states and one for
# the action. Different scenarios are assigned different linestyles.

fig, axs = plt.subplots(2, 1, constrained_layout=True, sharex=True)

t = np.linspace(0, T, N + 1)
for i, ls in zip(range(K), cycle(["-", "--", "-."])):
    x = sol.value(msmpc.states_i(i)["x"]).toarray()
    u = sol.value(msmpc.actions_i(i)["u"]).toarray().flatten()
    axs[0].plot(t, x[0], ls=ls, color="C0", label=f"$x_1$ (scenario {i + 1})")
    axs[0].plot(t, x[1], ls=ls, color="C1", label=f"$x_2$ (scenario {i + 1})")
    axs[1].step(
        t[:-1], u, ls=ls, color="C2", where="post", label=f"$u$ (scenario {i + 1})"
    )

axs[0].set_ylabel("States")
axs[1].set_ylabel("Action")
for ax in axs:
    ax.set_xlabel("t [s]")
    ax.legend()
plt.show()
