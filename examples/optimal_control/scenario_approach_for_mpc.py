r"""
Scenario-based MPC
==================

In this example, we demonstrate how to build a scenario-based model predictive control
(SCMPC) using :class:`csnlp.wrappers.ScenarioBasedMpc`. The problem is inspired by the
numerical results of :cite:`schildbach_scenario_2014` shown in Tables 1 and 2, for
:math:`R = R_1 = R_2 = 0`.
"""

# %%
# We start with the usual imports. Let us also fix the RNG.

import casadi as cs
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from csnlp import Nlp
from csnlp.wrappers import ScenarioBasedMpc

np_random = np.random.default_rng(69)


# %%
# Setup
# -----
# We then define the dynamics of the problem. The only peculiarity is that we have
# embedded a varying parameter in the dynamics, which is modelled as a disturbance.


def get_dynamics() -> cs.Function:
    x = cs.SX.sym("x", 2)
    u = cs.SX.sym("u", 2)
    d = cs.SX.sym("d", 3)  # 2 disturbances + 1 varying parameter (modelled as dist.)
    theta = d[2]
    A = cs.blockcat([[0.7, -0.1 * (2 + theta)], [-0.1 * (3 + 2 * theta), 0.9]])
    B = cs.DM.eye(2)
    x_next = A @ x + B @ u + d[:2]
    return cs.Function("F", [x, u, d], [x_next], ["x", "u", "d"], ["x+"])


# %%
# In the Scenario Approach, we also need a function that allows us to sample independent
# samples of the disturbances affecting the stochastic optimization problem.


def sample_disturbances(K: int, N: int) -> npt.NDArray[np.floating]:
    d = np_random.normal(scale=0.1, size=(K, 2, N))
    theta = np_random.uniform(size=(K, 1, N))
    return np.concatenate((d, theta), 1)


# %%
# Let us also define some constants.

# parameters
N = 5  # mpc horizon
K = 19  # number of scenarios
x0 = np.array([1, 1])  # initial state

# %%
# Building the SCMPC
# ------------------
# The interface of the :class:`csnlp.wrappers.ScenarioBasedMpc` remains mostly similar
# to the one of :class:`csnlp.wrappers.Mpc`, so if this procedure is not clear, see
# the other, simpler example in :ref:`optimal_control_examples`.

F = get_dynamics()
nx, nu, nd = F.size1_in(0), F.size1_in(1), F.size1_in(2)

scmpc = ScenarioBasedMpc[cs.SX](Nlp(), K, N)
x, _, _ = scmpc.state("x", nx, bound_initial=False)
u, _ = scmpc.action("u", nu, lb=-5, ub=+5)
d, _ = scmpc.disturbance("d", nd)
scmpc.set_nonlinear_dynamics(F)

# %%
# Since the SCMPC internally defines ``K`` copies of the state and disturbances, how
# can we define a constraint that should be applied to each of the scenarios? We can
# use the method :meth:`csnlp.wrappers.ScenarioBasedMpc.constraint_from_single`, which
# will automatically decline the constraint to each scenario. In this case, we impose
# a lower bound on the state (we could have achieved the same result by passing `lb=1`
# to the state definition).

_ = scmpc.constraint_from_single("x_lb", x[:, 1:], ">=", 1)  # equivalent to a lb on x

# %%
# The same reasoning is valid for the objective. It suffices to create the objective for
# one scenario, and then call
# :meth:`csnlp.wrappers.ScenarioBasedMpc.minimize_from_single`, which will automatically
# compute the average objective across all scenarios.
scmpc.minimize_from_single(
    sum(cs.sumsqr(x[:, i]) + cs.sumsqr(u[:, i]) for i in range(N))
)

# %%
# Simulation
# ----------
# We can finally simulate! First, let's initialize the solver.

opts = {
    "error_on_fail": True,
    "expand": True,
    "print_time": False,
    "record_time": True,
    "verbose": False,
    "printLevel": "none",
}
scmpc.init_solver(opts, "qpoases")

# %%
# Then, we'd like to simulate the solution of the MPC along a trajectory of length
# ``T``. We'll save the resulting states and actions in dedicated lists.
#
# As commonly done, we adopt a receding horizon strategy, where only first action of the
# solution to the MPC is applied, and the rest is discarded. The process is then
# repeated at the next time step.

T = 200  # simulation repetitions
x = x0
X, U = [], []
vals0 = None
for t in range(T):
    # sample disturbances, and assign each to a scenario
    sample = sample_disturbances(scmpc.n_scenarios, scmpc.prediction_horizon)
    pars = {scmpc.name_i("d", i): sample[i] for i in range(scmpc.n_scenarios)}
    pars["x_0"] = x  # set initial state

    # run the scenario-based MPC
    sol = scmpc.solve(pars, vals0)
    assert sol.success, f"Solver failed at time {t}!"
    u_opt = sol.vals["u"][:, 0]  # apply only the first action

    # step the real system dynamics
    actual_disturbance_realization = sample_disturbances(1, 1).flatten()
    x = F(x, u_opt, actual_disturbance_realization).full().flatten()
    U.append(u_opt)
    X.append(x)


# %%
# Results
# -------
# First of all, we can  compute the mean and standard deviation of the cost along the
# trajectory. The closer to zero, the better the performance of the controller. However,
# the controller should not be too aggressive, as it could lead to violating the
# constraints.
X, U = np.asarray(X), np.squeeze(U)
Q, R = np.eye(nx), np.eye(nu)
L = (X.dot(Q) * X).sum(1) + (U.dot(R) * U).sum(1)  # same as the objective function
print(f"Average cost: {L.mean():.2f} +/- {L.std():.2f}")

# %%
# We can then compute how many times either constraint (or both) has been violated. The
# higher the number of scenarios, the lower the chances of violations. In the plot, we
# show with marker ``x`` all the states that have violated a bound (different colors for
# different combinations of violations)

violated_con1, violated_con2 = X.T < 1
violated_either = np.logical_or(violated_con1, violated_con2)
print(f"Average violation 1: {violated_con1.sum() / T * 100.0:.2f}%")
print(f"Average violation 2: {violated_con2.sum() / T * 100.0:.2f}%")
print(f"Average violation 1 or 2: {violated_either.sum() / T * 100.0:.2f}%")

plt.axhline(1, color="darkgrey")
plt.axvline(1, color="darkgrey")
plt.plot(*X[~violated_con1 & ~violated_con2].T, "o", color="C0")
plt.plot(*X[violated_con1 & ~violated_con2].T, "x", color="C1")
plt.plot(*X[~violated_con1 & violated_con2].T, "x", color="C2")
plt.plot(*X[violated_con1 & violated_con2].T, "x", color="C3")
plt.axis("square")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.show()
