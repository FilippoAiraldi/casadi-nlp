"""Reproduces the results of [1] shown in Tables 1 and 2, for R = R_1 = R_2 = 0.

References
----------
[1] Schildbach, G., Fagiano, L., Frei, C. and Morari, M., 2014. The scenario approach
    for stochastic model predictive control with bounds on closed-loop constraint
    violations. Automatica, 50(12), pp.3009-3018.
"""

import casadi as cs
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from csnlp import Nlp, Solution
from csnlp.wrappers import ScenarioBasedMpc

plt.style.use("bmh")

np_random = np.random.default_rng(69)
opts = {"print_time": False, "ipopt": {"sb": "yes", "print_level": 0}}


def get_dynamics() -> cs.Function:
    x = cs.SX.sym("x", 2)
    u = cs.SX.sym("u", 2)
    d = cs.SX.sym("d", 3)  # 2 disturbances + 1 varying parameter (modelled as dist.)
    theta = d[2]
    A = cs.blockcat([[0.7, -0.1 * (2 + theta)], [-0.1 * (3 + 2 * theta), 0.9]])
    B = cs.DM.eye(2)
    x_next = A @ x + B @ u + d[:2]
    return cs.Function("F", [x, u, d], [x_next], ["x", "u", "d"], ["x+"])


def sample_disturbances(K: int, N: int) -> npt.NDArray[np.floating]:
    d = np_random.normal(scale=0.1, size=(K, 2, N))
    theta = np_random.uniform(size=(K, 1, N))
    return np.concatenate((d, theta), 1)


# parameters
T = 200  # simulation timesteps
N = 5  # mpc horizon
K = 19  # number of scenarios
x0 = np.array([1, 1])  # initial state


# build the scenario-based MPC
F = get_dynamics()
scmpc = ScenarioBasedMpc[cs.SX](Nlp(), K, N)
x, _, _ = scmpc.state("x", F.size1_in(0), remove_bounds_on_initial=True)
u, _ = scmpc.action("u", F.size1_in(1), lb=-5, ub=+5)
d, _ = scmpc.disturbance("d", F.size1_in(2))
scmpc.constraint_from_single("x_lb", x[:, 1:], ">=", 1)  # equivalent to lb, ub on x
scmpc.set_dynamics(F)
scmpc.minimize_from_single(
    sum(cs.sumsqr(x[:, i]) + cs.sumsqr(u[:, i]) for i in range(N))
)
scmpc.init_solver(opts)


# simulate
x, X, U = x0, [], []
vals0 = None
for t in range(T):
    if t > 0 and t % 20 == 0:
        print(f"t = {t}")

    # sample disturbances, and assign each to a scenario
    sample = sample_disturbances(scmpc.n_scenarios, scmpc.prediction_horizon)
    pars = {scmpc.name_i("d", i): sample[i] for i in range(scmpc.n_scenarios)}
    pars["x_0"] = x  # set initial state

    # run the scenario-based MPC
    sol: Solution = scmpc.solve(pars, vals0)
    assert sol.success, f"Solver failed at time {t}!"
    u_opt = sol.vals["u"][:, 0]

    # step the system dynamics
    X.append(x)
    U.append(u_opt)
    actual_disturbance_realization = sample_disturbances(1, 1).flatten()
    x = F(x, u_opt, actual_disturbance_realization).full().flatten()


# plot results
X, U = np.asarray(X), np.squeeze(U)
Q, R = np.eye(F.size1_in(0)), np.eye(F.size1_in(1))
L: np.ndarray = (X.dot(Q) * X).sum(1) + (U.dot(R) * U).sum(1)
violated_con1, violated_con2 = X.T < 1
violated_either = np.logical_or(violated_con1, violated_con2)

print(f"Avg. violation 1: {violated_con1.sum() / T * 100.0:.2f}%")
print(f"Avg. violation 1: {violated_con2.sum() / T * 100.0:.2f}%")
print(f"Avg. violation 1 or 2: {violated_either.sum() / T * 100.0:.2f}%")
print(f"Avg cost: {L.mean():.2f} +/- {L.std():.2f}")

plt.axhline(1, color="darkgrey")
plt.axvline(1, color="darkgrey")
plt.plot(*X[~violated_con1 & ~violated_con2].T, "o", color="C0")
plt.plot(*X[violated_con1 & ~violated_con2].T, "o", color="C1")
plt.plot(*X[~violated_con1 & violated_con2].T, "o", color="C2")
plt.plot(*X[violated_con1 & violated_con2].T, "o", color="C3")
plt.axis("square")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.show()
