# Inspired by https://web.casadi.org/blog/nlp-scaling/


import casadi as cs
import matplotlib.pyplot as plt
import numpy as np

from csnlp import Nlp, wrappers
from csnlp.solutions import Solution
from csnlp.util.scaling import Scaler

plt.style.use("bmh")


def get_dynamics(g: float, alpha: float, dt: float) -> cs.Function:
    x, u = cs.SX.sym("x", 3), cs.SX.sym("u")
    x_next = x + cs.vertcat(x[1], u / x[2] - g, -alpha * u) * dt
    return cs.Function("F", [x, u], [x_next], ["x", "u"], ["x+"])


# parameters
N = 100  # number of control intervals
T = 100  # Time horizon [s]
dt = T / N
m0 = 500000  # start mass [kg]
yT = 100000  # final height [m]
g = 9.81  # gravity 9.81 [m/s^2]
alpha = 1 / (300 * g)  # kg/(N*s)

# solver options
opts = {"print_time": False, "ipopt": {"sb": "yes", "print_level": 5}}

# plotting
time = np.linspace(0, T, N + 1)
_, axs = plt.subplots(5, 2, constrained_layout=True)
axs[0, 0].set_ylabel("Thrust [N]")
axs[1, 0].set_ylabel("Height [m]")
axs[2, 0].set_ylabel("Speed [m/s]")
axs[3, 0].set_ylabel("Mass [kg]")
axs[3, 0].set_xlabel("Time [s]")
axs[3, 1].set_xlabel("Time [s]")
axs[4, 1].set_ylabel("Primal/dual feasibility")
axs[4, 1].set_xlabel("Iteration number")


for i, SCALED in enumerate((False, True)):
    # create mpc
    nlp = Nlp(sym_type="SX")
    if SCALED:
        # NOTE: since the scaling affects constraint definition, the NLP must be first
        # wrapped in it, and only then in the MPC.
        y_nom = 1e5
        v_nom = 2e3
        m_nom = 3e5
        x_nom = cs.vertcat(y_nom, v_nom, m_nom)
        u_nom = 1e8
        scaler = Scaler()
        scaler.register("x", scale=x_nom)
        scaler.register("x_0", scale=x_nom)
        scaler.register("u", scale=u_nom)
        nlp = wrappers.NlpScaling(nlp, scaler=scaler)
    mpc = wrappers.Mpc(nlp, prediction_horizon=N)

    # state and actions
    x, _ = mpc.state("x", 3, lb=cs.DM([-cs.inf, -cs.inf, 0]))
    y = x[0, :]  # height
    v = x[1, :]  # velocity
    m = x[2, :]  # mass
    u, _ = mpc.action("u", lb=0, ub=5e7)

    # dynamics
    F = get_dynamics(g, alpha, dt)
    mpc.dynamics = F

    # boundary conditions
    x0 = cs.vertcat(0, 0, m0)
    mpc.constraint("yT", y[-1], "==", yT)

    # objective
    mpc.minimize(m[0] - m[-1])
    mpc.init_solver(opts)
    x_initial = cs.repmat([0, 0, 1e5], 1, N + 1)
    sol: Solution = mpc.solve(pars={"x_0": x0}, vals0={"x": x_initial, "u": 0})

    # plotting
    u: np.ndarray = sol.value(u).full()
    x: np.ndarray = sol.value(x).full()
    axs[0, i].step(time[:-1], u.flat, where="post")
    axs[1, i].plot(time, x[0, :].flat)
    axs[2, i].plot(time, x[1, :].flat)
    axs[3, i].plot(time, x[2, :].flat)
    if SCALED:
        axs2 = [axs[j, i].twinx() for j in range(4)]
        u_scaled: np.ndarray = sol.value(mpc.unscaled_variables["u"]).full()
        x_scaled: np.ndarray = sol.value(mpc.unscaled_variables["x"]).full()
        axs2[0].step(time[:-1], u_scaled.flat, "r--", where="post")
        axs2[1].plot(time, x_scaled[0, :].flat, "r--")
        axs2[2].plot(time, x_scaled[1, :].flat, "r--")
        axs2[3].plot(time, x_scaled[2, :].flat, "r--")
        for ax in axs2:
            ax.spines["right"].set_color("r")
            ax.tick_params(axis="y", colors="r")
    axs[4, i].semilogy(
        sol.stats["iterations"]["inf_pr"], "-", sol.stats["iterations"]["inf_du"], "-"
    )

plt.show()
