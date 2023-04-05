# Inspired by https://web.casadi.org/blog/nlp-scaling/


import casadi as cs
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from csnlp import Solution, multistart, scaling, wrappers
from csnlp.util.random import np_random

plt.style.use("bmh")


def get_dynamics(g: float, alpha: float, dt: float) -> cs.Function:
    x, u = cs.SX.sym("x", 3), cs.SX.sym("u")
    x_next = x + cs.vertcat(x[1], u / x[2] - g, -alpha * u) * dt
    return cs.Function("F", [x, u], [x_next], ["x", "u"], ["x+"])


# parameters
N = 100  # number of control intervals
T = 100  # Time horizon [s]
K = 3  # Nlp multistarts
dt = T / N
m0 = 500000  # start mass [kg]
yT = 100000  # final height [m]
g = 9.81  # gravity 9.81 [m/s^2]
alpha = 1 / (300 * g)  # kg/(N*s)
seed = 69

# solver options
opts = {
    "print_time": False,
    "ipopt": {
        # "linear_solver": "ma97",
        # "linear_system_scaling": "mc19",
        # "nlp_scaling_method": "equilibration-based",
        "sb": "yes",
        "print_level": 5,
    },
}

# plotting
time = np.linspace(0, T, N + 1)
_, axs = plt.subplots(5, 2, constrained_layout=True)
axs[0, 0].set_ylabel("Thrust [N]")
axs[1, 0].set_ylabel("Height [m]")
axs[2, 0].set_ylabel("Speed [m/s]")
axs[3, 0].set_ylabel("Mass [kg]")
axs[4, 0].set_ylabel("Primal/dual feasibility")
axs[3, 0].set_xlabel("Time [s]")
axs[3, 1].set_xlabel("Time [s]")
axs[4, 0].set_xlabel("Iteration number")
axs[4, 1].set_xlabel("Iteration number")


for i in range(2):
    is_scaled = bool(i)

    # create rng
    rng = np_random(seed)

    # create mpc
    nlp = multistart.StackedMultistartNlp[cs.SX](sym_type="SX", starts=K)
    if is_scaled:
        # NOTE: since the scaling affects constraint definition, the NLP must be first
        # wrapped in the scaling wrapper, and only then in the MPC wrapper.
        x_nom = cs.DM([1e5, 2e3, 3e5])
        u_nom = 1e8
        scaler = scaling.Scaler()
        scaler.register("x", scale=x_nom)
        scaler.register("x_0", scale=x_nom)
        scaler.register("u", scale=u_nom)
        nlp = wrappers.NlpScaling[cs.SX](nlp, scaler=scaler)  # type: ignore[assignment]
    mpc = wrappers.Mpc[cs.SX](nlp, prediction_horizon=N)

    # state and actions
    x, _ = mpc.state("x", 3, lb=cs.DM([-cs.inf, -cs.inf, 0]))
    y = x[0, :]  # type: ignore[index]
    v = x[1, :]  # type: ignore[index]
    m = x[2, :]  # type: ignore[index]
    u, _ = mpc.action("u", lb=0, ub=5e7)

    # dynamics
    F = get_dynamics(g, alpha, dt)
    mpc.set_dynamics(F)

    # boundary conditions
    x0 = cs.vertcat(0, 0, m0)
    mpc.constraint("yT", y[-1], "==", yT)

    # objective
    mpc.minimize(m[0] - m[-1])
    mpc.init_solver(opts)
    x_initial = cs.repmat([0, 0, 1e5], 1, N + 1)
    sol: Solution[cs.SX] = mpc.solve_multi(
        pars=({"x_0": x0} for _ in range(K)),
        vals0=(
            {
                "x": x_initial + rng.random(x_initial.shape) * 1e4,
                "u": rng.random() * 1e8,
            }
            for _ in range(K)
        ),
    )

    # plotting
    u_: npt.NDArray[np.floating] = sol.value(u).full()
    x_: npt.NDArray[np.floating] = sol.value(x).full()
    axs[0, i].step(time[:-1], u_.flat, where="post")
    axs[1, i].plot(time, x_[0, :].flat)
    axs[2, i].plot(time, x_[1, :].flat)
    axs[3, i].plot(time, x_[2, :].flat)
    if is_scaled:
        axs2 = [axs[j, i].twinx() for j in range(4)]
        u_us: npt.NDArray[np.floating] = sol.value(mpc.unscale(mpc.actions["u"])).full()
        x_us: npt.NDArray[np.floating] = sol.value(mpc.unscale(mpc.states["x"])).full()
        axs2[0].step(time[:-1], u_us.flat, "r--", where="post")
        axs2[1].plot(time, x_us[0, :].flat, "r--")
        axs2[2].plot(time, x_us[1, :].flat, "r--")
        axs2[3].plot(time, x_us[2, :].flat, "r--")
        for ax in axs2:
            ax.spines["right"].set_color("r")
            ax.tick_params(axis="y", colors="r")
    axs[4, i].semilogy(
        sol.stats["iterations"]["inf_pr"], "-", sol.stats["iterations"]["inf_du"], "-"
    )

plt.show()
