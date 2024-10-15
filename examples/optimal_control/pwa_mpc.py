r"""
PWA MPC controller
==================

This example demos how an MPC controller can be built for a system with
piecewise affine dynamics using :class:`csnlp.wrappers.HybridMpc`.
"""

# %%
# Imports
import casadi as cs
import matplotlib.pyplot as plt
import numpy as np

from csnlp import Nlp, wrappers

# %%
# Discrete time PWA dynamics for the one-sided spring-mass-damper system
tau = 0.5  # sampling time for discretization
k1 = 10  # spring constant when one-sided spring active
k2 = 1  # spring constant when one-sided spring not active
damp = 4  # damping constant
mass = 10  # mass of the system
A_spring_1 = np.array([[1, tau], [-((tau * 2 * k1) / mass), 1 - (tau * damp) / mass]])
A_spring_2 = np.array([[1, tau], [-((tau * 2 * k2) / mass), 1 - (tau * damp) / mass]])
B_spring = np.array([[0], [tau / mass]])
x_bnd = [5, 5]
u_bnd = 20
system_dict = {
    # Regions - switch about origin
    "S": [np.array([[1, 0]]), np.array([[-1, 0]])],
    "R": [np.zeros((1, 1)), np.zeros((1, 1))],
    "T": [np.array([[0]]), np.array([[0]])],
    # Dynamics
    "A": [A_spring_1, A_spring_2],
    "B": [B_spring, B_spring],
    "c": [np.array([[0], [0]]), np.array([[0], [0]])],
    # bounds
    "D": np.array([[1, 0], [-1, 0], [0, 1], [0, -1]]),
    "E": np.array([[x_bnd[0]], [x_bnd[0]], [x_bnd[1]], [x_bnd[1]]]),
    "F": np.array([[1], [-1]]),
    "G": np.array([[u_bnd], [u_bnd]]),
}

# %%
# Construct the MPC controller
N = 10
mpc = wrappers.PwaMpc(
    nlp=Nlp[cs.SX](sym_type="SX"), prediction_horizon=N, shooting="multi"
)
x, _ = mpc.state("x", 2)
u, _ = mpc.action("u")
mpc.set_pwa_dynamics(system_dict)
mpc.minimize(cs.sumsqr(x) + cs.sumsqr(u))
mpc.init_solver(solver="gurobi")  # other options: "bonmin", "knitro"

# %%
# Solve mpc problem
sol = mpc.solve(pars={"x_0": [-3, 0]})
t = np.linspace(0, N, N + 1)
plt.plot(t, sol.value(x).T)
plt.step(t[:-1], sol.vals["u"].T.full(), "-.", where="post")
plt.legend(["x1", "x2", "u"])
plt.xlim(t[0], t[-1])
plt.xlabel("t [s]")
plt.show()
