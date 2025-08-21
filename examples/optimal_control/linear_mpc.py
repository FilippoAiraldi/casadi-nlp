r"""
Linear MPC control
==================

This example demos how an MPC controller can be built for a linear system using
:class:`csnlp.wrappers.Mpc` and its :meth:`csnlp.wrappers.Mpc.set_affine_dynamics`
method. The system is a simple linear system with the following dynamics:

.. math:: x_{k+1} = A x_k + B u_k.

The goal is to minimize the cost function

.. math:: \sum_{k=0}^{N-1} x_k^T x_k + 10^{-4} (u_k - u_{k-1})^T (u_k - u_{k-1}),

while the system is subject to the following constraints:

.. math::
    -[4, 10, 4, 10]^\top \leq x_k \leq [4, 10, 4, 10]^\top, \ \ -0.5 \leq u_k \leq 0.5.

This example is taken from
`here <https://www.do-mpc.com/en/latest/example_gallery/oscillating_masses_discrete.html>`_.
"""

# %%
# We start with the usual imports, and set the random seed for reproducibility.

import casadi as cs
import matplotlib.pyplot as plt
import numpy as np

from csnlp import Nlp
from csnlp.wrappers import Mpc

np.random.seed(99)

# %%
# System's dynamics
# -----------------
# First of all, we define the system matrices for the linear system.

A = np.asarray(
    [
        [0.763, 0.460, 0.115, 0.020],
        [-0.899, 0.763, 0.420, 0.115],
        [0.115, 0.020, 0.763, 0.460],
        [0.420, 0.115, -0.899, 0.763],
    ]
)
B = np.asarray([[0.014], [0.063], [0.221], [0.367]])
ns, na = B.shape

# %%
# MPC setup
# ---------
# Then, we create the MPC controller with a prediction horizon of 7 steps and with
# single shooting method (for linear MPC cases, single shooting is often faster to
# solve). Since we are in single shooting mode, we do not have access to the state
# symbols straight away, but only after the dynamics have been set. After that, we can
# retrieve the state symbol from the `states` dictionary.

# %%
# Build the NLP and MPC instances and define the state and action.

N = 7
mpc = Mpc(Nlp[cs.SX](), prediction_horizon=N, shooting="single")
mpc.state("x", ns)
u, _ = mpc.action("u", na, lb=-0.5, ub=0.5)

# %%
# Set the linear dynamics.

_ = mpc.set_affine_dynamics(A, B)

# %%
# Define the constraints on the state only after having set the dynamics.

x = mpc.states["x"]
x_bound = np.asarray([[4.0], [10.0], [4.0], [10.0]])
mpc.constraint("x_lb", x, ">=", -x_bound)
_ = mpc.constraint("x_ub", x, "<=", x_bound)

# %%
# Define the cost function.

delta_u = cs.diff(u, 1, 1)
mpc.minimize(cs.sumsqr(x) + 1e-4 * cs.sumsqr(delta_u))

# %%
# Initialize the solver with a QP solver.

opts = {
    "error_on_fail": True,
    "expand": True,
    "print_time": False,
    "record_time": True,
    "verbose": False,
    "printLevel": "none",
}
mpc.init_solver(opts, "qpoases", type="conic")

# %%
# Simulation
# ----------
# Finally, we simulate the system for 50 steps and plot the results. We draw a random
# initial state and solve the MPC problem at each time step in a receding horizon
# fashion.

x = np.random.uniform(-3, 3, size=ns)
u_prev = 0
X, U, C = [x], [], []
times = []

for _ in range(50):
    sol = mpc.solve_ocp(x)
    times.append(sol.stats["t_wall_solver"])
    u_opt = sol.vals["u"][:, 0].full().reshape(na)
    x = A @ x + B @ u_opt
    X.append(x)
    U.append(u_opt)
    C.append(float(cs.sumsqr(x) + 1e-4 * cs.sumsqr(u_opt - u_prev)))
    u_prev = u_opt

X = np.squeeze(X)
U = np.squeeze(U)
C = np.squeeze(C)
print(f"avg solver time: {np.mean(times)}+/-{np.std(times)}")


fig, axs = plt.subplots(3, 1, constrained_layout=True, sharex=True)
timesteps = np.arange(X.shape[0])
axs[0].plot(timesteps, X)
axs[1].step(timesteps[:-1], U, where="post")
axs[2].plot(timesteps[:-1], C)
for ax, n in zip(axs, ("x", "u", "cost")):
    ax.set_ylabel(n)
axs[-1].set_xlabel("time step")
plt.show()
