r"""
MPC controller for PWA systems
==============================

This example demos how an MPC controller can be built for a simple system with piecewise
affine dynamics using :class:`csnlp.wrappers.PwaMpc`. We assume some knowledge of MPC
is already present; otherwise, refer to other optimal control examples. For more details
on the subject of control for PWA and hybrid systems , see :cite:`bemporad_control_1999`
and :cite:`borrelli_predictive_2017`.

Briefly, PWA systems are systems whose dynamics are described by a set of affine systems
that are active in different regions of the state space. For region :math:`i`, the
dynamics are described as

.. math:: x_+ = A_i x + B_i u + c_i & \text{if } S_i [x^\top, u^\top]^\top \leq T_i.

In the context of optimal control, with the proper procedure, these dynamics can be
translated into a mixed-integer optimization problem. To do so, polytopic bounds on the
state and input variables must be defined as

.. math:: D x \leq E, \quad F u \leq G.

The procedure to convert the PWA system into a mixed-integer optimization problem
requires solving linear programs, whose number increases with the number of regions in
the system. So, the fewer regions, the computationally lighter building the MPC is.
"""

# %%
# We start with the imports as usual. All MPC controller classes can be found in the
# :mod:`csnlp.wrappers` module.

import casadi as cs
import matplotlib.pyplot as plt
import numpy as np

from csnlp import Nlp, wrappers

# %%
# -----
# Setup
# -----

# %%
# Dynamics
# --------
# We consider a simple one-sided spring-mass-damper system with piecewise affine
# dynamics. In particular, the system has two modes: one where the spring is active and
# another where it is not, thus yielding two regions with different affine dynamics.

tau = 0.5  # sampling time for discretization
k1 = 10  # spring constant when one-sided spring active
k2 = 1  # spring constant when one-sided spring not active
damp = 4  # damping constant
mass = 10  # mass of the system
A_spring_1 = np.array([[1, tau], [-((tau * 2 * k1) / mass), 1 - (tau * damp) / mass]])
A_spring_2 = np.array([[1, tau], [-((tau * 2 * k2) / mass), 1 - (tau * damp) / mass]])
B_spring = np.array([[0], [tau / mass]])

# %%
# From these matrices, we can define a sequence of :class:`csnlp.wrappers.PwaRegion`
# objects that represent the dynamics of the system in each region.

pwa_system = (
    wrappers.PwaRegion(
        # region dynamics
        A=A_spring_1,
        B=B_spring,
        c=np.zeros(2),
        # region domain
        S=np.array([[1, 0, 0]]),
        T=np.zeros(1),
    ),
    wrappers.PwaRegion(
        # region dynamics
        A=A_spring_2,
        B=B_spring,
        c=np.zeros(2),
        # region domain
        S=np.array([[-1, 0, 0]]),
        T=np.zeros(1),
    ),
)

# %%
# Bounds
# ------
# In order for the PWA system to be converted to a mixed-integer optimization problem,
# we need to define the bounds of the system. In this case, we must impose polytopic
# bounds `D @ [x; u] <= E` on the states and the inputs as follows.

x_bnd = (5, 5)
u_bnd = 20
D1 = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
D2 = np.array([[1], [-1]])
D = cs.diagcat(D1, D2).sparse()
E1 = np.array([x_bnd[0], x_bnd[0], x_bnd[1], x_bnd[1]])
E2 = np.array([u_bnd, u_bnd])
E = np.concatenate((E1, E2))

# %%
# --------------
# MPC Controller
# --------------
# The MPC controller is built in the same way as for other MPC classes. The main
# difference is that now we must use the :meth:`csnlp.wrappers.PwaMpc.set_pwa_dynamics`
# instead of :meth:`csnlp.wrappers.Mpc.set_dynamics` to set the dynamics of the system.
N = 10
mpc = wrappers.PwaMpc(nlp=Nlp[cs.SX](sym_type="SX"), prediction_horizon=N)
x, _ = mpc.state("x", 2)
u, _ = mpc.action("u")
mpc.set_pwa_dynamics(pwa_system, D, E)
mpc.constraint("state_constraints", D1 @ x - E1, "<=", 0)
mpc.constraint("input_constraints", D2 @ u - E2, "<=", 0)
mpc.minimize(cs.sumsqr(x) + cs.sumsqr(u))
mpc.init_solver(solver="gurobi")  # "bonmin", "knitro", "gurobi"

# %%
# We then solve the MPC problem and plot the results.

sol_mint = mpc.solve(pars={"x_0": [-3, 0]})
t = np.linspace(0, N, N + 1)
plt.plot(t, sol_mint.value(x).T)
plt.step(t[:-1], sol_mint.vals["u"].T.full(), "-.", where="post")
plt.legend(["x1", "x2", "u"])
plt.xlim(t[0], t[-1])
plt.xlabel("t [s]")
plt.show()

# %%
# Now lets explore the setting of the sequence for the pwa mpc # TODO make example nicer
sequence = np.argmax(
    sol_mint.vals["delta"], axis=0
)  # extract the optimal sequence from the mixed-integer solution
mpc = wrappers.PwaMpc(
    nlp=Nlp[cs.SX](sym_type="SX"), prediction_horizon=N
)  # now we can craete a new mpc that uses the sequence
x, _ = mpc.state("x", 2)
u, _ = mpc.action("u")
mpc.set_time_varying_affine_dynamics(pwa_system)
mpc.constraint("state_constraints", D1 @ x - E1, "<=", 0)
mpc.constraint("input_constraints", D2 @ u - E2, "<=", 0)
mpc.minimize(cs.sumsqr(x) + cs.sumsqr(u))
mpc.init_solver(solver="qrqp")  # here we do not have to use a mixed-integer solver
mpc.set_sequence(
    sequence
)  # set the sequence to be the same as the optimal one from the previous solution
sol_qp = mpc.solve(pars={"x_0": [-3, 0]})
# We can see here that we get the same solution as before

# %% Now lets try passing a suboptimal but feasible fixed-sequence
sequence[3] = 0
mpc.set_sequence(sequence)
sol_qp_suboptimal = mpc.solve(pars={"x_0": [-3, 0]})
# we can see here that we get a feasible solution still but a higher cost, as the sequence was suboptimal
pass
