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

.. math::
    x_+ = A_i x + B_i u + c_i \quad \text{if} \quad S_i [x^\top, u^\top]^\top \leq T_i.

In the context of optimal control, with the proper procedure, these dynamics can be
translated into a mixed-integer optimization problem. To do so, polytopic bounds on the
state and input variables must be defined as

.. math:: D [x^\top, u^\top]^\top \leq E.

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
# ------------------
# PWA MPC Controller
# ------------------
# The MPC controller is built in the same way as for other MPC classes. The main
# difference is that now we must use the :meth:`csnlp.wrappers.PwaMpc.set_pwa_dynamics`
# instead of :meth:`csnlp.wrappers.Mpc.set_affine_dynamics` to set the dynamics of the
# system.

N = 10
mpc = wrappers.PwaMpc(nlp=Nlp[cs.SX](sym_type="SX"), prediction_horizon=N)
x, _ = mpc.state("x", 2)
u, _ = mpc.action("u")
mpc.set_pwa_dynamics(pwa_system, D, E)
mpc.constraint("state_constraints", D1 @ x - E1, "<=", 0)
mpc.constraint("input_constraints", D2 @ u - E2, "<=", 0)
mpc.minimize(cs.sumsqr(x) + cs.sumsqr(u))
mpc.init_solver({"record_time": True}, "bonmin")  # "bonmin", "knitro", "gurobi"

# %%
# We then solve the MPC problem and plot the results. The optimization problem will both
# optimize over the state and action trajectories, as well as the sequence of regions
# that the system will follow. For this reason, it is a mixed-integer optimization
# problem.

x_0 = [-3, 0]
sol_mixint = mpc.solve(pars={"x_0": x_0})

# %%
# ----------------------------
# Affine time-varying dynamics
# ----------------------------
# As stated above, when using :meth:`csnlp.wrappers.PwaMpc.set_pwa_dynamics` to specify
# the PWA dynamics, the numerical solver will optimize also over the sequence of regions
# that the system will follow, thus it must find the solution to a logical/integer
# problem. This is often computationally expensive. But an alternative exists: to
# specify a fixed switching sequence of regions manually/externally, and let the solver
# only optimize the state-action trajectory. This is of course in general
# computationally much cheaper.
# :class:`csnlp.wrappers.PwaMpc` also allows for defining the affine dynamics while
# manually providing the sequence of regions the system should follow, rather than
# letting the solver optimize it. The dynamics are thus time-varying  affine. It is then
# the user's responsibility to specify reasonable switching sequences.

# %%
# Building again the MPC, but this time, affine
# ---------------------------------------------
# Now lets explore the setting in which the switching sequence is passed rather than
# optimized. We build the MPC as before, but now using the
# :meth:`csnlp.wrappers.PwaMpc.set_affine_time_varying_dynamics` method to set the
# dynamics of the system instead. Note that, since the sequence is fixed, we do not need
# a mixed-integer solver, but we can use any QP solver.

mpc = wrappers.PwaMpc(nlp=Nlp[cs.SX](sym_type="SX"), prediction_horizon=N)
x, _ = mpc.state("x", 2)
u, _ = mpc.action("u")
mpc.set_affine_time_varying_dynamics(pwa_system)
mpc.constraint("state_constraints", D1 @ x - E1, "<=", 0)
mpc.constraint("input_constraints", D2 @ u - E2, "<=", 0)
mpc.minimize(cs.sumsqr(x) + cs.sumsqr(u))
mpc.init_solver({"record_time": True}, "qrqp")

# %%
# We then set the switching sequence to be the optimal one (gathered from
# the previous solution) via :meth:`csnlp.wrappers.PwaMpc.set_switching_sequence`, and
# solve the ensuing QP problem for the same initial state.

opt_sequence = wrappers.PwaMpc.get_optimal_switching_sequence(sol_mixint)
mpc.set_switching_sequence(opt_sequence)
sol_qp = mpc.solve(pars={"x_0": x_0})

# %%
# Effects of suboptimal sequences
# -------------------------------
# As aforementioned, the sequence now is specified by the user externally. This means
# that also suboptimal switching sequences can be passed. The solver will still find a
# solution, as long as the sequence is feasible, but the cost will be higher than when
# the optimal sequnce is passed or the sequence is part of the optimization.

subopt_sequence = opt_sequence.copy()
subopt_sequence[3] = 0
mpc.set_switching_sequence(subopt_sequence)
sol_qp_suboptimal = mpc.solve(pars={"x_0": x_0})

# %%
# -------
# Results
# -------
# Let's take a look at the optimality of the three solutions. Of course, we expect the
# mixed-integer solution to be the optimal one, the QP solution with the optimal
# sequence to be the same, and the QP solution with the suboptimal sequence to be worse.

print(f"Optimal mixed-integer cost: {sol_mixint.f}")
print(f"Optimal QP cost: {sol_qp.f}")
print(f"Suboptimal QP cost: {sol_qp_suboptimal.f}")

# %%
# However, we have gained some computational efficiency by not optimizing over the
# sequence of regions. This can be seen in the time taken to solve the problems.

print(f"Optimal mixed-integer time: {sol_mixint.stats['t_wall_total']}")
print(f"Optimal QP time: {sol_qp.stats['t_wall_total']}")
print(f"Suboptimal QP time: {sol_qp_suboptimal.stats['t_wall_total']}")

# %%
# We can also finally plot the three results (optimal mixed-integer, optimal QP, and
# suboptimal QP problem solutions).

_, axs = plt.subplots(1, 2, constrained_layout=True, sharey=True, figsize=(12, 5))

t = np.linspace(0, N, N + 1)
axs[0].step(t, sol_mixint.vals["x"].T, where="post")
axs[0].step(t[:-1], sol_mixint.vals["u"].T, where="post", color="C4")
axs[1].step(t, sol_qp.vals["x"].T, where="post")
axs[1].step(t, sol_qp_suboptimal.vals["x"].T, where="post", ls="--")
axs[1].step(t[:-1], sol_qp.vals["u"].T, where="post")
axs[1].step(t[:-1], sol_qp_suboptimal.vals["u"].T, where="post", ls="--")

axs[0].set_xlabel("Time step")
axs[0].set_title("Optimal mixed-integer solution")
axs[0].legend([r"$x^\text{MIQP}_1$", r"$x^\text{MIQP}_2$", r"$u^\text{MIQP}$"])
axs[0].set_xlabel("Time step")
axs[1].set_title("Optimal and suboptimal QP solutions")
axs[1].legend(
    [
        r"$x^\text{QP}_1$",
        r"$x^\text{QP}_2$",
        r"$u^\text{QP}$",
        r"$x^\text{subQP}_1$",
        r"$x^\text{subQP}_2$",
        r"$u^\text{subQP}$",
    ]
)

plt.show()
