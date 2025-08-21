r"""
Formulating an MPC controller for HPIPM
=======================================

`HPIPM <https://github.com/giaf/hpipm>`_ is a high-performance QP solver specifically
designed for Optimal Control Problems (OCPs). Briefly, it is able to exploit the
classical OCP structure to achieve faster convergence.

To use it with :class:`csnlp.wrappers.Mpc`, we need to give special care on how the
problem is defined. In particular, we cannot declare, e.g., all the states and then
all the actions, but we must declare states and actions in an interleaved fashion, along
the prediction horizon. This means that we define state at time step :math:`k=0`, then
action at `k=0`, then state at `k=1`, etc.

To achieve this, we can take advantage of
:meth:`csnlp.wrappers.Mpc.interleaved_states_and_actions`, which under the hood creates
the specified states and actions in an interleaved manner. This example demos how to
use this method. The example is taken from
`here <https://github.com/casadi/casadi/blob/b23ad1fa768ae1d137379047e61241c94d299e0c/test/python/conic.py#L962>`_,
and it consists of the following MPC problem

.. math::
    \begin{aligned}
        \min_{\substack{u_0,\dots,u_{N-1} \\ x_0,\dots,x_N}} \quad
            & \sum_{i=0}^{N-1} [x_i^\top, u_i^\top] Q [x_i^\top, u_i^\top]^\top
              + [x_i^\top, u_i^\top] g + x_N^\top x_N  \\
        \text{s.t.} \quad
            & x_0 = [2, 1]^\top \\
            & x_{i+1} = A x_i + B u_i, \quad i = 0,\ldots,N-1, \\
            & x_{2,1} = 0^\top,
    \end{aligned}

where the cost and dynamics matrices can be found later in the code. Note the presence
of an equality constraint on the first state at time step :math:`k=2`.
"""

# %%
# Creating the dynamcis and stage cost
# ------------------------------------
# First of all, we need to create a function that computes the linear dynamics and
# stage cost. After the usual imports, define the states (2) and action (1).

import casadi as cs

from csnlp import Nlp, wrappers

x1 = cs.MX.sym("x1")
x2 = cs.MX.sym("x2")
x = cs.vertcat(x1, x2)
u = cs.MX.sym("u")

# %%
# Formulate the dynamics and the stage cost, andwrap this two in a single
# :class:`casadi.Function`.

xdot = cs.vertcat(0.6 * x1 - 1.11 * x2 + 0.3 * u - 0.03, 0.7 * x1 + 0.01)
x_next = x + xdot
stage_cost = (
    x1**2 + 3 * x2**2 + 7 * u**2 - 0.4 * x1 * x2 - 0.3 * x1 * u + u - x1 - 2 * x2
)
F = cs.Function("F", [x, u], [x_next, stage_cost], ["x", "u"], ["x_next", "cost"])

# %%
# Definition of MPC problem
# -------------------------
# To define the MPC, first create an instance of NLP and MPC classes, as usual.

N = 4
mpc = wrappers.Mpc(nlp=Nlp(), prediction_horizon=N)

# %%
# Now, for the new part: do not create the symbolic states and actions via
# :meth:`csnlp.wrappers.Mpc.state` and :meth:`csnlp.wrappers.Mpc.action`, but use
# :meth:`csnlp.wrappers.Mpc.interleaved_states_and_actions`. This method can create
# multiple states and actions (as a generator), so it takes as inputs dictionaries of
# keywords for each variable. For the same reason, it also yields tuples of
# state-action-next state at each time step. The code looks as follows:

objective = 0

generator = mpc.interleaved_states_and_actions(
    {"name": "x", "size": 2, "lb": -100, "ub": 100},
    {"name": "u", "lb": -100, "ub": 100},
)
(x,), (u,) = next(generator)
for k, ((x_new,), (u_new,)) in enumerate(generator):
    x_new_predicted, stage_cost = F(x, u)
    mpc.constraint(f"dyn{k}", x_new, "==", x_new_predicted)
    if k == 2:
        mpc.constraint("other-constraint", x[0], "==", 0.0)  # custom constraint
    objective += stage_cost
    x, u = x_new, u_new

objective += cs.sumsqr(x_new)  # terminal cost

# %%
# Note how we have added a non-infinite lower and upper bounds to each state and action
# since HPIPM prefers it this way. To adhere to the OCP structure required by HPIPM, we
# also have created the dynamics constraints (and any other constraints, for this
# matter) in a stage-wise manner. At the same time, we have also accumulated the
# stage cost into the objective.

# %%
# Lastly, set the objective and initialize the solver. Then, call
# :meth:`csnlp.wrappers.Mpc.solve_ocp` to solve the problem for the given initial
# conditions.

mpc.minimize(objective)
mpc.init_solver({"print_time": False, "print_problem": False}, "hpipm", "conic")
sol = mpc.solve_ocp([2, 1])

print(sol.f)  # should be close to 219.22
