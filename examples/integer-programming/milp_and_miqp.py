r"""
A mixed integer linear programming example
==========================================

In this example, we show how to formualte and solve a simple mixed integer linear
programming (MILP) problem via :mod:`csnlp`. This simple problem, taken from
`here <https://en.wikipedia.org/wiki/Integer_programming#Example>`_, is

.. math::
   \begin{aligned}
   \max_{x, y \in \mathbb{Z}} \quad & y \\
   \textrm{s.t.} \quad & -x + y \leq 1 \\
                       & 3 x + 2 y \leq 12 \\
                       & 2 x + 3 y \leq 12 \\
                       & x, y \geq 0.
   \end{aligned}

The focus here is on how to formulate mixed integer problems, so for more basic usage of
the library, please refer to the other :ref:`introductory_examples`.
"""

# %%
# Creating the problem
# --------------------
# We can create the problem as usual, but we need to specify that (some of) the primal
# variables are discrete. This is done by passing the corresponding ``domain``` argument
# when creating each variable. The rest of the problem formulation is standard.

import casadi as cs

from csnlp import Nlp

nlp = Nlp[cs.MX](sym_type="MX")
x = nlp.variable("x", lb=0, discrete=True)[0]
y = nlp.variable("y", lb=0, discrete=True)[0]

nlp.minimize(-y)
_, _ = nlp.constraint("con1", -x + y, "<=", 1)
_, _ = nlp.constraint("con2", 3 * x + 2 * y, "<=", 12)
_, _ = nlp.constraint("con3", 2 * x + 3 * y, "<=", 12)

# %%
# Solving the problem
# -------------------
# However, pay attention: now we have to initialize a suitable solver. For example, we
# can use the ``cbc`` solver, which is an open-source mixed integer linear programming
# solver. We get a solution as usual by calling :meth:`csnlp.Nlp.solve`.

nlp.init_solver(solver="cbc")
sol = nlp.solve()

# %%
# We expect the optimal point to be either :math:`(x^\star, y^\star) = (1, 2)` or
# :math:`(x^\star, y^\star) = (2, 2)`.

print(sol.vals)


# %%
# A mixed integer quadratic programming (MIQP) example
# ----------------------------------------------------
# We are not limited to MILP. Unfortunatelly, that is only what ``cbc`` can handle. In
# case of more general mixed integer problems, we can use the ``bonmin`` solver, which
# can solve mixed integer nonlinear programs (MINLP). For example, let's consider the
# following MIQP problem
#
# .. math::
#    \min_{x \in \mathbb{Z}^n}{ \lVert A x - b \rVert_2^2 }
#
# The following code is very similar to the MILP case.

import numpy as np

m, n = 10, 5
A = np.random.rand(m, n)
b = np.random.randn(m)

nlp = Nlp[cs.MX](sym_type="MX")
x = nlp.variable("x", (n, 1), discrete=True)[0]
nlp.minimize(cs.sumsqr(A @ x - b))
nlp.init_solver(solver="bonmin")
sol = nlp.solve()

print(sol.vals["x"])
