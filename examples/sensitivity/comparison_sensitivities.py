"""
Comparison of CasADi's and :mod:`csnlp`'s sensitivity computations
==================================================================

In this example, we reproduce the sensitivity computations from
`this CasADi example <https://github.com/casadi/casadi/blob/develop/docs/examples/python/nlp_sensitivities.py>`_,
both with :mod:`casadi`'s native capabilities as well as the one from :mod:`csnlp`.

This example assumes you are already familiar with the basics of computing sensitivities
via :class:`csnlp.wrappers.NlpSensitivity`. If this is not the case, please refer to the
example :ref:`simple_sensitivity_example` for a more introductory example.
"""

import time

import casadi as cs
import numpy as np
from numpy.polynomial import Polynomial

from csnlp import Nlp, wrappers
from csnlp.core.data import array2cs, cs2array

# %%
# With CasADi's native functionality
# ----------------------------------
# We start by defining the collocation coefficients, the dynamics and cost functions,
# and some constants of the optimal control problem.


def get_coeffs(d: int) -> tuple[cs.DM, cs.DM, cs.DM]:
    """Gets the coefficients of the collocation equation, of the continuity equation,
    and lastly of the quadrature function."""
    tau_root = np.append(0, cs.collocation_points(d, "legendre"))
    C = cs.DM.zeros(d + 1, d + 1)
    D = cs.DM.zeros(d + 1)
    B = cs.DM.zeros(d + 1)
    for j in range(d + 1):
        p = Polynomial([1])
        for r in range(d + 1):
            if r != j:
                p *= Polynomial([-tau_root[r], 1]) / (tau_root[j] - tau_root[r])
        D[j] = p(1.0)
        pder = p.deriv()
        for r in range(d + 1):
            C[j, r] = pder(tau_root[r])
        pint = p.integ()
        B[j] = pint(1.0)
    return C, D, B


def get_dynamics_and_cost() -> cs.Function:
    """Returns the dynamics and objective function of the problem."""
    x1 = cs.SX.sym("x1")
    x2 = cs.SX.sym("x2")
    x = cs.vertcat(x1, x2)
    u = cs.SX.sym("u")
    xdot = cs.vertcat((1 - x2**2) * x1 - x2 + u, x1)
    L = x1**2 + x2**2 + u**2
    return cs.Function("f", [x, u], [xdot, L], ["x", "u"], ["xdot", "L"])


# time and control horizon
T = 10.0
N = 6  # number of control intervals
h = T / N

# coefficients and dynamics and cost functions
d = 3
C, D, B = get_coeffs(d)
F = get_dynamics_and_cost()

# %%
# Then, we build the problem itself. This time we use collocation points to enforce
# the dynamics and the continuity of the states. We also add a perturbation parameter
# to the initial state.

nlp = Nlp[cs.SX]()

# create variables
nx, nu = 2, 1
X, _, _ = nlp.variable("x", (nx, N + 1), lb=[[-0.25], [-np.inf]])
XC, _, _ = nlp.variable("x_colloc", (nx, N * d), lb=[[-0.25], [-np.inf]])
U, _, _ = nlp.variable("u", (nu, N), lb=-1, ub=0.85)

# create initial state and perturbation parameter
nlp.constraint("init", X[:, 0], "==", [0, 1])
p = nlp.parameter("p", (nx, 1))

# formulate NLP with collocation
J = 0.0
X_ = cs2array(X).T
XC_ = cs2array(XC).reshape((nx, N, d)).transpose((1, 0, 2))
U_ = cs2array(U).T
for k, (x, x_next, xc, u) in enumerate(zip(X_[:-1], X_[1:], XC_, U_)):
    # convert back to symbolic
    x = array2cs(x)
    x_next = array2cs(x_next)
    xc = array2cs(xc)
    u = array2cs(u)

    # perturb first state with p
    if k == 0:
        x += p

    # propagate collocation points and compute associated cost
    f_k, q_k = F(xc, u)

    # create collocation constraints
    xp = x @ C[0, 1:] + xc @ C[1:, 1:]
    nlp.constraint(f"colloc_{k}", h * f_k, "==", xp)

    # Add contribution to quadrature function
    J += h * (q_k @ B[1:])

    # create end state and constraint
    x_k_end = D[0] * x + xc @ D[1:]
    nlp.constraint(f"colloc_end_{k}", x_next, "==", x_k_end)


# set objective
nlp.minimize(J)

# %%
# Now that the problem has been created, we can use the high-level approach to
# sensitivity, native to CasADi. To use it, it is mandatory to use the
# ``sqpmethod`` solver.
#
# First, we convert the problem to a ``sqpmethod``-based problem, and solve it
# numerically.

prob = {"f": nlp.f, "x": nlp.x, "g": nlp.g, "p": nlp.p}

opts = {
    "qpsol": "qrqp",
    "qpsol_options": {"print_iter": False, "error_on_fail": False},
    "print_time": False,
}
solver = cs.nlpsol("solver", "sqpmethod", prob, opts)
kwargs = dict(lbx=nlp.lbx.data, ubx=nlp.ubx.data, lbg=0, ubg=0, p=0)
sol = solver(x0=0, **kwargs)

# %%
# We then spawn, with the native factory method, a new solver that yields the jacobian
# and hessian of the objective w.r.t. NLP parameters. Since we have already compute a
# solution, we can use it to warm-start the new solver.

t0 = time.perf_counter()

# create the jacobian and hessian solver
jac_and_hess_solver = solver.factory("s", solver.name_in(), ["jac:f:p", "hess:f:p:p"])

# call it with the previous solution to warm-start it
jac_and_hess_sol = jac_and_hess_solver(
    x0=sol["x"], lam_x0=sol["lam_x"], lam_g0=sol["lam_g"], **kwargs
)

t1 = time.perf_counter() - t0

# retrieve the arrays
jac_f_p = jac_and_hess_sol["jac_f_p"].full().squeeze()
hess_f_p_p = jac_and_hess_sol["hess_f_p_p"].full()

# %%
# With :mod:`csnlp`
# ----------------------------------
# We'll now turn our attention on how to compute the same sensitivities with our
# package. Since it does not rely on the ``sqpmethod`` solver, we can use any,
# especially IPOPT.
#
# First, let us initialize IPOPT and compute the same solution.

nlp.init_solver({"print_time": False, "ipopt": {"print_level": 0, "sb": "yes"}})
sol_ = nlp.solve(pars={"p": 0})

# %%
# Then, we can compute the sensitivities of the solution w.r.t. the parameter ``p`` with
# the :class:`csnlp.wrappers.NlpSensitivity` class. By passing a solution object to the
# :meth:`csnlp.wrappers.NlpSensitivity.parametric_sensitivity` method, computations are
# sped up because they are numerical.

nlp_ = wrappers.NlpSensitivity(nlp)

t2 = time.perf_counter()

dfdp, d2fdp2 = nlp_.parametric_sensitivity(nlp.f, solution=sol_, second_order=True)

t3 = time.perf_counter() - t2

dfdp = dfdp.full().squeeze()
d2fdp2 = d2fdp2.full()

# %%
# Comparison
# ----------
# We can now compare the results of the two methods.

print(
    "\nComparison",
    "Abs. differences in Jacobian of f w.r.t. p:",
    np.abs(jac_f_p.T - dfdp),
    "Abs. differences in Hessian of f w.r.t. p:",
    np.abs(hess_f_p_p - d2fdp2),
    sep="\n",
)

# %%
# The price we pay for such a general approach is that our method is invariably slower
# than the native one, especially as the problem size grows with ``N``. Specifically,
# the computations of the hessian are the slowest. The jacobian computations can
# sustain larger sizes.

print(
    "\nTimings", "CasADi time:", f" {t1:.6f} s", "csnlp time:", f" {t3:.6f} s", sep="\n"
)
