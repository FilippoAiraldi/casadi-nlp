# https://groups.google.com/g/casadi-users/c/izzPpN_FVYQ/m/w_HEcBI3BQAJ

import sys
import time
from typing import Tuple

import casadi as cs
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import Polynomial

from csnlp import Nlp, wrappers
from csnlp.core.data import array2cs, cs2array

plt.style.use("bmh")


def get_coeffs(d: int) -> Tuple[cs.DM, cs.DM, cs.DM]:
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


########################################################################################


nlp = Nlp[cs.SX]()

# create variables
nx, nu = 2, 1
X, _, _ = nlp.variable("x", (nx, N + 1), lb=[[-0.25], [-np.inf]])
XC, _, _ = nlp.variable("x_colloc", (nx, N * d), lb=[[-0.25], [-np.inf]])
U, _, _ = nlp.variable("u", (nu, N), lb=-1, ub=0.85)

# create initial state and perturbation parameter
x0 = nlp.parameter("x0", (nx, 1))  # == [0, 1]
nlp.constraint("init", X[:, 0], "==", x0)
p = nlp.parameter("p", (nx, 1))

# formulate nlp with collocation
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


# set objective and solver
nlp.minimize(J)
nlp.init_solver({"print_time": False, "ipopt": {"print_level": 0, "sb": "yes"}})

# solve nlp
sol = nlp.solve(pars={"x0": cs.DM([[0], [1]]), "p": 0})

# # plot result
# time = np.linspace(0, T, N + 1)
# x_opt = sol.vals["x"].full()
# u_opt = sol.vals["u"].full()
# plt.figure()
# plt.plot(time, x_opt[0], "--")
# plt.plot(time, x_opt[1], "-")
# plt.step(time[:-1], u_opt[0], "-.")
# plt.xlabel("t")
# plt.legend(["x1", "x2", "u"])
# plt.grid()
# plt.show()

# compute sensitivity of solution w.r.t. p
t0 = time.perf_counter()
nlp_ = wrappers.NlpSensitivity(nlp, target_parameters=p)
dfdp, d2fdp2 = nlp_.parametric_sensitivity(nlp.f, solution=sol, second_order=True)
duration1 = time.perf_counter() - t0


########################################################################################


# start with an empty NLP
J = 0.0
w, w0, lbw, ubw, g, lbg, ubg, x_plot, u_plot = [], [], [], [], [], [], [], [], []

# "lift" initial conditions
Xk = cs.MX.sym("X0", 2)
w.append(Xk)
lbw.append([0, 1])
ubw.append([0, 1])
w0.append([0, 1])
x_plot.append(Xk)

# perturb with P
P = cs.MX.sym("P", 2)
Xk += P

# Formulate the NLP
for k in range(N):
    # New NLP variable for the control
    Uk = cs.MX.sym("U_" + str(k))
    w.append(Uk)
    lbw.append([-1])
    ubw.append([0.85])
    w0.append([0])
    u_plot.append(Uk)

    # State at collocation points
    Xc = []
    for j in range(d):
        Xkj = cs.MX.sym("X_" + str(k) + "_" + str(j), 2)
        Xc.append(Xkj)
        w.append(Xkj)
        lbw.append([-0.25, -np.inf])
        ubw.append([np.inf, np.inf])
        w0.append([0, 0])

    # Loop over collocation points
    Xk_end = D[0] * Xk
    for j in range(1, d + 1):
        # Expression for the state derivative at the collocation point
        xp = C[0, j] * Xk
        for r in range(d):
            xp = xp + C[r + 1, j] * Xc[r]

        # Append collocation equations
        fj, qj = F(Xc[j - 1], Uk)
        g.append(h * fj - xp)
        lbg.append([0, 0])
        ubg.append([0, 0])

        # Add contribution to the end state
        Xk_end = Xk_end + D[j] * Xc[j - 1]

        # Add contribution to quadrature function
        J = J + B[j] * qj * h

    # New NLP variable for state at end of interval
    Xk = cs.MX.sym("X_" + str(k + 1), 2)
    w.append(Xk)
    lbw.append([-0.25, -np.inf])
    ubw.append([np.inf, np.inf])
    w0.append([0, 0])
    x_plot.append(Xk)

    # Add equality constraint
    g.append(Xk_end - Xk)
    lbg.append([0, 0])
    ubg.append([0, 0])

# Concatenate vectors
w, g = cs.vertcat(*w), cs.vertcat(*g)
x_plot, u_plot = cs.horzcat(*x_plot), cs.horzcat(*u_plot)
w0, lbw, ubw, lbg, ubg = (
    np.concatenate(w0),
    np.concatenate(lbw),
    np.concatenate(ubw),
    np.concatenate(lbg),
    np.concatenate(ubg),
)

# Create an NLP solver, using SQP and active-set QP for accurate multipliers
prob = {"f": J, "x": w, "g": g, "p": P}
opts = {
    "qpsol": "qrqp",
    "qpsol_options": {"print_iter": False, "error_on_fail": False},
    "print_time": False,
}
solver = cs.nlpsol("solver", "sqpmethod", prob, opts)
sol_ = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg, p=0)

# High-level approach:
# Use factory to e.g. to calculate Jacobian and Hessian of optimal f w.r.t. p
t0 = time.perf_counter()
jsolver = solver.factory("j", solver.name_in(), ["jac:f:p"])
hsolver = solver.factory("h", solver.name_in(), ["hess:f:p:p"])
kwargs = dict(
    x0=sol_["x"],
    lam_x0=sol_["lam_x"],
    lam_g0=sol_["lam_g"],
    lbx=lbw,
    ubx=ubw,
    lbg=lbg,
    ubg=ubg,
    p=0,
)
jsol = jsolver(**kwargs)
hsol = hsolver(**kwargs)
duration2 = time.perf_counter() - t0


# compare results
with open("log", "w") as sys.stdout:
    print("My Jacobian of f w.r.t. p:")
    print(dfdp.full().squeeze())
    print("\nMy Hessian of f w.r.t. p:")
    print(d2fdp2.full().squeeze())

    print("\nJacobian of f w.r.t. p:")
    print(jsol["jac_f_p"].full().squeeze())
    print("\nHessian of f w.r.t. p:")
    print(hsol["hess_f_p_p"].full().squeeze())

    print("\nComparison:")
    print(np.abs(jsol["jac_f_p"].T - dfdp).squeeze())
    print("\nHessian of f w.r.t. p:")
    print(np.abs(hsol["hess_f_p_p"] - d2fdp2).squeeze())
    print("\nDurations (mine vs native):", duration1, duration2)
