"""
Reproduces the sensitivity example from
https://github.com/casadi/casadi/blob/develop/docs/examples/python/nlp_sensitivities.py
"""

import time

import casadi as cs
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import Polynomial

from csnlp import Nlp, wrappers
from csnlp.core.data import array2cs, cs2array

plt.style.use("bmh")


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
N = 20  # number of control intervals
h = T / N

# coefficients and dynamics and cost functions
d = 3
C, D, B = get_coeffs(d)
F = get_dynamics_and_cost()

# create nlp problem
nlp = Nlp[cs.SX]()

# create variables
nx, nu = 2, 1
X, _, _ = nlp.variable("x", (nx, N + 1), lb=[[-0.25], [-np.inf]])
XC, _, _ = nlp.variable("x_colloc", (nx, N * d), lb=[[-0.25], [-np.inf]])
U, _, _ = nlp.variable("u", (nu, N), lb=-1, ub=0.85)

# create initial state and perturbation parameter
x0 = nlp.parameter("x0", (nx, 1))
nlp.constraint("init", X[:, 0], "==", x0)
p = nlp.parameter("p", (nx, 1))
x0_ = cs.DM([[0], [1]])
p_ = cs.DM([[0], [0]])

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
sol = nlp.solve(pars={"x0": x0_, "p": p_})

# compute sensitivity of solution w.r.t. p via wrapper (allows to use IPOPT)
t0 = time.perf_counter()
nlp_ = wrappers.NlpSensitivity(nlp, target_parameters=p)
dfdp, d2fdp2 = nlp_.parametric_sensitivity(nlp.f, solution=sol, second_order=True)
t1 = time.perf_counter() - t0
assert d2fdp2 is not None

# high-level approach, native to CasADi
prob = {"f": nlp.f, "x": nlp.x, "g": nlp.g, "p": nlp.p}
opts = {
    "qpsol": "qrqp",
    "qpsol_options": {"print_iter": False, "error_on_fail": False},
    "print_time": False,
}
solver = cs.nlpsol("solver", "sqpmethod", prob, opts)
sol_ = solver(x0=0, lbx=nlp.lbx, ubx=nlp.ubx, lbg=0, ubg=0, p=cs.vertcat(x0_, p_))

t2 = time.perf_counter()
jsolver = solver.factory("j", solver.name_in(), ["jac:f:p"])
hsolver = solver.factory("h", solver.name_in(), ["hess:f:p:p"])
kwargs = dict(
    x0=sol_["x"],
    lam_x0=sol_["lam_x"],
    lam_g0=sol_["lam_g"],
    lbx=nlp.lbx,
    ubx=nlp.ubx,
    lbg=0,
    ubg=0,
    p=cs.vertcat(x0_, p_),
)
jsol = jsolver(**kwargs)
hsol = hsolver(**kwargs)
t3 = time.perf_counter() - t2

# compare results
dfdp = dfdp.full().squeeze()
d2fdp2 = d2fdp2.full().squeeze()
print("My Jacobian of f w.r.t. p:")
print(dfdp)
print("\nMy Hessian of f w.r.t. p:")
print(d2fdp2)

jac_f_p = jsol["jac_f_p"].full().squeeze()[2:]
hess_f_p_p = hsol["hess_f_p_p"].full()[2:, 2:]
i_lower = np.tril_indices(hess_f_p_p.shape[0], -1)
hess_f_p_p[i_lower] = hess_f_p_p.T[i_lower]
print("\nJacobian of f w.r.t. p:")
print(jac_f_p)
print("\nHessian of f w.r.t. p:")
print(hess_f_p_p)

print("\nComparison:")
print(np.abs(jac_f_p.T - dfdp))
print("\nHessian of f w.r.t. p:")
print(np.abs(hess_f_p_p - d2fdp2))
print("\nDurations (mine vs native):", t1, "vs", t3)

# plot result
t = np.linspace(0, T, N + 1)
x_opt = sol.vals["x"].full()
u_opt = sol.vals["u"].full()
plt.figure()
plt.plot(t, x_opt[0], "--")
plt.plot(t, x_opt[1], "-")
plt.step(t[:-1], u_opt[0], "-.")
plt.xlabel("t")
plt.legend(["x1", "x2", "u"])
plt.grid()
plt.show()
