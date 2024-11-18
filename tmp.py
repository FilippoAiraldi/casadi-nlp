import casadi as cs
import numpy as np
from numpy.polynomial import Polynomial

from csnlp import Nlp, wrappers
from csnlp.core.data import array2cs, cs2array


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

nlp = Nlp[cs.MX]("MX")

# create variables
nx, nu = 2, 1
X, _, _ = nlp.variable("x", (nx, N + 1), lb=[[-0.25], [-np.inf]])
XC, _, _ = nlp.variable("x_colloc", (nx, N * d), lb=[[-0.25], [-np.inf]])
U, _, _ = nlp.variable("u", (nu, N), lb=-1, ub=0.85)

# create initial state and perturbation parameter
nlp.constraint("init", X[:, 0], "==", [0, 1])
p = nlp.parameter("p", (nx, 1))
z = nlp.parameter("z", (1, 1))

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
        x[0] += p[0]
        x[1] += p[1] ** 2 / p[0]

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
nlp.minimize(J + z**2)

prob = {"f": nlp.f, "x": nlp.x, "g": nlp.g, "p": nlp.p}

opts = {
    "qpsol": "qrqp",
    "qpsol_options": {"print_iter": True, "error_on_fail": False},
    "print_time": True,
}
solver_sqp = cs.nlpsol("solver", "sqpmethod", prob, opts)
solver_ip = cs.nlpsol("solver", "ipopt", prob)

np.random.seed(0)
p_val = cs.vertcat(0.01, 0.02, 2.3)
kwargs = dict(lbx=nlp.lbx.data, ubx=nlp.ubx.data, lbg=0, ubg=0, p=p_val)

sol_sqp = solver_sqp(x0=0, **kwargs)
sol_ip = solver_ip(x0=0, **kwargs)

########################################################################################
# LAGRANGIAN SENSITIVITIES
########################################################################################

# # manual method
# snlp = wrappers.NlpSensitivity(nlp)
# grad_L_p = snlp.jacobian("L-p")
# hess_L_p_p = snlp.hessian("L-pp")  # approximate hessian
# nlp.init_solver(opts, "sqpmethod")
# sol0 = nlp.solve(pars={"p": p_val[:2], "z": p_val[-1]})
# grad_L_p_, hess_L_p_p_ = snlp.parametric_sensitivity(
#     snlp.lagrangian, second_order=True, solution=sol0
# )
# grad_L_p = sol0.value(grad_L_p)
# hess_L_p_p = sol0.value(hess_L_p_p)
# grad_L_p_ = sol0.value(grad_L_p_)
# hess_L_p_p_ = sol0.value(hess_L_p_p_)

# # build lagrangian manually - finds the FULL hessian - only with SQPMETHOD
# syms_in = {n: solver_sqp.mx_in(n) for n in solver_sqp.name_in()}
# syms_out = solver_sqp(**syms_in)
# lagrangian = syms_out["f"] + cs.dot(syms_out["lam_g"], syms_out["g"])
# S = cs.Function(
#     "S",
#     list(syms_in.values()),
#     list(syms_out.values()) + [lagrangian],
#     list(syms_in.keys()),
#     list(syms_out.keys()) + ["gamma"],
# )
# f1 = S.factory("s", S.name_in(), S.name_out() + ["grad:f:p", "hess:f:p:p"])
# sol1 = f1(x0=0, **kwargs)  # solves AND computes sensitivities all at once
# sensitivities1 = {n: sol1[n] for n in ["grad_gamma_p", "hess_gamma_p_p"]}

# # use get_function - finds only GRAD - works with IPOPT too
# f2 = solver_ip.get_function("nlp_grad")
# sensitivities2 = f2(x=sol_ip["x"], p=p_val, lam_f=1, lam_g=sol_ip["lam_g"])
# sensitivities2 = {n: sensitivities2[n] for n in ["grad_gamma_p"]}

# # use oracle+factory (similar to inner working of get_function) - finds APPROX hessian -
# # works with IPOPT too
# # NOTE: input lam_g here is just a numerical value, so it is impossible for it to
# # compute lam:g:p
# f3 = solver_ip.oracle().factory(
#     "lagrangian",
#     ["x", "p", "lam:f", "lam:g"],
#     ["grad:gamma:p", "hess:gamma:p:p"],
#     {"gamma": ["f", "g"]},
# )
# sensitivities3 = f3(x=sol_ip["x"], p=kwargs["p"], lam_f=1, lam_g=sol_ip["lam_g"])

########################################################################################
# PRIMAL (AND POSSIBLY DUAL) SENSITIVITIES
########################################################################################

# manual method
snlp = wrappers.NlpSensitivity(nlp)
nlp.init_solver()
sol4 = nlp.solve(pars={"p": p_val[:2], "z": p_val[-1]})
dydp4, _ = snlp.parametric_sensitivity(solution=sol4)

# via factory - works only with SQPMETHOD
names = ["jac:x:p", "jac:lam_x:p", "jac:lam_g:p", "jac:lam_p:p"]
f5 = solver_sqp.factory("pd", solver_sqp.name_in(), solver_sqp.name_out() + names)
sol5 = f5(x0=0, **kwargs)  # solves AND computes sensitivities all at once
sensitivities5 = {n: sol5[n.replace(":", "_")] for n in names}

# # via low-level forward/reverse - works only with SQPMETHOD
# sol6 = solver_ip(x0=0, **kwargs)  # base, standard solution
# nfwd = nlp.np
# fwd_solver = solver_ip.forward(nfwd)
# sol_fwd = fwd_solver(
#     x0=sol6["x"],
#     lam_x0=sol6["lam_x"],
#     lam_g0=sol6["lam_g"],
#     out_x=sol6["x"],
#     out_lam_g=sol6["lam_g"],
#     out_lam_x=sol6["lam_x"],
#     out_f=sol6["f"],
#     out_g=sol6["g"],
#     **kwargs,  # lbx, ubx, lbg, ubg, p
#     fwd_p=cs.DM.eye(nfwd),
# )
# sensitivities6 = sol_fwd["fwd_x"]

# via KKT
kkt_fun = solver_ip.oracle().factory(
    "kkt",
    ["x", "p", "lam:f", "lam:g"],
    ["jac:g:x", "hess:gamma:x:x", "jac:g:p", "hess:gamma:x:p"],
    {"gamma": ["f", "g"]},
)
kkt_elements = kkt_fun(x=sol_ip["x"], p=kwargs["p"], lam_f=1, lam_g=sol_ip["lam_g"])
Jg = kkt_elements["jac_g_x"]
Hl = kkt_elements["hess_gamma_x_x"]
min_lam_ = 0.0
ubIx = sol_ip["lam_x"] > min_lam_
lbIx = sol_ip["lam_x"] < -min_lam_
bIx = ubIx + lbIx
iIx = 1 - bIx
ubIg = sol_ip["lam_g"] > min_lam_
lbIg = sol_ip["lam_g"] < -min_lam_
bIg = ubIg + lbIg
iIg = 1 - bIg
H_11 = cs.mtimes(cs.diag(iIx), Hl) + cs.diag(bIx)
H_12 = cs.mtimes(cs.diag(iIx), Jg.T)
H_21 = cs.mtimes(cs.diag(bIg), Jg)
H_22 = cs.diag(-iIg)
H = cs.blockcat([[H_11, H_12], [H_21, H_22]])  # probably Ky
Kp = cs.vertcat(kkt_elements["hess_gamma_x_p"], kkt_elements["jac_g_p"])

dydp__ = -np.linalg.solve(H.full(), Kp.full())

quit()

########################################################################################
# TESTS WITH FORWARD/REVERSE
########################################################################################

# nfwd = 3
# sol = solver_sqp(x0=0, **kwargs)

# # Forward mode AD for the NLP solver object
# fwd_solver = solver_sqp.forward(nfwd)
# print("fwd_solver generated")

# # Seeds, initalized to zero
# fwd_lbx = [cs.DM.zeros(sol["x"].sparsity()) for _ in range(nfwd)]
# fwd_ubx = [cs.DM.zeros(sol["x"].sparsity()) for _ in range(nfwd)]
# fwd_p = [cs.DM.zeros(nlp.p.sparsity()) for _ in range(nfwd)]
# fwd_lbg = [cs.DM.zeros(sol["g"].sparsity()) for _ in range(nfwd)]
# fwd_ubg = [cs.DM.zeros(sol["g"].sparsity()) for _ in range(nfwd)]

# # Let's preturb P
# fwd_p[0][0] = 1  # first nonzero of P
# fwd_p[1][1] = 1  # second nonzero of P
# fwd_p[2][2] = 1  # correct??

# # Calculate sensitivities using AD
# sol_fwd = fwd_solver(
#     x0=sol["x"], lam_x0=sol["lam_x"], lam_g0=sol["lam_g"],
#     out_x=sol["x"],
#     out_lam_g=sol["lam_g"],
#     out_lam_x=sol["lam_x"],
#     out_f=sol["f"],
#     out_g=sol["g"],
#     **kwargs,  # lbx, ubx, lbg, ubg, p
#     fwd_lbx=cs.horzcat(*fwd_lbx),
#     fwd_ubx=cs.horzcat(*fwd_ubx),
#     fwd_lbg=cs.horzcat(*fwd_lbg),
#     fwd_ubg=cs.horzcat(*fwd_ubg),
#     fwd_p=cs.horzcat(*fwd_p),
# )

# # Calculate the same thing using finite differences
# h = 1e-3
# pert = []
# for d in range(nfwd):
#     pert.append(
#         solver_sqp(
#             x0=sol["x"],
#             lam_g0=sol["lam_g"],
#             lam_x0=sol["lam_x"],
#             lbx=kwargs["lbx"] + h * (fwd_lbx[d] + fwd_ubx[d]),
#             ubx=kwargs["ubx"] + h * (fwd_lbx[d] + fwd_ubx[d]),
#             lbg=kwargs["lbg"] + h * (fwd_lbg[d] + fwd_ubg[d]),
#             ubg=kwargs["ubg"] + h * (fwd_lbg[d] + fwd_ubg[d]),
#             p=p_val + h * fwd_p[d],
#         )
#     )

# # Print the result
# for s in ["f"]:
#     print("==========")
#     print("Checking " + s)
#     print("finite differences")
#     for d in range(nfwd):
#         print((pert[d][s] - sol[s]) / h)
#     print("AD fwd")
#     M = sol_fwd["fwd_" + s].full()
#     for d in range(nfwd):
#         print(M[:, d].flatten())

# # Perturb again, in the opposite direction for second order derivatives
# pert2 = []
# for d in range(nfwd):
#     pert2.append(
#         solver_sqp(
#             x0=sol["x"],
#             lam_g0=sol["lam_g"],
#             lam_x0=sol["lam_x"],
#             lbx=kwargs["lbx"] - h * (fwd_lbx[d] + fwd_ubx[d]),
#             ubx=kwargs["ubx"] - h * (fwd_lbx[d] + fwd_ubx[d]),
#             lbg=kwargs["lbg"] - h * (fwd_lbg[d] + fwd_ubg[d]),
#             ubg=kwargs["ubg"] - h * (fwd_lbg[d] + fwd_ubg[d]),
#             p=p_val - h * fwd_p[d],
#         )
#     )

# # Print the result
# for s in ["f"]:
#     print("finite differences, second order: " + s)
#     for d in range(nfwd):
#         print((pert[d][s] - 2 * sol[s] + pert2[d][s]) / (h * h))


# ########################################################################################

# nadj = 1
# adj_solver = solver_sqp.reverse(nadj)
# print("adj_solver generated")

# # Seeds, initalized to zero
# adj_f = [cs.DM.zeros(sol["f"].sparsity()) for _ in range(nadj)]
# adj_g = [cs.DM.zeros(sol["g"].sparsity()) for _ in range(nadj)]
# adj_x = [cs.DM.zeros(sol["x"].sparsity()) for _ in range(nadj)]

# # Study which inputs influence f
# adj_f[0][0] = 1

# # Calculate sensitivities using AD
# sol_adj = adj_solver(
#     out_x=sol["x"],
#     out_lam_g=sol["lam_g"],
#     out_lam_x=sol["lam_x"],
#     out_f=sol["f"],
#     out_g=sol["g"],
#     **kwargs,  # lbx, ubx, lbg, ubg, p
#     adj_f=cs.horzcat(*adj_f),
#     adj_g=cs.horzcat(*adj_g),
#     adj_x=cs.horzcat(*adj_x),
# )

# # Print the result
# for s in ["p"]:
#     print("==========")
#     print("Checking " + s)
#     print("Reverse mode AD")
#     print(sol_adj["adj_" + s])

# quit()
