r"""
PWA MPC controller
==================

This example demos how an MPC controller can be built for a system with
piecewise affine dynamics using :class:`csnlp.wrappers.HybridMpc`.
"""
import os
os.environ["ARTELYS_LICENSE"] = r"C:\Program Files\Artelys\artelys_lic_2024-10-07_trial_full_knitro_14.0_Samuel_Mallick_62-6a-f1-5c-8d.txt"
import casadi as cs
import numpy as np

from csnlp import Nlp, wrappers

N = 5

# build dynamics of spring
tau = 0.5  # sampling time for discretization
k1 = 10  # spring constant when one-sided spring active
k2 = 1  # spring constant when one-sided spring not active
damp = 4  # damping constant
mass = 10  # mass of the system
A_spring_1 = np.array([[1, tau], [-((tau * 2 * k1) / mass), 1 - (tau * damp) / mass]])
A_spring_2 = np.array([[1, tau], [-((tau * 2 * k2) / mass), 1 - (tau * damp) / mass]])
B_spring = np.array([[0], [tau / mass]])
x1_lim = 5
x2_lim = 5
u_lim = 20

# Regions - switch about origin
S = [np.array([[1, 0]]), np.array([[-1, 0]])]
R = [np.zeros((1, 1)), np.zeros((1, 1))]
T = [np.array([[0]]), np.array([[0]])]
# Dynamics
A = [A_spring_1, A_spring_2]
B = [B_spring, B_spring]
c = [np.array([[0], [0]]), np.array([[0], [0]])]
# bounds
D = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
E = np.array([[x1_lim], [x1_lim], [x2_lim], [x2_lim]])
F = np.array([[1], [-1]])
G = np.array([[u_lim], [u_lim]])

mpc = wrappers.HybridMpc(
    nlp=Nlp[cs.SX](sym_type="SX"), prediction_horizon=N, shooting="multi"
)
x, _ = mpc.state("x", 2)
u, _ = mpc.action("u", lb=-1, ub=+1)
mpc.set_pwa_dynamics({"A": A, "B": B, "c": c, "S": S, "R": R, "T": T, "D": D, "E": E, "F": F, "G": G})
# vals0 = {'delta': np.zeros(mpc.variables['delta'].shape)}
# vals0['delta'][1, :] = 1
mpc.minimize(cs.sumsqr(x) + cs.sumsqr(u))
mpc.init_solver(solver="knitro")
sol = mpc.solve(pars={"x_0": [0.5, 0]})
pass
