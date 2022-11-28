# Inspired by https://www.youtube.com/watch?v=JI-AyLv68Xs&t=918s


import casadi as cs
import matplotlib.pyplot as plt
import numpy as np

try:
    from csnlp import Nlp, wrappers
except ImportError:
    import sys

    sys.path.insert(1, "src")
    from csnlp import Nlp, wrappers

plt.style.use("bmh")


# create dynamics
x = cs.MX.sym("x", 2)
u = cs.MX.sym("u")
ode = cs.vertcat((1 - x[1] ** 2) * x[0] - x[1] + u, x[0])
f = cs.Function("f", [x, u], [ode], ["x", "u"], ["ode"])
T = 10
N = 20
intg_options = {"tf": T / N, "simplify": True, "number_of_finite_elements": 4}
dae = {"x": x, "p": u, "ode": f(x, u)}
intg = cs.integrator("intg", "rk", dae, intg_options)
res = intg(x0=x, p=u)
x_next = res["xf"]
F = cs.Function("F", [x, u], [x_next], ["x", "u"], ["x_next"])

# build the MPC
sym_type = "SX"
shooting = "single"
mpc = wrappers.Mpc[Nlp](
    nlp=Nlp(sym_type=sym_type),
    prediction_horizon=N,
    shooting=shooting,
)
u, _ = mpc.action("u", lb=-1, ub=+1)
if shooting == "single":
    mpc.state("x", 2)
    mpc.dynamics = F
    x = mpc.states["x"]  # only accessible after dynamics have been set
    mpc.constraint("c0", x, ">=", -0.2)
else:
    x, _ = mpc.state("x", 2, lb=-0.2)  # must be created before dynamics
    mpc.dynamics = F
mpc.minimize(cs.sumsqr(x) + cs.sumsqr(u))

opts = {"print_time": False, "ipopt": {"sb": "yes", "print_level": 5}}
mpc.init_solver(opts)
sol = mpc.solve(pars={"x_0": [0, 1]})

t = np.linspace(0, T, N + 1)
plt.plot(t, sol.value(x).T)
plt.step(t[:-1], sol.vals["u"].T.full(), "-.", where="post")
plt.legend(["x1", "x2", "u"])
plt.xlim(t[0], t[-1])
plt.xlabel("t [s]")
plt.show()
