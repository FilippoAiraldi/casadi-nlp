from typing import List

import casadi as cs
import matplotlib.pyplot as plt
import numpy as np

from csnlp import MultistartNlp
from csnlp.nlp.solutions import Solution

plt.style.use("bmh")


def func(x):
    return (
        -0.3 * x**2
        - np.exp(-10 * x**2)
        + np.exp(-100 * (x - 1) ** 2)
        + np.exp(-100 * (x - 1.5) ** 2)
    )


# build the NLPquit()
N = 3
LB, UB = -0.5, 1.4
nlp = MultistartNlp[cs.SX](starts=N)
x = nlp.variable("x", lb=LB, ub=UB)[0]
nlp.parameter("p0")
nlp.parameter("p1")
nlp.minimize(func(x))
opts = {"print_time": False, "ipopt": {"sb": "yes", "print_level": 0}}
nlp.init_solver(opts)

# manually solve the problem from multiple initial conditions
x0s = [0.9, 0.5, 1.1]
xfs = [
    float(nlp.solve(pars={"p0": 0, "p1": 1}, vals0={"x": x0}).vals["x"]) for x0 in x0s
]

# use automatical multistart solver
args = ([{"p0": 0, "p1": 1} for _ in x0s], [{"x": x0} for x0 in x0s])
best_sol: Solution[cs.SX] = nlp.solve_multi(*args)  # type: ignore
all_sols: List[Solution[cs.SX]] = nlp.solve_multi(*args, return_all_sols=True)  # type: ignore

# plot function
fig, ax = plt.subplots(constrained_layout=True)
xs = np.linspace(LB, UB, 500)
ax.plot(xs, func(xs), "k--")

# plot manual solutions
for x0, xf in zip(x0s, xfs):
    xs = np.linspace(x0, xf, 100)
    lbl = rf"$x_0={{{x0:.1f}}} \rightarrow f^{{\star}}={{{func(xf):.2f}}}$"
    c = ax.plot(xs, func(xs), "-", lw=2, label=lbl)[0]
    ax.plot(x0, func(x0), "o", markersize=6, color=c.get_color())
    ax.plot(xf, func(xf), "*", markersize=8, color=c.get_color())

# plot all multistart solutions
for sol in all_sols:
    ax.plot(sol.vals["x"], sol.f, "ko", markersize=12, fillstyle="none")

# plot best solution
ax.plot(best_sol.value(x), best_sol.value(nlp.f), "gs", markersize=14, fillstyle="none")
x = float(best_sol.vals["x"])
ax.vlines(x, -1.1, best_sol.f, "g", ls="-.")
ax.hlines(best_sol.f, LB, x, "g", ls="-.")

ax.set_xlabel("x")
ax.set_ylabel("f(x)")
ax.set_xlim(LB, UB)
ax.set_ylim(-1.1, 0.8)
ax.legend()
plt.show()
