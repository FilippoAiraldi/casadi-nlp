import casadi as cs
import matplotlib.pyplot as plt
import numpy as np

from csnlp import Solution
from csnlp import multistart as ms

plt.style.use("bmh")


def func(x):
    return (
        -0.3 * x**2
        - np.exp(-10 * x**2)
        + np.exp(-100 * (x - 1) ** 2)
        + np.exp(-100 * (x - 1.5) ** 2)
    )


# build the NLP
N = 3
LB, UB = -0.5, 1.4
nlp = ms.StackedMultistartNlp[cs.SX](starts=N)
x = nlp.variable("x", lb=LB, ub=UB)[0]
nlp.parameter("p0")
nlp.parameter("p1")
nlp.minimize(func(x))
opts = {"print_time": False, "ipopt": {"sb": "yes", "print_level": 0}}
nlp.init_solver(opts)

# manually solve the problem from multiple initial conditions
x0s = list(
    ms.RandomStartPoints(
        points={"x": ms.RandomStartPoint("uniform", LB, UB)}, multistarts=3, seed=42
    )
)
xfs = [float(nlp.solve(pars={"p0": 0, "p1": 1}, vals0=x0).vals["x"]) for x0 in x0s]

# use automatic multistart solver
all_sols: list[Solution[cs.SX]] = nlp.solve_multi(  # type: ignore[assignment]
    pars={"p0": 0, "p1": 1},  # type: ignore[arg-type]
    vals0=x0s,
    return_all_sols=True,
)
best_sol = all_sols[np.argmin([s.f for s in all_sols])]

# plot function
fig, ax = plt.subplots(constrained_layout=True)
xs = np.linspace(LB, UB, 500)
ax.plot(xs, func(xs), "k--")

# plot manual solutions
for x0_, xf in zip(x0s, xfs):
    x0: float = x0_["x"]  # type: ignore[assignment]
    xs = np.linspace(x0, xf, 100)
    lbl = rf"$x_0={{{x0:.3f}}} \rightarrow f^{{\star}}={{{func(xf):.3f}}}$"
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
