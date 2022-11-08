# Reproduces https://web.casadi.org/blog/opti/rosenbrock.m


from casadi_nlp import Nlp
import casadi as cs
import numpy as np
import matplotlib.pyplot as plt


nlp = Nlp()
x = nlp.variable('x', (2, 1))[0]
r = nlp.parameter('r')

f = (1 - x[0])**2 + (x[1] - x[0]**2)**2
nlp.minimize(f)
_, lam = nlp.constraint('con1', cs.sumsqr(x), '<=', r)

nlp.init_solver()  # initializes ipopt under the hood

r_values = np.linspace(1, 3, 25)
f_values = []
lam_values = []
for r_value in r_values:
    sol = nlp.solve(pars={'r': r_value})
    f_values.append(sol.f)
    lam_values.append(sol.value(lam))

fig, ax = plt.subplots(constrained_layout=True)

ax.plot(r_values, f_values, 'o')

ts = np.linspace(-0.02, 0.02)
for r_value, f_value, lam_value in zip(r_values, f_values, lam_values):
    ax.plot(r_value + ts, -lam_value * ts + f_value, 'r-')

ax.set_xlabel('Value of r')
ax.set_ylabel('Objective value at solution')
plt.show()
