# Reproduces https://web.casadi.org/blog/opti/chain.m


from casadi_nlp import Nlp
import casadi as cs
import numpy as np
import matplotlib.pyplot as plt


N = 40
m = 40 / N
D = 70 * N / 2
g = 9.81
L = 1
nlp = Nlp()
fig, axs = plt.subplots(
    1, 3, sharey=True, constrained_layout=True, figsize=(7, 2))


# Problem 1: Simple hanging chain
p = nlp.variable('p', (2, N))[0]
x, y = p[0, :], p[1, :]
V = D * (cs.sumsqr(cs.diff(x)) + cs.sumsqr(cs.diff(y))) + g * m * cs.sum2(y)
nlp.minimize(V)
nlp.constraint('c1', p[:, 0], '==', [-2, 1])
nlp.constraint('c2', p[:, -1], '==', [2, 1])
nlp.init_solver()
sol = nlp.solve()

axs[0].plot(sol.value(x).full().flat, sol.value(y).full().flat, 'o-')


# Problem 2: adding ground constraints
nlp.constraint('c3', y, '>=', cs.cos(0.1 * x) - 0.5)
sol = nlp.solve(vals0={'p': sol.vals['p']})  # warm-starts the new solver run

axs[1].plot(sol.value(x).full().flat, sol.value(y).full().flat, 'o-')
xs = np.linspace(-2, 2, 100)
axs[1].plot(xs, np.cos(0.1 * xs) - 0.5, 'r--')


# Problem 3: Rest Length
V = D * cs.sum2((cs.sqrt(cs.diff(x)**2 + cs.diff(y)**2) - L / N)**2) \
    + g * m * cs.sum2(y)
nlp.minimize(V)
sol = nlp.solve(vals0={
    'p': np.row_stack((np.linspace(-2, 2, N), np.ones(y.shape)))
})

axs[2].plot(sol.value(x).full().flat, sol.value(y).full().flat, 'o-')
xs = np.linspace(-2, 2, 100)
axs[2].plot(xs, np.cos(0.1 * xs) - 0.5, 'r--')

plt.show()
