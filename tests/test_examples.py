import unittest
import casadi as cs
from csnlp import Nlp
import numpy as np
from scipy import io


OPTS = {
    "expand": True,
    "print_time": False,
    "ipopt": {
        "max_iter": 500,
        "sb": "yes",
        # for debugging
        "print_level": 0,
        "print_user_options": "no",
        "print_options_documentation": "no",
    },
}
RESULTS = io.loadmat(r"tests/example_data.mat", simplify_cells=True)


class TestChain(unittest.TestCase):
    def test(self):
        N = 40
        m = 40 / N
        D = 70 * N / 2
        g = 9.81
        L = 1
        nlp = Nlp()
        p = nlp.variable("p", (2, N))[0]
        x, y = p[0, :], p[1, :]
        V = D * cs.sum2(
            (cs.sqrt(cs.diff(x) ** 2 + cs.diff(y) ** 2) - L / N) ** 2
        ) + g * m * cs.sum2(y)
        nlp.minimize(V)
        nlp.constraint("c1", p[:, 0], "==", [-2, 1])
        nlp.constraint("c2", p[:, -1], "==", [2, 1])
        nlp.init_solver(OPTS)
        nlp.constraint("c3", y, ">=", cs.cos(0.1 * x) - 0.5)
        sol = nlp.solve(
            vals0={"p": np.row_stack((np.linspace(-2, 2, N), np.ones(y.shape)))}
        )
        np.testing.assert_allclose(sol.vals["p"], RESULTS['chain_p'], rtol=1e-7, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
