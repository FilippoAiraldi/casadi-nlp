import unittest

import casadi as cs
import numpy as np
from parameterized import parameterized, parameterized_class
from scipy import io

from csnlp import MultistartNlp, Nlp, wrappers

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
RESULTS = io.loadmat(r"tests/examples_data.mat", simplify_cells=True)


@parameterized_class("sym_type", [("SX",), ("MX",)])
class TestExamples(unittest.TestCase):
    def test__chain(self):
        N = 40
        m = 40 / N
        D = 70 * N / 2
        g = 9.81
        L = 1
        nlp = Nlp(sym_type=self.sym_type)
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
        np.testing.assert_allclose(sol.vals["p"], RESULTS["chain_p"])

    def test__multistart_nlp(self):
        def func(x):
            return (
                -0.3 * x**2
                - np.exp(-10 * x**2)
                + np.exp(-100 * (x - 1) ** 2)
                + np.exp(-100 * (x - 1.5) ** 2)
            )

        N = 3
        LB, UB = -0.5, 1.4
        x0s = [0.9, 0.5, 1.1]
        nlp = MultistartNlp(starts=N, sym_type=self.sym_type)
        x = nlp.variable("x", lb=LB, ub=UB)[0]
        nlp.parameter("p0")
        nlp.parameter("p1")
        nlp.minimize(func(x))
        nlp.init_solver(OPTS)
        args = ([{"p0": 0, "p1": 1} for _ in x0s], [{"x": x0} for x0 in x0s])
        best_sol = nlp.solve_multi(*args)
        all_sols = nlp.solve_multi(*args, return_all_sols=True)
        fs = [all_sol.f for all_sol in all_sols]
        self.assertEqual(best_sol.f, min(fs))
        np.testing.assert_allclose(fs, RESULTS["multistart_nlp_fs"])

    @parameterized.expand([("single",), ("multi",)])
    def test__optimal_ctrl(self, shooting: str):
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
        mpc = wrappers.Mpc(
            nlp=Nlp(sym_type=self.sym_type),
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
        mpc.init_solver(OPTS)
        sol = mpc.solve(pars={"x_0": [0, 1]})
        u_opt = sol.vals["u"].full().flat
        np.testing.assert_allclose(
            u_opt, RESULTS["optimal_ctrl_u"], rtol=1e-6, atol=1e-6
        )

    def test__rosenbrock(self):
        nlp = Nlp(sym_type=self.sym_type)
        x = nlp.variable("x", (2, 1))[0]
        r = nlp.parameter("r")
        f = (1 - x[0]) ** 2 + (x[1] - x[0] ** 2) ** 2
        nlp.minimize(f)
        _, lam = nlp.constraint("con1", cs.sumsqr(x), "<=", r)
        nlp.init_solver(OPTS)
        r_values = np.linspace(1, 3, 25)
        f_values = []
        lam_values = []
        for r_value in r_values:
            sol = nlp.solve(pars={"r": r_value})
            f_values.append(sol.f)
            lam_values.append(sol.value(lam))
        f_values = np.array(f_values).squeeze()
        lam_values = np.array(lam_values).squeeze()
        np.testing.assert_allclose(f_values, RESULTS["rosenbrock_f"])
        np.testing.assert_allclose(lam_values, RESULTS["rosenbrock_lam"])

    def test__sensitivity(self):
        def z1(x, lam, p):
            return (x[1, :] ** p[1] - x[0, :]) * cs.exp(-10 * lam + p[0]) / p[1]

        def z2(x):
            return x[1, :] - x[0, :]

        def z3(x, p):
            return x[1, :] ** (1 / p[1]) - x[0, :]

        def z4(lam, p):
            return cs.exp(-10 * lam + p[0]) / p[1]

        nlp = wrappers.NlpSensitivity(Nlp(sym_type=self.sym_type))
        x = nlp.variable("x", (2, 1), lb=[[0], [-np.inf]])[0]
        p = nlp.parameter("p", (2, 1))
        nlp.minimize((1 - x[0]) ** 2 + p[0] * (x[1] - x[0] ** 2) ** 2)
        g = (x[0] + 0.5) ** 2 + x[1] ** 2
        nlp.constraint("c1", (p[1] / 2) ** 2, "<=", g)
        _, lam = nlp.constraint("c2", g, "<=", p[1] ** 2)
        opts = {"print_time": False, "ipopt": {"sb": "yes", "print_level": 0}}
        nlp.init_solver(opts)
        Z = cs.blockcat(
            [
                [z1(x, lam, p) ** 2, z2(x)],
                [1 / (1 + z3(x, p)), z4(lam, p)],
                [z3(x, p) * (-1 - 10 * z1(x, lam, p)), z4(lam, p) / (1 + z2(x))],
            ]
        )
        z0, j0, h0 = [], [], []
        J, H = nlp.parametric_sensitivity(expr=Z)
        for p0 in (1.2, 1.45, 1.9):
            sol = nlp.solve(pars={"p": [0.2, p0]})
            z0.append(sol.value(Z))
            j0.append(sol.value(J))
            h0.append(sol.value(H))
        z0 = np.array(z0).squeeze()
        j0 = np.array(j0).squeeze()
        h0 = np.array(h0).squeeze()
        np.testing.assert_allclose(RESULTS["sensitivity_z"], z0, rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(RESULTS["sensitivity_j"], j0, rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(RESULTS["sensitivity_h"], h0, rtol=1e-6, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
