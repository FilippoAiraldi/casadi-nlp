import pickle
import unittest
import warnings
from copy import deepcopy

import casadi as cs
import numpy as np
from parameterized import parameterized, parameterized_class

from csnlp import Nlp, scaling
from csnlp.core.solutions import subsevalf
from csnlp.util.math import log
from csnlp.wrappers import (
    Mpc,
    NlpScaling,
    NlpSensitivity,
    NonRetroactiveWrapper,
    Wrapper,
)

OPTS = {
    "expand": True,
    "print_time": False,
    "ipopt": {
        "max_iter": 500,
        "tol": 1e-7,
        "barrier_tol_factor": 5,
        "sb": "yes",
        # for debugging
        "print_level": 0,
        "print_user_options": "no",
        "print_options_documentation": "no",
    },
}


class TestWrapper(unittest.TestCase):
    def test_attr__raises__when_accessing_private_attrs(self):
        nlp = Nlp()
        nlp._x
        wrapped = Wrapper[cs.SX](nlp)
        with self.assertRaisesRegex(
            AttributeError, "Accessing private attribute '_x' is prohibited."
        ):
            wrapped._x

    def test_unwrapped__unwraps_nlp_correctly(self):
        nlp = Nlp()
        self.assertIs(nlp, nlp.unwrapped)
        wrapped = Wrapper[cs.SX](nlp)
        self.assertIs(nlp, wrapped.unwrapped)

    def test_str_and_repr(self):
        nlp = Nlp()
        wrapped = Wrapper[cs.SX](nlp)
        S = wrapped.__str__()
        self.assertIn(Wrapper.__name__, S)
        self.assertIn(nlp.__str__(), S)
        S = wrapped.__repr__()
        self.assertIn(Wrapper.__name__, S)
        self.assertIn(nlp.__repr__(), S)

    def test_is_wrapped(self):
        nlp = Nlp()
        self.assertFalse(nlp.is_wrapped())

        wrapped = Mpc[cs.SX](nlp=nlp, prediction_horizon=10)
        self.assertTrue(wrapped.is_wrapped(Mpc))
        self.assertFalse(wrapped.is_wrapped(NlpSensitivity))

        wrapped = NlpSensitivity[cs.SX](nlp=nlp)
        self.assertFalse(wrapped.is_wrapped(Mpc))
        self.assertTrue(wrapped.is_wrapped(NlpSensitivity))

        wrapped = NlpSensitivity[cs.SX](nlp=Mpc(nlp=nlp, prediction_horizon=10))
        self.assertTrue(wrapped.is_wrapped(Mpc))
        self.assertTrue(wrapped.is_wrapped(NlpSensitivity))

        with cs.global_pickle_context():
            pickled = pickle.dumps(wrapped)
        with cs.global_unpickle_context():
            wrapped_pickled = pickle.loads(pickled)
        wrapped_copied = deepcopy(wrapped)

        self.assertEqual(wrapped.name, wrapped_pickled.name)
        self.assertEqual(repr(wrapped), repr(wrapped_pickled))
        self.assertEqual(wrapped.name, wrapped_copied.name)
        self.assertEqual(str(wrapped), str(wrapped_copied))
        self.assertEqual(repr(wrapped), repr(wrapped_copied))


class TestNonRetroactiveWrapper(unittest.TestCase):
    def test_init__raises__with_variable_already_defined(self):
        nlp = Nlp()
        nlp.variable("x")
        with self.assertRaisesRegex(ValueError, "Nlp already defined."):
            NonRetroactiveWrapper(nlp)

    def test_init__raises__with_parameter_already_defined(self):
        nlp = Nlp()
        nlp.parameter("p")
        with self.assertRaisesRegex(ValueError, "Nlp already defined."):
            NonRetroactiveWrapper(nlp)


@parameterized_class("sym_type", [("SX",), ("MX",)])
class TestNlpSensitivity(unittest.TestCase):
    def test_lagrangian__is_correct__example_1a_b(self):
        # https://en.wikipedia.org/wiki/Lagrange_multiplier#Example_1
        nlp = NlpSensitivity(Nlp(sym_type=self.sym_type))
        x = nlp.variable("x")[0]
        y = nlp.variable("y")[0]
        _, lam = nlp.constraint("c1", x**2 + y**2, "==", 1)
        for f in [-x - y, (x + y) ** 2]:
            nlp.minimize(f)
            L = f + lam * (x**2 + y**2 - 1)
            x_ = np.random.randn(*x.shape)
            y_ = np.random.randn(*y.shape)
            lam_ = np.random.randn(*lam.shape)
            np.testing.assert_allclose(
                subsevalf(nlp.lagrangian, [x, y, lam], [x_, y_, lam_]),
                subsevalf(L, [x, y, lam], [x_, y_, lam_]),
            )

    def test_lagrangian__is_correct__example_3(self):
        # https://en.wikipedia.org/wiki/Lagrange_multiplier#Example_3
        n = 50
        lbx = 1 / n * 1e-6
        nlp = NlpSensitivity(Nlp(sym_type=self.sym_type))
        p, lam_lbx, _ = nlp.variable("p", (n, 1), lb=lbx)
        c1, lam_g = nlp.constraint("c1", cs.sum1(p), "==", 1)
        f = cs.sum1(p * log(p, 2))
        nlp.minimize(f)
        L = f + lam_g * c1 + lam_lbx.T @ (lbx - p)
        p_ = np.random.rand(*p.shape)
        lam_lbx_ = np.random.randn(*lam_lbx.shape)
        lam_g_ = np.random.randn(*lam_g.shape)
        np.testing.assert_allclose(
            subsevalf(nlp.lagrangian, [p, lam_lbx, lam_g], [p_, lam_lbx_, lam_g_]),
            subsevalf(L, [p, lam_lbx, lam_g], [p_, lam_lbx_, lam_g_]),
        )

    def test_lagrangian__is_correct__example_5(self):
        # https://personal.math.ubc.ca/~israel/m340/kkt2.pdf
        nlp = NlpSensitivity(Nlp(sym_type=self.sym_type))
        x, lam_lbx, lam_ubx = nlp.variable("x", (2, 1), lb=0, ub=1)
        c1, lam_h = nlp.constraint("c1", x[0] + x[1] ** 2, "<=", 2)
        f = -x[0] * x[1]
        nlp.minimize(f)
        L = f + lam_h * c1 + lam_lbx.T @ (0 - x) + lam_ubx.T @ (x - 1)
        x_ = np.random.rand(*x.shape)
        lam_lbx_ = np.random.randn(*lam_lbx.shape)
        lam_ubx_ = np.random.randn(*lam_ubx.shape)
        lam_h_ = np.random.randn(*lam_h.shape)
        np.testing.assert_allclose(
            subsevalf(
                nlp.lagrangian,
                [x, lam_lbx, lam_ubx, lam_h],
                [x_, lam_lbx_, lam_ubx_, lam_h_],
            ),
            subsevalf(
                L, [x, lam_lbx, lam_ubx, lam_h], [x_, lam_lbx_, lam_ubx_, lam_h_]
            ),
        )

    @parameterized.expand([(False,), (True,)])
    def test_kkt__returns_tau_correctly(self, flag: bool):
        nlp = NlpSensitivity(Nlp(sym_type=self.sym_type), include_barrier_term=flag)
        x = nlp.variable("x")[0]
        nlp.constraint("c", x, "<=", 1)
        nlp.minimize(x)
        _, tau = nlp.kkt
        if flag:
            self.assertIsInstance(tau, nlp.unwrapped.sym_type)
        else:
            self.assertIsNone(tau)

    def test_kkt__computes_kkt_conditions_correctly__example_1a(self):
        # https://en.wikipedia.org/wiki/Lagrange_multiplier#Example_1
        nlp = NlpSensitivity(Nlp(sym_type=self.sym_type))
        x = nlp.variable("x")[0]
        y = nlp.variable("y")[0]
        nlp.constraint("c1", x**2 + y**2, "==", 1)
        nlp.minimize(-x - y)
        nlp.init_solver(OPTS)
        sol = nlp.solve()
        kkt, _ = nlp.kkt
        np.testing.assert_allclose(sol.value(kkt), 0, atol=1e-9)

    def test_kkt__computes_kkt_conditions_correctly__example_1b(self):
        # https://en.wikipedia.org/wiki/Lagrange_multiplier#Example_1
        nlp = NlpSensitivity(Nlp(sym_type=self.sym_type))
        x = nlp.variable("x")[0]
        y = nlp.variable("y")[0]
        nlp.constraint("c1", x**2 + y**2, "==", 1)
        nlp.minimize((x + y) ** 2)
        nlp.init_solver(OPTS)
        sol = nlp.solve(vals0={"x": 0.5, "y": np.sqrt(1 - 0.5**2)})
        kkt, _ = nlp.kkt
        np.testing.assert_allclose(sol.value(kkt), 0, atol=1e-6)

    def test_kkt__computes_kkt_conditions_correctly__example_2(self):
        # https://en.wikipedia.org/wiki/Lagrange_multiplier#Example_2
        nlp = NlpSensitivity(Nlp(sym_type=self.sym_type))
        x = nlp.variable("x")[0]
        y = nlp.variable("y")[0]
        nlp.constraint("c1", x**2 + y**2, "==", 3)
        nlp.minimize(x**2 * y)
        nlp.init_solver(OPTS)
        sol = nlp.solve(vals0={"x": np.sqrt(3 - 0.8**2), "y": -0.8})
        kkt, _ = nlp.kkt
        np.testing.assert_allclose(sol.value(kkt), 0, atol=1e-6)

    def test_kkt__computes_kkt_conditions_correctly__example_3(self):
        # https://en.wikipedia.org/wiki/Lagrange_multiplier#Example_3
        n = 50
        nlp = NlpSensitivity(Nlp(sym_type=self.sym_type))
        p = nlp.variable("p", (n, 1), lb=1 / n * 1e-6)[0]
        nlp.constraint("c1", cs.sum1(p), "==", 1)
        nlp.minimize(cs.sum1(p * log(p, 2)))
        nlp.init_solver(OPTS)
        sol = nlp.solve(vals0={"p": np.random.rand(n)})
        kkt, tau = nlp.kkt
        kkt = subsevalf(kkt, tau, sol.barrier_parameter, eval=False)
        np.testing.assert_allclose(sol.value(kkt), 0, atol=1e-6)

    def test_kkt__computes_kkt_conditions_correctly__example_4(self):
        # https://en.wikipedia.org/wiki/Lagrange_multiplier#Example_4
        nlp = NlpSensitivity(Nlp(sym_type=self.sym_type))
        x = nlp.variable("x")[0]
        nlp.constraint("c1", x**2, "==", 1)
        nlp.minimize(x**2)
        nlp.init_solver(OPTS)
        sol = nlp.solve(vals0={"x": 1 + 0.1 * np.random.rand()})
        kkt, _ = nlp.kkt
        np.testing.assert_allclose(sol.value(kkt), 0, atol=1e-6)

    def test_kkt__computes_kkt_conditions_correctly__example_5(self):
        # https://personal.math.ubc.ca/~israel/m340/kkt2.pdf
        nlp = NlpSensitivity(Nlp(sym_type=self.sym_type))
        x = nlp.variable("x", (2, 1), lb=0)[0]
        nlp.constraint("c1", x[0] + x[1] ** 2, "<=", 2)
        nlp.minimize(-x[0] * x[1])
        nlp.init_solver(OPTS)
        sol = nlp.solve(vals0={"x": [1, 1]})
        kkt, tau = nlp.kkt
        kkt = subsevalf(kkt, tau, sol.barrier_parameter, eval=False)
        np.testing.assert_allclose(sol.value(kkt), 0, atol=1e-7)

    def test_kkt__computes_sensitivity_correctly__example_7(self):
        #  Example 4.2 from [1]
        #
        # References
        # ----------
        # [1] Buskens, C. and Maurer, H. (2001). Sensitivity analysis and
        #     real-time optimization of parametric nonlinear programming
        #     problems. In M. Grotschel, S.O. Krumke, and J. Rambau (eds.),
        #     Online Optimization of Large Scale Systems, 3–16. Springer,
        #     Berlin, Heidelberg.
        nlp = NlpSensitivity(Nlp(sym_type=self.sym_type))
        z = nlp.variable("z", (2, 1))[0]
        p = nlp.parameter("p")
        nlp.minimize(-((0.5 + p) * cs.sqrt(z[0]) + (0.5 - p) * z[1]))
        _, lam = nlp.constraint("c1", cs.sum1(z), "<=", 1)
        nlp.constraint("c2", z[0], ">=", 0.1)
        nlp.init_solver(OPTS)
        sol = nlp.solve(pars={"p": 0}, vals0={"z": [0.1, 0.9]})

        np.testing.assert_allclose(sol.vals["z"], [[0.25], [0.75]], rtol=1e-5)
        np.testing.assert_allclose(sol.value(lam), 0.5)

        kkt, tau = nlp.kkt
        kkt = subsevalf(kkt, tau, sol.barrier_parameter, eval=False)
        np.testing.assert_allclose(sol.value(kkt), 0, atol=1e-7)

        S1 = sol.value(nlp.parametric_sensitivity()[0]).full().flatten()
        S2 = nlp.parametric_sensitivity(solution=sol)[0].flatten()
        for S in (S1, S2):
            np.testing.assert_allclose(S, [2, -2, -1, 0], atol=1e-5)

    def test_kkt__computes_sensitivity_correctly__example_8(self):
        #  Example 4.5 from [1]
        #
        # References
        # ----------
        # [1] Buskens, C. and Maurer, H. (2001). Sensitivity analysis and
        #     real-time optimization of parametric nonlinear programming
        #     problems. In M. Grotschel, S.O. Krumke, and J. Rambau (eds.),
        #     Online Optimization of Large Scale Systems, 3–16. Springer,
        #     Berlin, Heidelberg.
        nlp = NlpSensitivity(Nlp(sym_type=self.sym_type))
        z = nlp.variable("z", (2, 1))[0]
        p = nlp.parameter("p")
        nlp.minimize(cs.sumsqr(z + [1, -2]))
        _, lam1 = nlp.constraint("c1", -z[0] + p, "<=", 0)
        _, lam2 = nlp.constraint("c2", 2 * z[0] + z[1], "<=", 6)
        nlp.init_solver(OPTS)
        sol = nlp.solve(pars={"p": 1})

        np.testing.assert_allclose(sol.f, 4)
        np.testing.assert_allclose(sol.vals["z"], [[1], [2]], rtol=1e-7)
        np.testing.assert_allclose(sol.value(lam1), 4, atol=1e-7)
        np.testing.assert_allclose(sol.value(lam2), 0, atol=1e-7)

        kkt, tau = nlp.kkt
        kkt = subsevalf(kkt, tau, sol.barrier_parameter, eval=False)
        np.testing.assert_allclose(sol.value(kkt), 0, atol=1e-7)

        S1 = nlp.parametric_sensitivity()[0]
        S2 = nlp.parametric_sensitivity(solution=sol)[0]
        np.testing.assert_allclose(sol.value(S1).full().flat, [1, 0, 2, 0], atol=1e-5)
        np.testing.assert_allclose(S2.flat, [1, 0, 2, 0], atol=1e-5)

        Fp1, Fpp1 = (
            sol.value(o)
            for o in nlp.parametric_sensitivity(expr=nlp.f, second_order=True)
        )
        Fp2, Fpp2 = nlp.parametric_sensitivity(
            expr=nlp.f, solution=sol, second_order=True
        )
        np.testing.assert_allclose(Fp1, 4, atol=1e-7)
        np.testing.assert_allclose(Fp2, 4, atol=1e-7)
        np.testing.assert_allclose(Fpp1, 2, atol=1e-7)
        np.testing.assert_allclose(Fpp2, 2, atol=1e-7)

    def test_licq__computes_qualification_correctly__example_1(self):
        # https://de.wikipedia.org/wiki/Linear_independence_constraint_qualification#LICQ
        nlp = NlpSensitivity(Nlp(sym_type=self.sym_type))
        x = nlp.variable("x", (2, 1))[0]
        nlp.constraint("c1", cs.sum1(x), "<=", 1)
        nlp.constraint("c2", cs.sumsqr(x), "<=", 1)
        nlp.minimize(x[0] ** 2 + (x[1] - 1) ** 2)  # any objective

        x_ = cs.DM([[0], [1]])
        d = subsevalf(nlp.licq, x, x_).full()
        np.testing.assert_allclose(d, [[1, 1], [0, 2]])
        self.assertEqual(np.linalg.matrix_rank(d), d.shape[0])

    def test_licq__computes_qualification_correctly__example_2(self):
        # https://de.wikipedia.org/wiki/Linear_independence_constraint_qualification#MFCQ_ohne_LICQ
        nlp = NlpSensitivity(Nlp(sym_type=self.sym_type))
        x = nlp.variable("x", (2, 1))[0]
        nlp.constraint("c1", x[1], ">=", 0)
        nlp.constraint("c2", x[0] ** 4 - x[1], "<=", 1)
        nlp.minimize(x[0] ** 2 + x[1] ** 2)  # any objective

        x_ = cs.DM([[0], [0]])
        d = subsevalf(nlp.licq, x, x_).full()
        np.testing.assert_allclose(d, [[0, -1], [0, -1]])
        self.assertNotEqual(np.linalg.matrix_rank(d), d.shape[0])

    def test_parametric_sensitivity__is_correct(self):
        # https://web.casadi.org/blog/nlp_sens/
        p_values_and_solutions = [
            (1.25, (-0.442401495488, 0.400036517837, 3.516850030791)),
            (1.4, (-0.350853791920, 0.766082818877, 1.401822559490)),
            (2, (-0.000000039328, 0, 0)),
        ]

        def z(x):
            return x[1, :] - x[0, :]

        nlp = NlpSensitivity(Nlp(sym_type=self.sym_type))
        x = nlp.variable("x", (2, 1), lb=[[0], [-np.inf]])[0]
        p = nlp.parameter("p", (2, 1))
        nlp.minimize((1 - x[0]) ** 2 + p[0] * (x[1] - x[0] ** 2) ** 2)
        g = (x[0] + 0.5) ** 2 + x[1] ** 2
        nlp.constraint("c1", (p[1] / 2) ** 2, "<=", g)
        nlp.constraint("c2", g, "<=", p[1] ** 2)
        nlp.init_solver(OPTS)

        Z1_ = z(x)
        J1_, H1_ = nlp.parametric_sensitivity(expr=Z1_, second_order=True)
        for p, (Z, J, H) in p_values_and_solutions:
            sol = nlp.solve(pars={"p": [0.2, p]})
            Z1 = sol.value(Z1_)
            J1 = sol.value(J1_)
            H1 = sol.value(H1_)
            J2, H2 = nlp.parametric_sensitivity(
                expr=Z1_, solution=sol, second_order=True
            )
            J2, H2 = np.squeeze(J2), np.squeeze(H2)
            np.testing.assert_allclose(J1.full().flat, J2, atol=1e-7)
            np.testing.assert_allclose(H1.full(), H2, atol=1e-7)
            np.testing.assert_allclose(Z, Z1, atol=1e-4)
            np.testing.assert_allclose(J, J1[1], atol=1e-4)
            np.testing.assert_allclose(J, J2[1], atol=1e-4)
            np.testing.assert_allclose(H, H1[1, 1], atol=1e-4)
            np.testing.assert_allclose(H, H2[1, 1], atol=1e-4)

    def test_parametric_sensitivity__with_specific_pars__is_correct(self):
        # https://web.casadi.org/blog/nlp_sens/
        p_values_and_solutions = [
            (1.25, (-0.442401495488, 0.400036517837, 3.516850030791)),
            (1.4, (-0.350853791920, 0.766082818877, 1.401822559490)),
            (2, (-0.000000039328, 0, 0)),
        ]

        def z(x):
            return x[1, :] - x[0, :]

        nlp = Nlp(sym_type=self.sym_type)
        x = nlp.variable("x", (2, 1), lb=[[0], [-np.inf]])[0]
        p = nlp.parameter("p", (2, 1))
        p2 = nlp.parameter("p2", (1, 1))
        nlp.minimize((1 - x[0]) ** 2 + p[0] * (x[1] - x[0] ** 2) ** 2)
        g = (x[0] + 0.5) ** 2 + x[1] ** 2 + p2
        nlp.constraint("c1", (p[1] / 2) ** 2, "<=", g)
        nlp.constraint("c2", g, "<=", p[1] ** 2)
        nlp.init_solver(OPTS)

        sensitivity_all = NlpSensitivity(nlp)
        sensitivity_partial = NlpSensitivity(nlp, target_parameters=p[1])

        kwargs = {"expr": z(x), "second_order": True}
        J1_, H1_ = sensitivity_all.parametric_sensitivity(**kwargs)
        J2_, H2_ = sensitivity_partial.parametric_sensitivity(**kwargs)
        for p, (_, J_exp, H_exp) in p_values_and_solutions:
            sol = nlp.solve(pars={"p": [0.2, p], "p2": 0})
            J1_act = sol.value(J1_)
            H1_act = sol.value(H1_)
            J2_act = sol.value(J2_)
            H2_act = sol.value(H2_)
            np.testing.assert_allclose(J_exp, J1_act[1], atol=1e-4)
            np.testing.assert_allclose(H_exp, H1_act[1, 1], atol=1e-4)
            np.testing.assert_allclose(J_exp, J2_act, atol=1e-4)
            np.testing.assert_allclose(H_exp, H2_act, atol=1e-4)

    def test_can_be_pickled(self):
        nlp = NlpSensitivity(Nlp(sym_type=self.sym_type))
        x = nlp.variable("x", (2, 1), lb=[[0], [-np.inf]])[0]
        p = nlp.parameter("p", (2, 1))
        nlp.minimize((1 - x[0]) ** 2 + p[0] * (x[1] - x[0] ** 2) ** 2)
        g = (x[0] + 0.5) ** 2 + x[1] ** 2
        nlp.constraint("c1", (p[1] / 2) ** 2, "<=", g)
        nlp.constraint("c2", g, "<=", p[1] ** 2)
        nlp.init_solver(OPTS)

        with cs.global_pickle_context():
            pickled = pickle.dumps(nlp)
        with cs.global_unpickle_context():
            other_nlp = pickle.loads(pickled)

        self.assertIn(repr(nlp), repr(other_nlp))


class TestNlpScaling(unittest.TestCase):
    @parameterized.expand([("variable",), ("parameter",)])
    def test_parameter_and_variable__warns__when_cannot_scale(self, method: str):
        scaler = scaling.Scaler()
        nlp = NlpScaling[cs.SX](Nlp(sym_type="SX"), scaler=scaler, warns=True)
        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            with self.assertRaisesRegex(
                RuntimeWarning, f"Scaling for {method} p not found."
            ):
                getattr(nlp, method)("p")

    def test_parameter__scales_correctly(self):
        scaler = scaling.Scaler({"p": (0, 2)})
        nlp = NlpScaling[cs.SX](Nlp(sym_type="SX"), scaler=scaler, warns=True)
        p = nlp.parameter("p")
        self.assertIn("p", nlp.scaled_parameters)
        self.assertIn("p", nlp.unscaled_parameters)
        p_s = nlp.scale(p)
        p_u = nlp.unscale(p)
        np.testing.assert_allclose(cs.evalf(nlp.scaled_parameters["p"] - p_s), 0)
        np.testing.assert_allclose(cs.evalf(nlp.unscaled_parameters["p"] - p_u), 0)

    def test_variable__scales_correctly(self):
        scaler = scaling.Scaler({"x": (0, 2)})
        nlp = NlpScaling[cs.SX](Nlp(sym_type="SX"), scaler=scaler, warns=True)
        lb, ub = -5, 4
        x, _, _ = nlp.variable("x", lb=lb, ub=ub)
        self.assertIn("x", nlp.scaled_variables)
        self.assertIn("x", nlp.unscaled_variables)
        x_s = nlp.scale(x)
        x_u = nlp.unscale(x)
        np.testing.assert_allclose(cs.evalf(nlp.scaled_variables["x"] - x_s), 0)
        np.testing.assert_allclose(cs.evalf(nlp.unscaled_variables["x"] - x_u), 0)
        np.testing.assert_allclose(nlp.lbx, scaler.scale("x", lb))
        np.testing.assert_allclose(nlp.ubx, scaler.scale("x", ub))


if __name__ == "__main__":
    unittest.main()
