import pickle
import unittest
from typing import List, Union

import casadi as cs
import numpy as np
from parameterized import parameterized, parameterized_class

from csnlp import Nlp
from csnlp.nlp.solutions import subsevalf
from csnlp.util.math import log

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


@parameterized_class(["sym_type"], [("SX",), ("MX",)])
class TestNlp(unittest.TestCase):
    def cmp(
        self,
        lhs: Union[cs.SX, cs.MX],
        rhs: Union[cs.SX, cs.MX],
        vars: Union[List[cs.SX], List[cs.MX]],
        force_numerical: bool = False,
    ):
        if isinstance(rhs, (int, float)):
            rhs = np.full(lhs.shape, rhs)
        if not force_numerical and self.sym_type == "SX":
            return cs.is_equal(lhs, rhs)
        old = cs.vertcat(*(cs.vec(o) for o in vars))
        new = np.random.randn(*old.shape)
        np.testing.assert_allclose(subsevalf(lhs, old, new), subsevalf(rhs, old, new))

    def test_init__raises__with_invalid_sym_type(self):
        with self.assertRaises(AttributeError):
            Nlp(sym_type="a_random_sym_type")

    def test_init__saves_sym_type(self):
        nlp = Nlp(sym_type=self.sym_type)
        self.assertEqual(nlp._csXX.__name__, self.sym_type)

    def test_np_random__creates_rng_when_fetched(self):
        nlp = Nlp(sym_type=self.sym_type)
        rng = nlp.np_random
        self.assertIsNotNone(rng)
        self.assertIsInstance(rng, np.random.Generator)

    def test_parameter__creates_correct_parameter(self):
        shape1 = (4, 3)
        shape2 = (2, 2)
        _np = np.prod(shape1) + np.prod(shape2)
        nlp = Nlp(sym_type=self.sym_type, debug=True)
        p1 = nlp.parameter("p1", shape1)
        p2 = nlp.parameter("p2", shape2)
        self.assertEqual(p1.shape, shape1)
        self.assertEqual(p2.shape, shape2)
        self.assertEqual(nlp.np, _np)

        p = cs.vertcat(cs.vec(p1), cs.vec(p2))
        self.cmp(nlp.p, p, vars=[p1, p2])

        i = 0
        for name, shape in [("p1", shape1), ("p2", shape2)]:
            for _ in range(np.prod(shape)):
                self.assertEqual(name, nlp.debug.p_describe(i).name)
                self.assertEqual(shape, nlp.debug.p_describe(i).shape)
                i += 1
        with self.assertRaises(IndexError):
            nlp.debug.p_describe(_np + 1)

        self.assertTrue(cs.is_equal(nlp.parameters["p1"], p1))
        self.assertTrue(cs.is_equal(nlp.parameters["p2"], p2))

    def test_parameter__raises__with_parameters_with_same_name(self):
        nlp = Nlp(sym_type=self.sym_type)
        nlp.parameter("p")
        with self.assertRaises(ValueError):
            nlp.parameter("p")

    def test_variable__creates_correct_variable(self):
        shape1 = (4, 3)
        shape2 = (2, 2)
        lb1, ub1 = np.random.rand(*shape1) - 1, np.random.rand(*shape1) + 1
        lb2, ub2 = np.random.rand(*shape2) - 1, np.random.rand(*shape2) + 1
        nx = np.prod(shape1) + np.prod(shape2)
        nlp = Nlp(sym_type=self.sym_type, debug=True)
        x1, lam1_lb, lam1_ub = nlp.variable("x1", shape1, lb=lb1, ub=ub1)
        x2, lam2_lb, lam2_ub = nlp.variable("x2", shape2, lb=lb2, ub=ub2)
        for o in (x1, lam1_lb, lam1_ub):
            self.assertEqual(o.shape, shape1)
        for o in (x2, lam2_lb, lam2_ub):
            self.assertEqual(o.shape, shape2)
        self.assertEqual(nlp.nx, nx)

        x = cs.vertcat(cs.vec(x1), cs.vec(x2))
        self.cmp(nlp.x, x, vars=[x1, x2])

        lb = cs.vertcat(cs.vec(lb1), cs.vec(lb2))
        ub = cs.vertcat(cs.vec(ub1), cs.vec(ub2))
        np.testing.assert_allclose(nlp.lbx, lb.full().flat)
        np.testing.assert_allclose(nlp.ubx, ub.full().flat)

        i = 0
        for name, shape in [("x1", shape1), ("x2", shape2)]:
            for _ in range(np.prod(shape)):
                self.assertEqual(name, nlp.debug.x_describe(i).name)
                self.assertEqual(shape, nlp.debug.x_describe(i).shape)
                i += 1
        with self.assertRaises(IndexError):
            nlp.debug.x_describe(nx + 1)

        self.assertTrue(cs.is_equal(nlp.variables["x1"], x1))
        self.assertTrue(cs.is_equal(nlp.variables["x2"], x2))

    def test_variable__raises__with_variables_with_same_name(self):
        nlp = Nlp(sym_type=self.sym_type)
        nlp.variable("x")
        with self.assertRaises(ValueError):
            nlp.variable("x")

    def test_variable__raises__with_invalid_bounds(self):
        nlp = Nlp(sym_type=self.sym_type)
        with self.assertRaises(ValueError):
            nlp.variable("x", lb=1, ub=0)

    def test_minimize__raises__with_nonscalar_objective(self):
        nlp = Nlp(sym_type=self.sym_type)
        x = nlp.variable("x", (5, 1))[0]
        with self.assertRaises(ValueError):
            nlp.minimize(x)

    def test_minimize__sets_objective_correctly(self):
        nlp = Nlp(sym_type=self.sym_type)
        x = nlp.variable("x", (5, 1))[0]
        f = x.T @ x
        nlp.minimize(f)
        self.assertTrue(cs.is_equal(nlp.f, f))

    def test_constraint__raises__with_constraints_with_same_name(self):
        nlp = Nlp(sym_type=self.sym_type)
        x = nlp.variable("x")[0]
        nlp.constraint("c1", x, "<=", 5)
        with self.assertRaises(ValueError):
            nlp.constraint("c1", x, "<=", 5)

    def test_constraint__raises__with_unknown_operator(self):
        nlp = Nlp(sym_type=self.sym_type)
        x = nlp.variable("x")[0]
        for op in ["=", ">", "<"]:
            with self.assertRaises(ValueError):
                nlp.constraint("c1", x, op, 5)

    def test_constraint__raises__with_nonsymbolic_terms(self):
        nlp = Nlp(sym_type=self.sym_type)
        with self.assertRaises(TypeError):
            nlp.constraint("c1", 5, "==", 5)

    @parameterized.expand([("==", "=="), ("==", ">="), (">=", ">=")])
    def test_constraint__creates_constraint_correctly(self, op1: str, op2: str):
        shape1, shape2 = (4, 3), (2, 2)
        nc = np.prod(shape1) + np.prod(shape2)
        nlp = Nlp(sym_type=self.sym_type, debug=True)
        x = nlp.variable("x", shape1)[0]
        y = nlp.variable("y", shape2)[0]
        c1, lam_c1 = nlp.constraint("c1", x, op1, 5)
        c2, lam_c2 = nlp.constraint("c2", 5, op2, y)

        self.assertTrue(c1.shape == lam_c1.shape == shape1)
        self.assertTrue(c2.shape == lam_c2.shape == shape2)
        self.assertEqual(nlp.ng + nlp.nh, nc)

        i, prev_op = 0, op1
        for c, name, op in zip((c1, c2), ("c1", "c2"), (op1, op2)):
            grp = "g" if op == "==" else "h"
            if op != prev_op:
                i = 0
            describe = getattr(nlp.debug, f"{grp}_describe")
            for _ in range(np.prod(c.shape)):
                self.assertEqual(name, describe(i).name)
                self.assertEqual(c.shape, describe(i).shape)
                i += 1
            with self.assertRaises(IndexError):
                describe(nc + 1)

        c = cs.vertcat(cs.vec(c1), cs.vec(c2))
        lam = cs.vertcat(cs.vec(lam_c1), cs.vec(lam_c2))
        expected_c = cs.vertcat(nlp.g, nlp.h)
        expected_lam = cs.vertcat(nlp.lam_g, nlp.lam_h)
        self.cmp(expected_c, c, vars=[x, y])
        self.cmp(expected_lam, lam, vars=[lam_c1, lam_c2])

    def test_constraint__adds_soft_variable_correctly(self):
        nlp = Nlp(sym_type=self.sym_type)
        x = nlp.variable("x")[0]
        nlp.constraint("c0", x, ">=", 0, soft=True)
        self.assertEqual(nlp.nx, 2)
        self.assertIn("slack_c0", nlp._vars)

    def test_constraint__solves_correctly__with_soft_variable(self):
        # From https://www.gams.com/latest/docs/UG_EMP_SoftConstraints.html
        # solve with manual soft variable
        nlp = Nlp(sym_type=self.sym_type)
        x = nlp.variable("x", (2, 1), lb=0)[0]
        v = nlp.variable("v", lb=0)[0]
        nlp.constraint("c0", 3 * x[0] + x[1], "<=", 5)
        nlp.constraint("c1", v, ">=", 20 - x[1] ** 2)
        nlp.minimize(-x[0] ** 2 + 5 * (cs.log(x[0]) - 1) ** 2 + 2 * v)
        nlp.init_solver(OPTS)
        sol1 = nlp.solve(vals0={"x": [1, 1], "v": 0})

        # solve with automatic soft variable
        nlp = Nlp(sym_type=self.sym_type)
        x = nlp.variable("x", (2, 1), lb=0)[0]
        nlp.constraint("c0", 3 * x[0] + x[1], "<=", 5)
        _, _, v = nlp.constraint("c1", x[1] ** 2, ">=", 20, soft=True)
        nlp.minimize(-x[0] ** 2 + 5 * (cs.log(x[0]) - 1) ** 2 + 2 * v)
        nlp.init_solver(OPTS)
        sol2 = nlp.solve(vals0={"x": [1, 1], "slack_c1": 0})

        # assert results equal
        np.testing.assert_allclose(sol1.f, sol2.f)
        np.testing.assert_allclose(sol1.vals["x"], sol2.vals["x"])
        np.testing.assert_allclose(sol1.vals["v"], sol2.vals["slack_c1"])

    @parameterized.expand(
        [
            (True, "==", "=="),
            (True, "==", ">="),
            (True, ">=", ">="),
            (False, "==", "=="),
            (False, "==", ">="),
            (False, ">=", ">="),
        ]
    )
    def test_dual__returns_dual_variables_correctly(
        self, flag: bool, op1: str, op2: str
    ):
        shape1, shape2 = (4, 3), (2, 2)
        nlp = Nlp(sym_type=self.sym_type, remove_redundant_x_bounds=flag)
        x, lam_lb_x, lam_ub_x = nlp.variable("x", shape1)
        y, lam_lb_y, lam_ub_y = nlp.variable("y", shape2, ub=10)
        _, lam_c1 = nlp.constraint("c1", x, op1, 5)
        _, lam_c2 = nlp.constraint("c2", 5, op2, y)

        dv = nlp.dual_variables
        c1_name = f'lam_{"g" if op1 == "==" else "h"}_c1'
        c2_name = f'lam_{"g" if op2 == "==" else "h"}_c2'
        self.assertTrue(cs.is_equal(dv["lam_lb_x"], lam_lb_x))
        self.assertTrue(cs.is_equal(dv["lam_ub_x"], lam_ub_x))
        self.assertTrue(cs.is_equal(dv["lam_lb_y"], lam_lb_y))
        self.assertTrue(cs.is_equal(dv["lam_ub_y"], lam_ub_y))
        self.assertTrue(cs.is_equal(dv[c1_name], lam_c1))
        self.assertTrue(cs.is_equal(dv[c2_name], lam_c2))

        if flag:
            lams = [lam_c1, lam_c2, lam_ub_y]
        else:
            lams = [lam_c1, lam_c2, lam_lb_x, lam_lb_y, lam_ub_x, lam_ub_y]
        actual_lam = cs.vertcat(*(cs.vec(o) for o in lams))
        expected_lam = nlp.lam
        self.cmp(actual_lam, expected_lam, vars=lams)

    def test_primal_dual_variables__returns_correctly(self):
        nlp = Nlp(sym_type=self.sym_type)
        x, lam_lbx, lam_ubx = nlp.variable("x", (2, 3), lb=[[0], [-np.inf]], ub=1)
        _, lam_g = nlp.constraint("c1", x[:, 0], "==", 2)
        _, lam_h = nlp.constraint("c2", x[0, :] + x[1, :] ** 2, "<=", 2)
        v = cs.vec
        vc = cs.vertcat

        lam = vc(v(lam_g), v(lam_h), v(lam_lbx[0, :]), v(lam_ubx))
        lam_all = vc(v(lam_g), v(lam_h), v(lam_lbx), v(lam_ubx))
        y = vc(v(x), v(lam_g), v(lam_h), v(lam_lbx[0, :]), v(lam_ubx))
        y_all = vc(v(x), v(lam_g), v(lam_h), v(lam_lbx), v(lam_ubx))

        vars = [x, lam_g, lam_h, lam_lbx, lam_ubx]
        self.cmp(nlp.lam, lam, vars=vars)
        self.cmp(nlp.lam_all, lam_all, vars=vars)
        self.cmp(nlp.primal_dual_vars(all=False), y, vars=vars)
        self.cmp(nlp.primal_dual_vars(all=True), y_all, vars=vars)

    @parameterized.expand([(True,), (False,)])
    def test_h_lbx_ubx__returns_correct_indices(self, flag: bool):
        nlp = Nlp(sym_type=self.sym_type, remove_redundant_x_bounds=flag)

        x1, lam_lbx1, lam_ubx1 = nlp.variable("x1", (2, 1))
        (h_lbx, lam_lbx), (h_ubx, lam_ubx) = nlp.h_lbx, nlp.h_ubx
        vars = [x1, lam_lbx1, lam_ubx1]
        if flag:
            self.assertTrue(all(o.is_empty() for o in [h_lbx, lam_lbx, h_ubx, lam_ubx]))
        else:
            if self.sym_type == "SX":
                self.cmp((h_lbx - (-np.inf - x1)), 0, vars=vars, force_numerical=True)
                self.cmp((h_ubx - (x1 - np.inf)), 0, vars=vars, force_numerical=True)
            self.cmp(cs.evalf(lam_lbx - lam_lbx1), 0, vars=vars)
            self.cmp(cs.evalf(lam_ubx - lam_ubx1), 0, vars=vars)

        x2, lam_lbx2, lam_ubx2 = nlp.variable("x2", (2, 1), lb=0)
        (h_lbx, lam_lbx), (h_ubx, lam_ubx) = nlp.h_lbx, nlp.h_ubx
        vars.extend([x2, lam_lbx2, lam_ubx2])
        if flag:
            self.assertTrue(all(o.is_empty() for o in [h_ubx, lam_ubx]))
            self.cmp(cs.evalf(h_lbx - (-x2)), 0, vars=vars)
            self.cmp(cs.evalf(lam_lbx - lam_lbx2), 0, vars=vars)
        else:
            if self.sym_type == "SX":
                self.cmp(
                    cs.evalf(h_lbx - cs.vertcat(-np.inf - x1, 0 - x2)),
                    0,
                    vars=vars,
                )
                self.cmp(
                    cs.evalf(h_ubx - cs.vertcat(x1 - np.inf, x2 - np.inf)),
                    0,
                    vars=vars,
                )
            self.cmp(
                (lam_lbx - cs.vertcat(lam_lbx1, lam_lbx2)),
                0,
                vars=vars,
                force_numerical=True,
            )
            self.cmp(
                (lam_ubx - cs.vertcat(lam_ubx1, lam_ubx2)),
                0,
                vars=vars,
            )

        x3, lam_lbx3, lam_ubx3 = nlp.variable("x3", (2, 1), ub=1)
        (h_lbx, lam_lbx), (h_ubx, lam_ubx) = nlp.h_lbx, nlp.h_ubx
        vars.extend([x3, lam_lbx3, lam_ubx3])
        if flag:
            self.cmp(cs.evalf(h_lbx - (-x2)), 0, vars=vars)
            self.cmp(cs.evalf(lam_lbx - lam_lbx2), 0, vars=vars)
            self.cmp((h_ubx - (x3 - 1)), 0, vars=vars, force_numerical=True)
            self.cmp(cs.evalf(lam_ubx - lam_ubx3), 0, vars=vars)
        else:
            if self.sym_type == "SX":
                self.cmp(
                    cs.evalf(h_lbx - cs.vertcat(-np.inf - x1, 0 - x2, -np.inf - x3)),
                    0,
                    vars=vars,
                )
                self.cmp(
                    cs.evalf(h_ubx - cs.vertcat(x1 - np.inf, x2 - np.inf, x3 - 1)),
                    0,
                    vars=vars,
                )
            self.cmp(
                (lam_lbx - cs.vertcat(lam_lbx1, lam_lbx2, lam_lbx3)),
                0,
                vars=vars,
                force_numerical=True,
            )
            self.cmp(
                (lam_ubx - cs.vertcat(lam_ubx1, lam_ubx2, lam_ubx3)),
                0,
                vars=vars,
                force_numerical=True,
            )

        x4, lam_lbx4, lam_ubx4 = nlp.variable("x4", (2, 1), lb=0, ub=1)
        (h_lbx, lam_lbx), (h_ubx, lam_ubx) = nlp.h_lbx, nlp.h_ubx
        vars.extend([x4, lam_lbx4, lam_ubx4])
        if flag:
            self.cmp((h_lbx - cs.vertcat(-x2, -x4)), 0, vars=vars, force_numerical=True)
            self.cmp(
                (lam_lbx - cs.vertcat(lam_lbx2, lam_lbx4)),
                0,
                vars=vars,
                force_numerical=True,
            )
            self.cmp(
                (h_ubx - cs.vertcat(x3 - 1, x4 - 1)), 0, vars=vars, force_numerical=True
            )
            self.cmp(
                (lam_ubx - cs.vertcat(lam_ubx3, lam_ubx4)),
                0,
                vars=vars,
                force_numerical=True,
            )
        else:
            if self.sym_type == "SX":
                self.cmp(
                    cs.evalf(
                        h_lbx - cs.vertcat(-np.inf - x1, 0 - x2, -np.inf - x3, 0 - x4)
                    ),
                    0,
                    vars=vars,
                )
                self.cmp(
                    cs.evalf(
                        h_ubx - cs.vertcat(x1 - np.inf, x2 - np.inf, x3 - 1, x4 - 1)
                    ),
                    0,
                    vars=vars,
                )
            self.cmp(
                (lam_lbx - cs.vertcat(lam_lbx1, lam_lbx2, lam_lbx3, lam_lbx4)),
                0,
                vars=vars,
                force_numerical=True,
            )
            self.cmp(
                (lam_ubx - cs.vertcat(lam_ubx1, lam_ubx2, lam_ubx3, lam_ubx4)),
                0,
                vars=vars,
                force_numerical=True,
            )

    def test_init_solver__raises__when_objective_not_set(self):
        nlp = Nlp(sym_type=self.sym_type)
        with self.assertRaises(RuntimeError):
            nlp.init_solver(OPTS)

    def test_init_solver__saves_options_correctly(self):
        nlp = Nlp(sym_type=self.sym_type)
        x = nlp.variable("x")[0]
        nlp.minimize(x**2)
        nlp.init_solver(OPTS)
        self.assertDictEqual(OPTS, nlp.solver_opts)

    def test_solve__raises__with_uninit_solver(self):
        nlp = Nlp(sym_type=self.sym_type)
        with self.assertRaisesRegex(RuntimeError, "Solver uninitialized."):
            nlp.solve(None)

    def test_solve__raises__with_free_parameters(self):
        nlp = Nlp(sym_type=self.sym_type)
        x = nlp.variable("x")[0]
        p = nlp.parameter("p")
        nlp.minimize(p * (x**2))
        nlp.init_solver(OPTS)
        with self.assertRaisesRegex(
            RuntimeError, "Trying to solve the NLP with unspecified parameters: p."
        ):
            nlp.solve({})

    def test_solve__computes_correctly__example_0(self):
        nlp = Nlp(sym_type=self.sym_type)
        x = nlp.variable("x", (2, 1))[0]
        y = nlp.variable("y", (3, 1))[0]
        p = nlp.parameter("p")
        nlp.minimize(p + (x.T @ x + y.T @ y))
        nlp.init_solver(OPTS)
        sol = nlp.solve({"p": 3})
        self.assertTrue(sol.success)
        np.testing.assert_allclose(sol.f, 3)
        for k in sol.vals.keys():
            np.testing.assert_allclose(sol.vals[k], 0)
        o = sol.value(p + (x.T @ x + y.T @ y))
        np.testing.assert_allclose(sol.f, o)

    def test_solve__computes_corretly__example_1a(self):
        # https://en.wikipedia.org/wiki/Lagrange_multiplier#Example_1a
        nlp = Nlp(sym_type=self.sym_type)
        x = nlp.variable("x")[0]
        y = nlp.variable("y")[0]
        nlp.constraint("c1", x**2 + y**2, "==", 1)
        nlp.minimize(-x - y)
        nlp.init_solver(OPTS)
        sol = nlp.solve()
        np.testing.assert_allclose(-sol.f, np.sqrt(2))
        for k in ("x", "y"):
            np.testing.assert_allclose(sol.vals[k], np.sqrt(2) / 2, atol=1e-9)
        np.testing.assert_allclose(sol.value(nlp.lam_g), 1 / np.sqrt(2), atol=1e-9)

    def test_solve__computes_corretly__example_1b(self):
        # https://en.wikipedia.org/wiki/Lagrange_multiplier#Example_1b
        nlp = Nlp(sym_type=self.sym_type)
        x = nlp.variable("x")[0]
        y = nlp.variable("y")[0]
        nlp.constraint("c1", x**2 + y**2, "==", 1)
        nlp.minimize((x + y) ** 2)
        nlp.init_solver(OPTS)
        sol = nlp.solve(vals0={"x": 0.5, "y": np.sqrt(1 - 0.5**2)})
        np.testing.assert_allclose(sol.f, 0, atol=1e-9)
        np.testing.assert_allclose(abs(sol.vals["x"]), np.sqrt(2) / 2, atol=1e-9)
        np.testing.assert_allclose(abs(sol.vals["y"]), np.sqrt(2) / 2, atol=1e-9)
        np.testing.assert_allclose(sol.value(nlp.lam_g), 0, atol=1e-9)

    def test_solve__computes_corretly__example_2(self):
        # https://en.wikipedia.org/wiki/Lagrange_multiplier#Example_2
        nlp = Nlp(sym_type=self.sym_type)
        x = nlp.variable("x")[0]
        y = nlp.variable("y")[0]
        nlp.constraint("c1", x**2 + y**2, "==", 3)
        nlp.minimize(x**2 * y)
        nlp.init_solver(OPTS)
        sol = nlp.solve(vals0={"x": np.sqrt(3 - 0.8**2), "y": -0.8})
        np.testing.assert_allclose(sol.f, -2, atol=1e-9)
        np.testing.assert_allclose(abs(sol.vals["x"]), np.sqrt(2), atol=1e-9)
        np.testing.assert_allclose(sol.vals["y"], -1, atol=1e-9)
        np.testing.assert_allclose(
            sol.value(sol.vals["x"] * (sol.vals["y"] + nlp.lam_g)), 0, atol=1e-9
        )

    def test_solve__computes_corretly__example_3(self):
        # https://en.wikipedia.org/wiki/Lagrange_multiplier#Example_3:_Entropy
        n = 50
        nlp = Nlp(sym_type=self.sym_type)
        p = nlp.variable("p", (n, 1), lb=1 / n * 1e-6)[0]
        nlp.constraint("c1", cs.sum1(p), "==", 1)
        nlp.minimize(cs.sum1(p * log(p, 2)))
        nlp.init_solver(OPTS)
        sol = nlp.solve(vals0={"p": np.random.rand(n)})
        np.testing.assert_allclose(sol.vals["p"], 1 / n, atol=1e-9)
        np.testing.assert_allclose(
            sol.value(-(1 / cs.log(2) + log(p, 2)) - nlp.lam_g), 0, atol=1e-6
        )

    def test_solve__computes_corretly__example_4(self):
        # https://en.wikipedia.org/wiki/Lagrange_multiplier#Example_4:_Numerical_optimization
        nlp = Nlp(sym_type=self.sym_type)
        x = nlp.variable("x")[0]
        nlp.constraint("c1", x**2, "==", 1)
        nlp.minimize(x**2)
        nlp.init_solver(OPTS)
        sol = nlp.solve(vals0={"x": 1 + 0.1 * np.random.rand()})
        np.testing.assert_allclose(sol.f, 1, atol=1e-9)
        np.testing.assert_allclose(sol.value(nlp.lam_g), -1, atol=1e-9)

    def test_solve__computes_corretly__example_5(self):
        # https://personal.math.ubc.ca/~israel/m340/kkt2.pdf
        nlp = Nlp(sym_type=self.sym_type)
        x = nlp.variable("x", (2, 1), lb=0)[0]
        nlp.constraint("c1", x[0] + x[1] ** 2, "<=", 2)
        nlp.minimize(-x[0] * x[1])
        nlp.init_solver(OPTS)
        sol = nlp.solve(vals0={"x": [1, 1]})
        np.testing.assert_allclose(-sol.f, np.sqrt(2 / 3) * 4 / 3, atol=1e-9)
        np.testing.assert_allclose(
            sol.vals["x"].full().flatten(), [4 / 3, np.sqrt(2 / 3)], atol=1e-9
        )
        np.testing.assert_allclose(
            sol.value(cs.vertcat(nlp.lam_h, nlp.lam_lbx)).full().flatten(),
            [np.sqrt(2 / 3), 0, 0],
            atol=1e-7,
        )

    def test_solve__computes_corretly__example_6(self):
        # https://www.reddit.com/r/cheatatmathhomework/comments/sw6nqs/optimization_using_kkt_conditions_for_constraints/
        nlp = Nlp(sym_type=self.sym_type)
        x = nlp.variable("x", (3, 1), lb=0)[0]
        y = nlp.variable("y", (3, 1), lb=0)[0]
        z = nlp.variable("z", (3, 1), lb=0)[0]
        nlp.minimize(cs.sum1(cs.vertcat(x, y, z)))
        nlp.constraint("c1", x.T @ [25, 15, 75] + 3 * cs.sqrt(cs.sumsqr(x)), "<=", 56)
        nlp.constraint("c2", y.T @ [75, 3, 3] + 3 * cs.sqrt(cs.sumsqr(y)), "<=", 87)
        nlp.constraint("c3", z.T @ [15, 22, 4] + 3 * cs.sqrt(cs.sumsqr(z)), "<=", 38)
        nlp.init_solver(OPTS)
        sol = nlp.solve()
        for (name, v) in [("x", x), ("y", y), ("z", z)]:
            val1 = sol.vals[name]
            val2 = sol.value(v)
            val3 = sol.value(nlp.variables[name])
            val4 = sol.value(nlp._vars[name])
            np.testing.assert_allclose(val1, val2, atol=1e-9)
            np.testing.assert_allclose(val2, val3, atol=1e-9)
            np.testing.assert_allclose(val3, val4, atol=1e-9)

    def test_to_function__raises__with_uninitiliazed_solver(self):
        nlp = Nlp(sym_type=self.sym_type)
        x = nlp.variable("x", lb=0)[0]
        y = nlp.variable("y")[0]
        xy = cs.vertcat(x, y)
        p = nlp.parameter("p")
        with self.assertRaises(RuntimeError):
            nlp.to_function("M", [p, xy], [xy], ["p", "xy"], ["xy"])

    def test_to_function__raises__with_free_variables(self):
        a = 0.2
        nlp = Nlp(sym_type=self.sym_type)
        x = nlp.variable("x", lb=0)[0]
        y = nlp.variable("y")[0]
        xy = cs.vertcat(x, y)
        c = nlp._csXX.sym("c")
        p = nlp.parameter("p")
        nlp.minimize((1 - x) ** 2 + a * (y - x**2) ** 2)
        nlp.init_solver(OPTS)
        with self.assertRaises(ValueError):
            nlp.to_function("M", [p], [xy], ["xy"], ["xy"])
        with self.assertRaises(ValueError):
            nlp.to_function("M", [p, xy], [xy, c], ["xy"], ["xy"])

    @parameterized.expand([(False,), (True,)])
    def test_to_function__computes_correct_solution__also_with_deepcopy(
        self, copy: bool
    ):
        a = 0.2
        nlp = Nlp(sym_type=self.sym_type)
        if copy:
            nlp = nlp.copy()
        x = nlp.variable("x", lb=0)[0]
        y = nlp.variable("y")[0]
        xy = cs.vertcat(x, y)
        p = nlp.parameter("p")
        nlp.minimize((1 - x) ** 2 + a * (y - x**2) ** 2)
        g = (x + 0.5) ** 2 + y**2
        nlp.constraint("c1", (p / 2) ** 2, "<=", g)
        nlp.constraint("c2", g, "<=", p**2)
        nlp.init_solver(OPTS)

        M = nlp.to_function("M", [p, xy], [xy], ["p", "xy"], ["xy"])

        sol = nlp.solve(pars={"p": 1.25})
        xy1 = sol.value(xy).full().flatten()
        xy2 = M(1.25, 0).full().flatten()

        np.testing.assert_allclose(xy1, [0.719011, 0.276609], atol=1e-4)
        np.testing.assert_allclose(xy2, [0.719011, 0.276609], atol=1e-4)

    def test_can_be_pickled(self):
        a = 0.2
        nlp = Nlp(sym_type=self.sym_type)
        x = nlp.variable("x", lb=0)[0]
        y = nlp.variable("y")[0]
        p = nlp.parameter("p")
        nlp.minimize((1 - x) ** 2 + a * (y - x**2) ** 2)
        g = (x + 0.5) ** 2 + y**2
        nlp.constraint("c1", (p / 2) ** 2, "<=", g)
        nlp.constraint("c2", g, "<=", p**2)
        nlp.init_solver(OPTS)

        with nlp.pickleable():
            nlp2 = pickle.loads(pickle.dumps(nlp))

        self.assertEqual(nlp.name, nlp2.name)


if __name__ == "__main__":
    unittest.main()
