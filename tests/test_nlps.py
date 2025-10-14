import pickle
import unittest
from copy import deepcopy
from itertools import product
from typing import Union
from unittest.mock import Mock

import casadi as cs
import numpy as np
from parameterized import parameterized, parameterized_class

from csnlp import Nlp
from csnlp.core.solutions import subsevalf
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
        vars: Union[list[cs.SX], list[cs.MX]],
        force_numerical: bool = False,
    ) -> bool:
        if isinstance(rhs, (int, float)):
            rhs = np.full(lhs.shape, rhs)
        if not force_numerical and self.sym_type == "SX":
            return cs.is_equal(lhs, rhs)
        old = cs.vvcat(vars)
        new = np.random.randn(*old.shape)
        np.testing.assert_allclose(subsevalf(lhs, old, new), subsevalf(rhs, old, new))
        return True

    def test_init__raises__with_invalid_sym_type(self):
        with self.assertRaises(AttributeError):
            Nlp(sym_type="a_random_sym_type")

    def test_init__saves_sym_type(self):
        nlp = Nlp(sym_type=self.sym_type)
        self.assertEqual(nlp._sym_type.__name__, self.sym_type)

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

    def test_variable__discrete__is_invalidated_when_new_variable_is_added(self):
        n_variables = 30
        np_random = np.random.default_rng()

        nlp = Nlp(sym_type=self.sym_type)
        discretes = np_random.random(n_variables) < 0.5
        shapes = [tuple(np_random.integers(1, 10, 2)) for _ in range(n_variables)]
        discrete_dict = {f"x{i}": d for i, d in enumerate(discretes)}
        discrete = np.concatenate(
            [
                (np.ones if d else np.zeros)(np.prod(s), dtype=bool)
                for s, d in zip(shapes, discrete_dict.values())
            ]
        )
        cumnx = np.cumsum([np.prod(s) for s in shapes])

        for i, (shape, discrete_) in enumerate(zip(shapes, discretes)):
            nlp.variable(f"x{i}", shape, discrete_)
            np.testing.assert_array_equal(nlp.discrete, discrete[: cumnx[i]])

    def test_variable__creates_correct_variable(self):
        n_variables = 30
        np_random = np.random.default_rng()

        nlp = Nlp(sym_type=self.sym_type, debug=True)
        discretes = np_random.random(n_variables) < 0.5
        shapes = [tuple(np_random.integers(1, 10, 2)) for _ in range(n_variables)]
        lbs = [np_random.random(size=s) - 1 for s in shapes]
        ubs = [np_random.random(size=s) + 1 for s in shapes]
        xs, lams_lb, lams_ub = [], [], []
        for i, (shape, discrete, lb, ub) in enumerate(zip(shapes, discretes, lbs, ubs)):
            x, lam_lb, lam_ub = nlp.variable(f"x{i}", shape, discrete, lb, ub)
            xs.append(x)
            lams_lb.append(lam_lb)
            lams_ub.append(lam_ub)

        for shape, x, lam_lb, lam_ub in zip(shapes, xs, lams_lb, lams_ub):
            self.assertEqual(x.shape, shape)
            self.assertEqual(lam_lb.shape, (np.prod(shape), 1))
            self.assertEqual(lam_ub.shape, (np.prod(shape), 1))

        nx = sum(np.prod(s) for s in shapes)
        self.assertEqual(nlp.nx, nx)
        self.cmp(nlp.x, cs.vvcat(xs), vars=xs)

        discrete = np.concatenate(
            [
                (np.ones if d else np.zeros)(np.prod(s), dtype=bool)
                for s, d in zip(shapes, discretes)
            ]
        )
        np.testing.assert_array_equal(nlp.discrete, discrete)

        lb = cs.vvcat(lbs)
        np.testing.assert_allclose(nlp.lbx, lb.full().flat)
        ub = cs.vvcat(ubs)
        np.testing.assert_allclose(nlp.ubx, ub.full().flat)

        k = 0
        for i, shape in enumerate(shapes):
            name = f"x{i}"
            for _ in range(np.prod(shape)):
                self.assertEqual(name, nlp.debug.x_describe(k).name)
                self.assertEqual(shape, nlp.debug.x_describe(k).shape)
                k += 1
            self.assertTrue(cs.is_equal(nlp.variables[name], xs[i]))

        with self.assertRaises(IndexError):
            nlp.debug.x_describe(nx + 1)

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
        nc1, nc2 = np.prod(shape1), np.prod(shape2)
        nc = nc1 + nc2
        nlp = Nlp(sym_type=self.sym_type, debug=True)
        x = nlp.variable("x", shape1)[0]
        y = nlp.variable("y", shape2)[0]
        c1, lam_c1 = nlp.constraint("c1", x, op1, 5)
        c2, lam_c2 = nlp.constraint("c2", 5, op2, y)

        self.assertEqual(c1.shape, shape1)
        self.assertEqual(lam_c1.shape, (nc1, 1))
        self.assertEqual(c2.shape, shape2)
        self.assertEqual(lam_c2.shape, (nc2, 1))
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

        c = cs.veccat(c1, c2)
        lam = cs.vertcat(lam_c1, lam_c2)
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
        nlp.minimize(-(x[0] ** 2) + 5 * (cs.log(x[0]) - 1) ** 2 + 2 * v)
        nlp.init_solver(OPTS)
        sol1 = nlp.solve(vals0={"x": [1, 1], "v": 0})

        # solve with automatic soft variable
        nlp = Nlp(sym_type=self.sym_type)
        x = nlp.variable("x", (2, 1), lb=0)[0]
        nlp.constraint("c0", 3 * x[0] + x[1], "<=", 5)
        _, _, v = nlp.constraint("c1", x[1] ** 2, ">=", 20, soft=True)
        nlp.minimize(-(x[0] ** 2) + 5 * (cs.log(x[0]) - 1) ** 2 + 2 * v)
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
        c1_name = f"lam_{'g' if op1 == '==' else 'h'}_c1"
        c2_name = f"lam_{'g' if op2 == '==' else 'h'}_c2"
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

        lam = cs.vertcat(lam_g, lam_h, lam_lbx, lam_ubx)
        y = cs.veccat(x, lam_g, lam_h, lam_lbx, lam_ubx)

        vars = [x, lam_g, lam_h, lam_lbx, lam_ubx]
        self.cmp(nlp.lam, lam, vars=vars)
        self.cmp(nlp.primal_dual, y, vars=vars)

    @parameterized.expand([(True,), (False,)])
    def test_h_lbx_ubx__returns_correct_indices(self, remove_red_bnds: bool):
        nlp = Nlp(sym_type=self.sym_type, remove_redundant_x_bounds=remove_red_bnds)

        x1, lam_lbx_c1, lam_ubx_c1 = nlp.variable("x1", (2, 1))
        lam_lbx1, lam_ubx1 = nlp.lam_lbx, nlp.lam_ubx
        h_lbx1, h_ubx1 = nlp.h_lbx, nlp.h_ubx
        vars1 = [x1, lam_lbx_c1, lam_ubx_c1]
        x2, lam_lbx_c2, lam_ubx_c2 = nlp.variable("x2", (2, 1), lb=0)
        lam_lbx2, lam_ubx2 = nlp.lam_lbx, nlp.lam_ubx
        h_lbx2, h_ubx2 = nlp.h_lbx, nlp.h_ubx
        vars2 = vars1.copy() + [x2, lam_lbx_c2, lam_ubx_c2]
        x3, lam_lbx_c3, lam_ubx_c3 = nlp.variable("x3", (2, 1), ub=1)
        lam_lbx3, lam_ubx3 = nlp.lam_lbx, nlp.lam_ubx
        h_lbx3, h_ubx3 = nlp.h_lbx, nlp.h_ubx
        vars3 = vars2.copy() + [x3, lam_lbx_c3, lam_ubx_c3]
        x4, lam_lbx_c4, lam_ubx_c4 = nlp.variable("x4", (2, 1), lb=0, ub=1)
        lam_lbx4, lam_ubx4 = nlp.lam_lbx, nlp.lam_ubx
        h_lbx4, h_ubx4 = nlp.h_lbx, nlp.h_ubx
        vars4 = vars3.copy() + [x4, lam_lbx_c4, lam_ubx_c4]

        if remove_red_bnds:
            self.assertTrue(
                all(o.is_empty() for o in [h_lbx1, lam_lbx1, h_ubx1, lam_ubx1])
            )
            self.assertTrue(all(o.is_empty() for o in [h_ubx2, lam_ubx2]))
            self.cmp(cs.evalf(h_lbx2 - (-x2)), 0, vars=vars2)
            self.cmp(cs.evalf(lam_lbx2 - lam_lbx_c2), 0, vars=vars2)
            self.cmp(cs.evalf(h_lbx3 - (-x2)), 0, vars=vars3)
            self.cmp(cs.evalf(lam_lbx3 - lam_lbx2), 0, vars=vars3)
            self.cmp((h_ubx3 - (x3 - 1)), 0, vars=vars3, force_numerical=True)
            self.cmp(cs.evalf(lam_ubx3 - lam_ubx_c3), 0, vars=vars3)
            self.cmp(h_lbx4 - cs.vertcat(-x2, -x4), 0, vars=vars4, force_numerical=True)
            e = lam_lbx4 - cs.vertcat(lam_lbx_c2, lam_lbx_c4)
            self.cmp(e, 0, vars=vars4, force_numerical=True)
            e = h_ubx4 - cs.vertcat(x3 - 1, x4 - 1)
            self.cmp(e, 0, vars=vars4, force_numerical=True)
            e = lam_ubx4 - cs.vertcat(lam_ubx_c3, lam_ubx_c4)
            self.cmp(e, 0, vars=vars4, force_numerical=True)
        else:
            self.cmp(cs.evalf(lam_lbx1 - lam_lbx_c1), 0, vars=vars1)
            self.cmp(cs.evalf(lam_ubx1 - lam_ubx_c1), 0, vars=vars1)
            e = lam_lbx2 - cs.vertcat(lam_lbx_c1, lam_lbx_c2)
            self.cmp(e, 0, vars=vars2, force_numerical=True)
            e = lam_ubx2 - cs.vertcat(lam_ubx_c1, lam_ubx_c2)
            self.cmp(e, 0, vars=vars2)
            e = lam_lbx3 - cs.vertcat(lam_lbx_c1, lam_lbx_c2, lam_lbx_c3)
            self.cmp(e, 0, vars=vars3, force_numerical=True)
            e = lam_ubx3 - cs.vertcat(lam_ubx_c1, lam_ubx_c2, lam_ubx_c3)
            self.cmp(e, 0, vars=vars3, force_numerical=True)
            e = lam_lbx4 - cs.vertcat(lam_lbx_c1, lam_lbx_c2, lam_lbx_c3, lam_lbx_c4)
            self.cmp(e, 0, vars=vars4, force_numerical=True)
            e = lam_ubx4 - cs.vertcat(lam_ubx_c1, lam_ubx_c2, lam_ubx_c3, lam_ubx_c4)
            self.cmp(e, 0, vars=vars4, force_numerical=True)

            if self.sym_type == "SX":
                self.cmp(h_lbx1 - (-np.inf - x1), 0, vars=vars1, force_numerical=True)
                self.cmp(h_ubx1 - (x1 - np.inf), 0, vars=vars1, force_numerical=True)
                e = h_lbx2 - cs.vertcat(-np.inf - x1, 0 - x2)
                self.cmp(cs.evalf(e), 0, vars=vars2)
                e = h_ubx2 - cs.vertcat(x1 - np.inf, x2 - np.inf)
                self.cmp(cs.evalf(e), 0, vars=vars2)
                e = h_lbx3 - cs.vertcat(-np.inf - x1, 0 - x2, -np.inf - x3)
                self.cmp(cs.evalf(e), 0, vars=vars3)
                e = h_ubx3 - cs.vertcat(x1 - np.inf, x2 - np.inf, x3 - 1)
                self.cmp(cs.evalf(e), 0, vars=vars3)
                e = h_lbx4 - cs.vertcat(-np.inf - x1, 0 - x2, -np.inf - x3, 0 - x4)
                self.cmp(cs.evalf(e), 0, vars=vars4)
                e = h_ubx4 - cs.vertcat(x1 - np.inf, x2 - np.inf, x3 - 1, x4 - 1)
                self.cmp(cs.evalf(e), 0, vars=vars4)

    def test_init_solver__raises__when_objective_not_set(self):
        nlp = Nlp(sym_type=self.sym_type)
        with self.assertRaises(RuntimeError):
            nlp.init_solver(OPTS)

    def test_init_solver__raises__when_type_is_wrong(self):
        nlp = Nlp(sym_type=self.sym_type)
        with self.assertRaises(ValueError):
            nlp.init_solver(OPTS, type="a_random_type")

    def test_init_solver__saves_options_correctly(self):
        nlp = Nlp(sym_type=self.sym_type)
        x = nlp.variable("x")[0]
        nlp.minimize(x**2)
        nlp.init_solver(OPTS)
        self.assertDictEqual(OPTS, nlp.solver_opts)

    @parameterized.expand([("sqpmethod",), ("osqp",)])
    def test_init_solver__chooses_conic_or_nlp_correctly(self, solver: str):
        nlp = Nlp(sym_type=self.sym_type)
        x = nlp.variable("x")[0]
        nlp.minimize(x**2)

        old_qpsol, old_nlpsol = cs.qpsol, cs.nlpsol
        cs.qpsol = mock_qpsol = Mock()
        cs.nlpsol = mock_nlpsol = Mock()
        try:
            nlp.init_solver(solver=solver)
        finally:
            cs.qpsol, cs.nlpsol = old_qpsol, old_nlpsol

        if solver == "sqpmethod":
            mock_qpsol.assert_not_called()
            mock_nlpsol.assert_called_once()
        else:
            mock_qpsol.assert_called_once()
            mock_nlpsol.assert_not_called()

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
        for v in sol.vals.values():
            np.testing.assert_allclose(v, 0)
        o = sol.value(p + (x.T @ x + y.T @ y))
        np.testing.assert_allclose(sol.f, o)

    def test_solve__computes_corretly__example_1a(self):
        # https://en.wikipedia.org/wiki/Lagrange_multiplier#Example_1
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
        # https://en.wikipedia.org/wiki/Lagrange_multiplier#Example_1
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
        # https://en.wikipedia.org/wiki/Lagrange_multiplier#Example_3
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
        # https://en.wikipedia.org/wiki/Lagrange_multiplier#Example_4
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
        for name, v in [("x", x), ("y", y), ("z", z)]:
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
        c = nlp._sym_type.sym("c")
        p = nlp.parameter("p")
        nlp.minimize((1 - x) ** 2 + a * (y - x**2) ** 2)
        nlp.init_solver(OPTS)
        with self.assertRaises(RuntimeError):
            nlp.to_function("M", [p], [xy], ["xy"], ["xy"])
        with self.assertRaises(RuntimeError):
            nlp.to_function("M", [p, xy], [xy, c], ["xy"], ["xy"])

    @parameterized.expand([(False,), (True,)])
    def test_to_function__computes_correct_solution__also_with_deepcopy(
        self, copy: bool
    ):
        a = 0.2
        nlp = Nlp(sym_type=self.sym_type)
        if copy:
            nlp = deepcopy(nlp)
        x = nlp.variable("x", lb=0)[0]
        y = nlp.variable("y")[0]
        xy = cs.vertcat(x, y)
        p = nlp.parameter("p")
        nlp.minimize((1 - x) ** 2 + a * (y - x**2) ** 2)
        g = (x + 0.5) ** 2 + y**2
        nlp.constraint("c1", (p / 2) ** 2, "<=", g)
        nlp.constraint("c2", g, "<=", p**2)
        nlp.init_solver(OPTS)

        M = nlp.to_function("M", [p, xy], [xy], ["p", "xy"], ["xy_opt"])

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

        with cs.global_pickle_context():
            pickled = pickle.dumps(nlp)
        with cs.global_unpickle_context():
            other = pickle.loads(pickled)

        self.assertEqual(nlp.name, other.name)

    @parameterized.expand(product(("both", "lb", "ub"), (True, False)))
    def test_remove_variable_bounds__remove_bounds_correctly(
        self, direction: str, all_idx: bool
    ):
        shape = tuple(np.random.randint(3, 10, size=2))
        lb = np.random.rand(*shape) - 3
        ub = np.random.rand(*shape) + 3
        if all_idx:
            idx_to_remove = list(product(range(shape[0]), range(shape[1])))
        else:
            n_to_remove = np.random.randint(1, np.prod(shape) // 2)
            idx_to_remove = np.random.randint((0, 0), shape, size=(n_to_remove, 2))

        nlp = Nlp(sym_type=self.sym_type, remove_redundant_x_bounds=True)
        u_size = np.prod(nlp.variable("u", (5, 2), ub=+1)[0].shape)  # to create noise
        nlp.variable("x", shape, lb=lb, ub=ub)
        nlp.variable("z", (7, 9), lb=+2, ub=+3)  # to create noise
        nlp.remove_variable_bounds("x", direction, None if all_idx else idx_to_remove)

        if direction in {"both", "lb"}:
            lb_ = lb.copy()
            lb_mask_ = np.full(lb.shape, False)
            for i in idx_to_remove:
                lb_[tuple(i)] = -np.inf
                lb_mask_[tuple(i)] = True
            exp_lb = nlp.lbx.data[u_size : u_size + np.prod(shape)]
            exp_lb_mask = nlp.lbx.mask[u_size : u_size + np.prod(shape)]
            np.testing.assert_array_equal(lb_.reshape(-1, order="F"), exp_lb)
            np.testing.assert_array_equal(lb_mask_.reshape(-1, order="F"), exp_lb_mask)
            self.assertTrue(
                nlp.dual_variables["lam_lb_x"].size1() == (~exp_lb_mask).sum()
            )
            self.assertTrue(nlp.h_lbx.shape == nlp.lam_lbx.shape)
        if direction in {"both", "ub"}:
            ub_ = ub.copy()
            ub_mask_ = np.full(ub.shape, False)
            for i in idx_to_remove:
                ub_[tuple(i)] = +np.inf
                ub_mask_[tuple(i)] = True
            exp_ub = nlp.ubx.data[u_size : u_size + np.prod(shape)]
            exp_ub_mask = nlp.ubx.mask[u_size : u_size + np.prod(shape)]
            np.testing.assert_array_equal(ub_.reshape(-1, order="F"), exp_ub)
            np.testing.assert_array_equal(ub_mask_.reshape(-1, order="F"), exp_ub_mask)
            self.assertTrue(
                nlp.dual_variables["lam_ub_x"].size1() == (~exp_ub_mask).sum()
            )
            self.assertTrue(nlp.h_ubx.shape == nlp.lam_ubx.shape)

    @parameterized.expand(product(("both", "g", "h"), (False, True)))
    def test_remove_constraints__remove_bounds_correctly(
        self, remove: str, all_idx: bool
    ):
        nlp = Nlp(sym_type=self.sym_type)
        x = nlp.variable("x", tuple(np.random.randint(2, 5, size=2)))[0]
        y = nlp.variable("y", tuple(np.random.randint(2, 5, size=2)))[0]
        z = nlp.variable("z", tuple(np.random.randint(2, 5, size=2)))[0]  # noise
        w = nlp.variable("w", tuple(np.random.randint(2, 5, size=2)))[0]  # noise
        q = nlp.variable("q", tuple(np.random.randint(2, 5, size=2)))[0]  # noise
        p = nlp.variable("p", tuple(np.random.randint(2, 5, size=2)))[0]  # noise
        nlp.minimize(cs.sumsqr(cs.veccat(x, y, z, w, q, p)))
        nlp.constraint("h0", z, ">=", 0.0)  # noise
        nlp.constraint("h1", x, ">=", 1.0)
        nlp.constraint("h2", q, ">=", 0.0)  # noise
        nlp.constraint("g0", w, "==", 0.0)  # noise
        nlp.constraint("g1", y, "==", 2.0)
        nlp.constraint("g2", p, "==", 0.0)  # noise
        nlp.init_solver(OPTS)

        if all_idx:
            idx_to_remove_h1 = idx_to_remove_g1 = None
        else:
            n_to_remove = np.random.randint(1, np.prod(x.shape) // 2)
            idx_to_remove_h1 = np.unique(
                np.random.randint((0, 0), x.shape, size=(n_to_remove, 2)), axis=0
            )
            n_to_remove = np.random.randint(1, np.prod(y.shape) // 2)
            idx_to_remove_g1 = np.unique(
                np.random.randint((0, 0), y.shape, size=(n_to_remove, 2)), axis=0
            )
        remove_h = remove in {"both", "h"}
        remove_g = remove in {"both", "g"}
        if remove_h:
            nlp.remove_constraints("h1", idx_to_remove_h1)
        if remove_g:
            nlp.remove_constraints("g1", idx_to_remove_g1)

        sol = nlp.solve()

        expected_nh = cs.veccat(z, x, q).size1()
        expected_ng = cs.veccat(w, y, p).size1()
        expected_x = np.full(x.shape, 1.0)
        expected_y = np.full(y.shape, 2.0)
        if remove_h:
            if all_idx:
                expected_nh -= np.prod(x.shape)
                expected_x.fill(0.0)
            else:
                expected_nh -= idx_to_remove_h1.shape[0]
                for idx in idx_to_remove_h1:
                    expected_x[idx[0], idx[1]] = 0.0
        if remove_g:
            if all_idx:
                expected_ng -= np.prod(y.shape)
                expected_y.fill(0.0)
            else:
                expected_ng -= idx_to_remove_g1.shape[0]
                for idx in idx_to_remove_g1:
                    expected_y[idx[0], idx[1]] = 0.0

        self.assertEqual(nlp.nh, expected_nh)
        self.assertEqual(nlp.ng, expected_ng)
        np.testing.assert_allclose(sol.vals["z"], 0.0, atol=1e-4, rtol=1e-4)
        np.testing.assert_allclose(sol.vals["x"], expected_x, atol=1e-4, rtol=1e-4)
        np.testing.assert_allclose(sol.vals["q"], 0.0, atol=1e-4, rtol=1e-4)
        np.testing.assert_allclose(sol.vals["w"], 0.0, atol=1e-4, rtol=1e-4)
        np.testing.assert_allclose(sol.vals["y"], expected_y, atol=1e-4, rtol=1e-4)
        np.testing.assert_allclose(sol.vals["p"], 0.0, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
