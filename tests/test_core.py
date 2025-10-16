import random
import unittest
from functools import cached_property, lru_cache
from itertools import product, repeat
from typing import Union

import casadi as cs
import numpy as np
from parameterized import parameterized

from csnlp import Nlp
from csnlp.core.cache import invalidate_cache
from csnlp.core.data import array2cs, cs2array, find_index_in_vector
from csnlp.core.debug import NlpDebug, NlpDebugEntry
from csnlp.core.derivatives import hohessian, hojacobian
from csnlp.core.scaling import MinMaxScaler, Scaler
from csnlp.core.solutions import EagerSolution, Solution, subsevalf

GROUPS = set(NlpDebug._types.keys())


class Dummy:
    def __init__(self) -> None:
        self.counter1 = 0
        self.counter2 = 0

    @cached_property
    def prop1(self) -> int:
        self.counter1 += 1
        return self.counter1

    @lru_cache
    def method2(self) -> int:
        self.counter2 += 1
        return self.counter2

    @invalidate_cache(prop1, method2)
    def clear_cache(self) -> None:
        return


class Dummy2(Dummy):
    def __init__(self) -> None:
        super().__init__()
        self.counter3 = 0

    @cached_property
    def prop3(self) -> int:
        self.counter3 += 1
        return self.counter3

    @invalidate_cache(prop3)
    def clear_cache(self) -> None:
        return super().clear_cache()


class DummySolution(Solution):
    def __init__(self, f, success, status):
        self._f = f
        self._stats = {"success": success, "return_status": status}
        self._solver_plugin = "osqp"

    @property
    def f(self):
        return self._f


class TestFuncs(unittest.TestCase):
    def test_invalidate_cache__raises__with_invalid_type(self):
        with self.assertRaises(TypeError):
            invalidate_cache(5)

    def test_invalidate_cache__clears_property_cache(self):
        dummy = Dummy()
        dummy.prop1
        dummy.prop1
        dummy.method2()
        dummy.method2()
        self.assertEqual(dummy.counter1, 1)
        self.assertEqual(dummy.counter2, 1)
        dummy.clear_cache()
        dummy.prop1
        dummy.method2()
        self.assertEqual(dummy.counter1, 2)
        self.assertEqual(dummy.counter2, 2)
        dummy.clear_cache()
        dummy.prop1
        dummy.prop1
        dummy.method2()
        dummy.method2()
        self.assertEqual(dummy.counter1, 3)
        self.assertEqual(dummy.counter2, 3)

    def test_invalidate_cache__invalidates_only_current_object(self):
        dummy1, dummy2 = Dummy(), Dummy()
        dummy1.prop1
        dummy2.prop1
        self.assertEqual(dummy1.counter1, 1)
        self.assertEqual(dummy2.counter1, 1)
        dummy1.clear_cache()
        dummy1.prop1
        dummy2.prop1
        self.assertEqual(dummy1.counter1, 2)
        self.assertEqual(dummy2.counter1, 1)

    def test_invalidate_cache__accepts_new_caches_to_clear(self):
        dummy = Dummy2()
        dummy.prop1
        dummy.prop1
        dummy.method2()
        dummy.method2()
        dummy.prop3
        dummy.prop3
        self.assertEqual(dummy.counter1, 1)
        self.assertEqual(dummy.counter2, 1)
        self.assertEqual(dummy.counter3, 1)
        dummy.clear_cache()
        dummy.prop1
        dummy.method2()
        dummy.prop3
        self.assertEqual(dummy.counter1, 2)
        self.assertEqual(dummy.counter2, 2)
        self.assertEqual(dummy.counter3, 2)


class TestNlpDebug(unittest.TestCase):
    def test_register__adds_correct_info(self):
        debug = NlpDebug()
        name = "a name"
        shape = (2, 3)
        for group in GROUPS:
            debug.register(group, name, shape)
            info: tuple[range, NlpDebugEntry] = getattr(debug, f"_{group}_info")[0]
            self.assertEqual(info[0], range(shape[0] * shape[1]))
            self.assertEqual(info[1].name, name)
            self.assertEqual(info[1].shape, shape)
            self.assertEqual(info[1].type, NlpDebug._types[group])

    def test_register__raises__with_invalid_group(self):
        debug = NlpDebug()
        while True:
            group = chr(random.randint(ord("a"), ord("z")))
            if group not in GROUPS:
                break
        with self.assertRaises(AttributeError):
            debug.register(group, "var1", (1, 2))

    def test_xhg_describe__gets_corret_variables(self):
        debug = NlpDebug()
        for group in GROUPS:
            debug.register(group, "var1", (3, 3))
            debug.register(group, "var2", (1, 1))
            info1: NlpDebugEntry = getattr(debug, f"{group}_describe")(0)
            info2: NlpDebugEntry = getattr(debug, f"{group}_describe")(9)
            self.assertEqual(info1.name, "var1")
            self.assertEqual(info2.name, "var2")

    def test_xhg_describe__raises__with_outofbound_index(self):
        debug = NlpDebug()
        for group in GROUPS:
            debug.register(group, "var1", (1, 1))
            debug.register(group, "var2", (1, 1))
            with self.assertRaises(IndexError):
                getattr(debug, f"{group}_describe")(10_000)


class TestSolutions(unittest.TestCase):
    def test_subsevalf__raises__when_type_is_invalid(self):
        with self.assertRaises(Exception):
            subsevalf(None, unittest.TestCase, None)

    def test_subsevalf__raises__when_evaluating_a_symbolic_expr(self):
        x = cs.SX.sym("x")
        y = cs.SX.sym("y")
        with self.assertRaises(RuntimeError):
            subsevalf(x**2, x, y, eval=True)

    @parameterized.expand([(cs.SX,), (cs.MX,)])
    def test_subsevalf__computes_correct_value(
        self, XX: Union[type[cs.SX], type[cs.MX]]
    ):
        shape = (3, 4)
        V = {
            "x": (XX.sym("x", *shape), np.random.rand(*shape) * 10),
            "y": (XX.sym("y", *shape), np.random.rand(*shape) * 5),
            "z": (XX.sym("z"), np.random.rand() + 1),
        }
        expr, expected_val = ((V["x"][i] / V["y"][i]) ** V["z"][i] for i in range(2))
        actual_vals = [
            subsevalf(
                subsevalf(
                    subsevalf(expr, V["x"][0], V["x"][1], eval=False),
                    V["y"][0],
                    V["y"][1],
                    eval=False,
                ),
                V["z"][0],
                V["z"][1],
                eval=True,
            ),
            subsevalf(
                expr,
                (s for s, _ in V.values()),
                (v for _, v in V.values()),
                eval=True,
            ),
            subsevalf(
                expr,
                {n: s for n, (s, _) in V.items()},
                {n: v for n, (_, v) in V.items()},
                eval=True,
            ),
        ]
        for actual_val in actual_vals:
            np.testing.assert_allclose(expected_val, actual_val)

    @parameterized.expand([(cs.SX,), (cs.MX,)])
    def test_eager_solution__computes_correct_value(
        self, XX: Union[type[cs.SX], type[cs.MX]]
    ):
        shape = (3, 4)
        V = {
            "x": (XX.sym("x", shape), np.random.rand(*shape) * 10),
            "y": (XX.sym("y", shape), np.random.rand(*shape) * 5),
            "p": (XX.sym("p"), np.random.rand() + 1),
        }
        D = {
            "g": (XX.sym("g", shape), np.random.rand(*shape) * 10),
            "h": (XX.sym("h", shape), np.random.rand(*shape) * 5),
            "lbx": (XX.sym("lbx", shape), np.random.rand(*shape) * 10),
            "ubx": (XX.sym("ubx", shape), np.random.rand(*shape) * 5),
        }

        def func(x, y, p, g, h, lbx, ubx):
            return (lbx / h) * cs.exp(x * p / y + g * ubx)

        expr, expected = (
            func(
                V["x"][i],
                V["y"][i],
                V["p"][i],
                D["g"][i],
                D["h"][i],
                D["lbx"][i],
                D["ubx"][i],
            )
            for i in range(2)
        )
        S = EagerSolution(
            0.0,
            V["p"][0],
            V["p"][1],
            cs.veccat(V["x"][0], V["y"][0]),
            cs.veccat(V["x"][1], V["y"][1]),
            cs.veccat(D["g"][0], D["h"][0]),
            cs.veccat(D["g"][1], D["h"][1]),
            cs.veccat(D["lbx"][0], D["ubx"][0]),
            cs.veccat(D["lbx"][1], D["ubx"][1]),
            {n: s for n, (s, _) in V.items()},
            {n: v for n, (_, v) in V.items()},
            {n: s for n, (s, _) in D.items()},
            {n: v for n, (_, v) in D.items()},
            {},
            "a_solver_plugin",
        )
        np.testing.assert_allclose(expected, S.value(expr))

    @parameterized.expand([(False,), (True,)])
    def test_eager_solution__reports_success_and_barrier_properly(self, flag: bool):
        mu = np.abs(np.random.randn(10)).tolist()
        S = EagerSolution(
            *repeat(None, 13),
            stats={"success": flag, "iterations": {"mu": mu}},
            solver_plugin="a_solver_plugin",
        )
        self.assertEqual(S.success, flag)
        self.assertEqual(S.barrier_parameter, mu[-1])

    def test_cmp_key__returns_correct_solution(self):
        test_cases = [
            {
                "name": "All successful, varying 'f' values",
                "sols": [(1, True, ""), (2, True, ""), (3, True, "")],
                "expected": 0,
            },
            {
                "name": "Mixed success, all feasible",
                "sols": [(1, False, ""), (2, True, ""), (3, True, "")],
                "expected": 1,
            },
            {
                "name": "No successful, mixed feasibility",
                "sols": [(1, False, "infeasible"), (2, False, ""), (3, False, "")],
                "expected": 1,
            },
            {
                "name": "All infeasible",
                "sols": [
                    (1, False, "infeasible"),
                    (2, False, "infeasible"),
                    (3, False, "infeasible"),
                ],
                "expected": 0,
            },
            {
                "name": "Mixed success, infeasibility, varying 'f'",
                "sols": [
                    (2, True, ""),
                    (5, True, ""),
                    (3, False, ""),
                    (4, False, "infeasible"),
                ],
                "expected": 0,
            },
            {
                "name": "Real example",
                "sols": [
                    (387.48883666213, True, ""),
                    (387.488836662, False, "infeasible"),
                    (387.488836662129, True, ""),
                    (387.4888366621303, True, ""),
                ],
                "expected": 2,
            },
        ]
        for case in test_cases:
            with self.subTest(name=case["name"]):
                sols = (
                    {"f": f, "stats": {"success": success, "return_status": status}}
                    for f, success, status in case["sols"]
                )
                sols = enumerate(DummySolution(*s) for s in case["sols"])
                min_index, _ = min(sols, key=lambda sol: DummySolution.cmp_key(sol[1]))
                self.assertEqual(min_index, case["expected"])

    @parameterized.expand(
        product(
            ("MX", "SX"),
            (True, False),
            [
                # (
                #     "nlp",
                #     "sqpmethod",
                #     {
                #         "error_on_fail": False,
                #         "print_time": False,
                #         "print_status": False,
                #         "print_header": False,
                #         "print_iteration": False,
                #         "qpsol_options": {
                #             "error_on_fail": False, "printLevel": "none"
                #         },
                #     },
                # ),
                (
                    "nlp",
                    "ipopt",
                    {
                        "print_time": False,
                        "ipopt": {"print_level": 0, "sb": "yes"},
                    },
                ),
                # (
                #     "conic",
                #     "osqp",
                #     {
                #         "error_on_fail": False,
                #         "print_time": False,
                #         "osqp": {"verbose": False},
                #     },
                # ),
                ("conic", "qpoases", {"error_on_fail": False, "printLevel": "none"}),
                # ("conic", "proxqp", {"error_on_fail": False}),
                (
                    "conic",
                    "qrqp",
                    {
                        "error_on_fail": False,
                        "print_time": False,
                        "print_header": False,
                        "print_info": False,
                        "print_iter": False,
                    },
                ),
                ("conic", "clp", {"error_on_fail": False}),
                (
                    "nlp",
                    "bonmin",
                    {
                        "print_time": False,
                        "bonmin": {
                            "fp_log_level": 0,
                            "lp_log_level": 0,
                            "milp_log_level": 0,
                            "nlp_log_level": 0,
                            "oa_cuts_log_level": 0,
                            "oa_log_level": 0,
                        },
                    },
                ),
                # ("conic", "cbc", {"error_on_fail": False}),
            ],
        )
    )
    def test_infeasible(self, sym_type, is_feas, solver_data):
        solver_type, solver, solver_options = solver_data

        prob = Nlp(sym_type=sym_type)
        discrete = solver in ("bonmin", "cbc", "gurobi", "knitro")
        x, _, _ = prob.variable("x", discrete=discrete)
        lb = -abs(np.random.randn())
        ub = abs(np.random.randn())
        if not is_feas:
            lb, ub = ub, lb
        prob.constraint("lb", x, ">=", lb)
        prob.constraint("ub", x, "<=", ub)
        prob.minimize(x)
        prob.init_solver(solver_options, solver, solver_type)
        sol = prob.solve()

        self.assertEqual(sol.success, is_feas)
        self.assertEqual(not sol.infeasible, is_feas)


class TestData(unittest.TestCase):
    @parameterized.expand(product([cs.MX, cs.SX], [(1, 1), (3, 1), (1, 3), (3, 3)]))
    def test_cs2array_array2cs__convert_properly(
        self, sym_type: Union[type[cs.SX], type[cs.MX]], shape: tuple[int, int]
    ):
        x = sym_type.sym("x", *shape)
        a = cs2array(x)
        y = array2cs(a)
        for i in np.ndindex(shape):
            if sym_type is cs.SX:
                self.assertTrue(cs.is_equal(x[i], a[i]))
                self.assertTrue(cs.is_equal(x[i], y[i]))
            else:
                x_ = cs.DM(np.random.rand(*x.shape))
                o = cs.evalf(cs.substitute(x[i] - a[i], x, x_))
                np.testing.assert_allclose(o, 0, atol=1e-9)
                o = cs.evalf(cs.substitute(y[i] - a[i], x, x_))
                np.testing.assert_allclose(o, 0, atol=1e-9)

    @parameterized.expand([(cs.SX,), (cs.MX,)])
    def test_find_index_in_vector__works_properly(
        self, sym_type: Union[type[cs.SX], type[cs.MX]]
    ):
        X = sym_type.sym("X", 5)
        x = cs.horzcat(X[3], X[2])
        np.testing.assert_array_equal([3, 2], find_index_in_vector(X, x))


class TestDerivatives(unittest.TestCase):
    @parameterized.expand([((2, 2),), ((3, 1),), ((1, 3),)])
    def test_hojacobian__computes_right_derivatives(self, shape: tuple[int, int]):
        x = cs.SX.sym("x", *shape)
        y = (
            (x.reshape((-1, 1)) @ x.reshape((1, -1)) + (x.T if x.is_row() else x))
            if x.is_vector()
            else (x * x.T - x)
        )
        J = hojacobian(y, x)
        self.assertEqual(J.ndim, 4)
        for index in np.ndindex(J.shape):
            x_ = np.random.randn(*x.shape)
            idx1, idx2 = index[:2], index[2:]
            o = cs.evalf(cs.substitute(J[index] - cs.jacobian(y[idx1], x[idx2]), x, x_))
            np.testing.assert_allclose(o, 0, atol=1e-9)

    @parameterized.expand([((2, 2),), ((3, 1),), ((1, 3),)])
    def test_hohessian__computes_right_derivatives(self, shape: tuple[int, int]):
        x = cs.SX.sym("x", *shape)
        y = (
            (x.reshape((-1, 1)) @ x.reshape((1, -1)) + (x.T if x.is_row() else x))
            if x.is_vector()
            else (x * x.T - x)
        )
        H, _ = hohessian(y, x)
        self.assertEqual(H.ndim, 6)
        for i in np.ndindex(y.shape):
            x_ = np.random.randn(*x.shape)
            H_ = cs.hessian(y[i], x)[0]
            diff = array2cs(H[i].reshape(H_.shape, order="F")) - H_
            o = cs.evalf(cs.substitute(diff, x, x_))
            np.testing.assert_allclose(o, 0, atol=1e-9)


class TestScaling(unittest.TestCase):
    def test_str_and_repr(self):
        N = Scaler({"x": [-1, 1]})
        for S in [N.__str__(), N.__repr__()]:
            self.assertIn(Scaler.__name__, S)

    def test_register__raises__when_registering_duplicate_ranges(self):
        N = Scaler({"x": [-1, 1]})
        with self.assertRaises(KeyError):
            N.register("x", 0, 2)
        with self.assertRaises(KeyError):
            N.register("x", 0, 2)
        N.register("y", 0, 2)

    def test_can_scale__only_valid_ranges(self):
        N = Scaler({"x": [-1, 1]})
        self.assertTrue(N.can_scale("x"))
        self.assertFalse(N.can_scale("u"))

    def test_scaling__scale_unscale__computes_right_values(self):
        N = Scaler({"x1": (-5, 2), "x2": ([-1, 7], [2, 10])})
        x1 = 4
        y1 = N.scale("x1", x1)
        z1 = N.unscale("x1", y1)
        np.testing.assert_equal(y1, (4 + 5) / 2)
        np.testing.assert_equal(x1, z1)

        x2 = np.asarray([-5, 2])
        y2 = N.scale("x2", x2)
        z2 = N.unscale("x2", y2)
        np.testing.assert_equal(y2, [(-5 + 1) / 2, (2 - 7) / 10])
        np.testing.assert_equal(x2, z2)

    def test_minmaxscaling_scale_unscale__computes_right_values(self):
        N = MinMaxScaler(
            {"x1": (-5, 2), "x2": (np.asarray([-1, 7]), np.asarray([2, 10]))}
        )
        x1 = 4
        y1 = N.scale("x1", x1)
        z1 = N.unscale("x1", y1)
        np.testing.assert_equal(y1, (4 + 5) / (5 + 2))
        np.testing.assert_equal(x1, z1)

        x2 = np.asarray([-5, 2])
        y2 = N.scale("x2", x2)
        z2 = N.unscale("x2", y2)
        np.testing.assert_equal(y2, [(-5 + 1) / (2 + 1), (2 - 7) / (10 - 7)])
        np.testing.assert_equal(x2, z2)


if __name__ == "__main__":
    unittest.main()
