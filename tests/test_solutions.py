import unittest
from functools import partial

import casadi as cs
import numpy as np
from casadi.tools import entry, struct_MX, struct_SX

from csnlp.solutions import Solution, subsevalf


class TestSolutions(unittest.TestCase):
    def test_subsevalf__raises__when_type_is_invalid(self):
        with self.assertRaises(TypeError):
            subsevalf(None, unittest.TestCase, None)

    def test_subsevalf__raises__when_evaluating_a_symbolic_expr(self):
        x = cs.SX.sym("x")
        y = cs.SX.sym("y")
        with self.assertRaises(RuntimeError):
            subsevalf(x**2, x, y, eval=True)

    def test_subsevalf__computes_correct_value(self):
        shape = (3, 4)

        for XX, struct_X in [(cs.SX, struct_SX), (cs.MX, struct_MX)]:
            V = {
                "x": (XX.sym("x", *shape), np.random.rand(*shape) * 10),
                "y": (XX.sym("y", *shape), np.random.rand(*shape) * 5),
                "z": (XX.sym("z"), np.random.rand() + 1),
            }
            V_vec = cs.vertcat(*(cs.vec(v) for _, v in V.values()))
            V_struct = struct_X([entry(n, expr=s) for n, (s, _) in V.items()])
            expr, expected_val = (
                (V["x"][i] / V["y"][i]) ** V["z"][i] for i in range(2)
            )

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
                subsevalf(expr, V_struct, V_struct(V_vec), eval=True),
            ]
            for actual_val in actual_vals:
                np.testing.assert_allclose(expected_val, actual_val)

    def test_solution__computes_correct_value(self):
        f = 0
        shape = (3, 4)

        for XX, struct_X in [(cs.SX, struct_SX), (cs.MX, struct_MX)]:
            V = {
                "x": (XX.sym("x", *shape), np.random.rand(*shape) * 10),
                "y": (XX.sym("y", *shape), np.random.rand(*shape) * 5),
                "z": (XX.sym("z"), np.random.rand() + 1),
            }
            V_vec = cs.vertcat(*(cs.vec(v) for _, v in V.values()))
            V_struct = struct_X([entry(n, expr=s) for n, (s, _) in V.items()])
            expr, expected_val = (
                (V["x"][i] / V["y"][i]) ** V["z"][i] for i in range(2)
            )

            vals = V_struct(V_vec)
            get_value = partial(subsevalf, old=V_struct, new=vals)
            S = Solution(
                f=f, vars=V_struct, vals=V_struct(V_vec), stats={}, _get_value=get_value
            )
            np.testing.assert_allclose(expected_val, S.value(expr))

    def test_solution__reports_success_and_barrier_properly(self):
        mu = np.abs(np.random.randn(10)).tolist()
        for flag in (True, False):
            S = Solution(
                f=None,
                vars=None,
                vals=None,
                stats={"success": flag, "iterations": {"mu": mu}},
                _get_value=lambda x: x,
            )
            self.assertEqual(S.success, flag)
            self.assertEqual(S.barrier_parameter, mu[-1])


if __name__ == "__main__":
    unittest.main()
