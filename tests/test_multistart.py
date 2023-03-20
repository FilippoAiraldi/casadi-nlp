import pickle
import unittest
from itertools import product
from typing import Type, TypeVar, Union

import casadi as cs
import numpy as np
from parameterized import parameterized, parameterized_class

from csnlp.core.solutions import subsevalf
from csnlp.multistart import ParallelMultistartNlp, StackedMultistartNlp
from csnlp.multistart.multistart_nlp import _n

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
TMultiNlp = TypeVar("TMultiNlp", ParallelMultistartNlp, StackedMultistartNlp)


@parameterized_class("sym_type", [("SX",), ("MX",)])
class TestMultistartNlp(unittest.TestCase):
    def test_init__raises__with_invalid_number_of_starts(self):
        with self.assertRaisesRegex(
            ValueError, "Number of scenarios must be positive and > 0."
        ):
            StackedMultistartNlp(starts=0, sym_type=self.sym_type)

    def test_variable_parameter_and_constraint__builds_correct_copies(self):
        N = 3
        nlp = StackedMultistartNlp(starts=N, sym_type=self.sym_type)
        x = nlp.variable("x")[0]
        y = nlp.variable("y")[0]
        z = nlp.variable("z")[0]
        p = nlp.parameter("p")
        nlp.constraint("c0", x**2, "<=", p)
        nlp.constraint("c1", y, "==", p)
        nlp.constraint("c2", z, ">=", -p)
        self.assertEqual(nlp.nx, nlp._stacked_nlp.nx // N)
        self.assertEqual(nlp.np, nlp._stacked_nlp.np // N)
        self.assertEqual(nlp.ng, nlp._stacked_nlp.ng // N)
        self.assertEqual(nlp.nh, nlp._stacked_nlp.nh // N)

    def test_minimize__sums_objectives_in_unique_function(self):
        N = 3
        nlp = StackedMultistartNlp(starts=N, sym_type=self.sym_type)
        x = nlp.variable("x")[0]
        nlp.minimize(cs.exp((x - 1) ** 2))
        x_ = cs.DM(np.random.randn(*x.shape))
        x_dict = {_n("x", i): x_ for i in range(N)}
        f1 = subsevalf(nlp.f, x, x_)
        f2 = subsevalf(nlp._stacked_nlp.f, nlp._stacked_nlp._vars, x_dict)
        self.assertAlmostEqual(f1, f2 / N)

    def test_solve__raises__with_both_flags_on(self):
        N = 3
        nlp = StackedMultistartNlp(starts=N, sym_type=self.sym_type)
        with self.assertRaisesRegex(
            AssertionError,
            "`return_all_sols` and `return_stacked_sol` can't be both true.",
        ):
            nlp(None, None, return_all_sols=True, return_stacked_sol=True)

    @parameterized.expand(
        product([False, True], [StackedMultistartNlp, ParallelMultistartNlp])
    )
    def test_solve__computes_right_solution(
        self, copy: bool, multinlp_cls: Type[TMultiNlp]
    ):
        N = 3
        nlp = multinlp_cls(starts=N, sym_type=self.sym_type)
        x = nlp.variable("x", lb=-0.5, ub=1.4)[0]
        nlp.parameter("p")
        nlp.minimize(
            -0.3 * x**2
            - cs.exp(-10 * x**2)
            + cs.exp(-100 * (x - 1) ** 2)
            + cs.exp(-100 * (x - 1.5) ** 2)
        )
        nlp.init_solver(OPTS)
        if copy:
            nlp = nlp.copy()

        # solve manually
        x0s = [0.9, 0.5, 1.1]
        xfs, fs = [], []
        for x0 in x0s:
            sol = nlp.solve(pars={"p": 0}, vals0={"x": x0})
            xfs.append(sol.vals["x"])
            fs.append(sol.f)

        # solve with multistart
        args = ({"p": 0}, [{"x": x0} for x0 in x0s])
        best_sol = nlp.solve_multi(*args)
        all_sols = nlp.solve_multi(*args, return_all_sols=True)

        for xf, f, sol in zip(xfs, fs, all_sols):
            np.testing.assert_allclose(xf, sol.vals["x"])
            np.testing.assert_allclose(xf, sol.value(x))
            np.testing.assert_allclose(f, sol.f)
            np.testing.assert_allclose(f, sol.value(nlp.f))
        np.testing.assert_allclose(best_sol.f, min(fs))
        np.testing.assert_allclose(best_sol.value(nlp.f), min(fs))

    @parameterized.expand([(StackedMultistartNlp,), (ParallelMultistartNlp,)])
    def test_is_pickleable(self, multinlp_cls: Type[TMultiNlp]):
        N = 3
        nlp = multinlp_cls(starts=N, sym_type=self.sym_type)
        x = nlp.variable("x", lb=-0.5, ub=1.4)[0]
        nlp.parameter("p")
        nlp.minimize(
            -0.3 * x**2
            - cs.exp(-10 * x**2)
            + cs.exp(-100 * (x - 1) ** 2)
            + cs.exp(-100 * (x - 1.5) ** 2)
        )
        nlp.init_solver(OPTS)
        nlp2: Union[StackedMultistartNlp, ParallelMultistartNlp] = pickle.loads(
            pickle.dumps(nlp)
        )
        self.assertEqual(nlp.name, nlp2.name)
        if multinlp_cls is StackedMultistartNlp:
            self.assertEqual(nlp._stacked_nlp.name, nlp2._stacked_nlp.name)


if __name__ == "__main__":
    unittest.main()
