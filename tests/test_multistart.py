import pickle
import unittest
from copy import deepcopy
from itertools import product
from typing import TypeVar
from unittest.mock import Mock

import casadi as cs
import numpy as np
from parameterized import parameterized, parameterized_class

from csnlp.core.solutions import subsevalf
from csnlp.multistart import (
    MappedMultistartNlp,
    ParallelMultistartNlp,
    RandomStartPoint,
    RandomStartPoints,
    StackedMultistartNlp,
    StructuredStartPoint,
    StructuredStartPoints,
)
from csnlp.multistart.multistart_nlp import MultistartNlp, _n

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
MULTI_NLP_CLASSES = [ParallelMultistartNlp, StackedMultistartNlp, MappedMultistartNlp]
TMultiNlp = TypeVar("TMultiNlp", *MULTI_NLP_CLASSES)


class TestStartPoints(unittest.TestCase):
    def test_random_start_points__raises__with_invalid_generator_method(self):
        method = "an_invalid_method"
        S = RandomStartPoints({"x": RandomStartPoint(method)}, 10)
        with self.assertRaisesRegex(
            AttributeError,
            f"'numpy.random._generator.Generator' object has no attribute '{method}'",
        ):
            list(S)

    def test_random_start_points__calls_random_method_with_args_and_kwargs(self):
        method = "uniform"
        multistarts = 10
        args, kwargs = object(), object()
        S = RandomStartPoints(
            {"x": RandomStartPoint(method, args, optional=kwargs)}, multistarts
        )
        S.np_random = Mock()
        points = list(S)
        self.assertEqual(len(points), multistarts)
        self.assertEqual(len(S.np_random.mock_calls), multistarts)
        getattr(S.np_random, method).assert_called_with(args, optional=kwargs)

    def test_random_start_points__returns_correct_values__when_seeded(self):
        multistarts = 5
        seed = 69
        biases = {"x": 0.5, "y": -0.2}
        scales = {"x": 0.1, "y": 0.3}
        S = RandomStartPoints(
            points={"x": RandomStartPoint("uniform"), "y": RandomStartPoint("normal")},
            multistarts=multistarts,
            biases=biases,
            scales=scales,
            seed=seed,
        )
        expecteds = [
            (0.5803723752156749, 0.33860174484688865),
            (0.8649616587869663, -0.9947886944731732),
            (0.47658484007382285, 1.0424725873226823),
            (0.5446347761155709, -0.5473146662455354),
            (0.008451104579143554, -0.34391256021075023),
        ]
        for actual, expected in zip(S, expecteds):
            self.assertAlmostEqual(actual["x"], expected[0] * scales["x"] + biases["x"])
            self.assertAlmostEqual(actual["y"], expected[1] * scales["y"] + biases["y"])

    def test_structured_start_points__returns_correct_values(self):
        multistarts = np.random.randint(10, 100)
        x_bnds = (np.random.rand() * 10, np.random.rand() * 100 + 20)
        y_bnds = (np.random.rand() * 10, np.random.rand() * 100 + 20)
        S = StructuredStartPoints(
            {"x": StructuredStartPoint(*x_bnds), "y": StructuredStartPoint(*y_bnds)},
            multistarts,
        )
        x_space = np.linspace(*x_bnds, multistarts)
        y_space = np.linspace(*y_bnds, multistarts)
        for actual, expected_x, expected_y in zip(S, x_space, y_space):
            self.assertEqual(actual["x"], expected_x)
            self.assertEqual(actual["y"], expected_y)


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
        y = nlp.variable("y", (3, 1))[0]
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
            nlp((None,), (None,), return_all_sols=True, return_stacked_sol=True)

    @parameterized.expand(product([False, True], MULTI_NLP_CLASSES))
    def test_solve__computes_right_solution(
        self, copy: bool, multinlp_cls: type[TMultiNlp]
    ):
        N = 3
        nlp = multinlp_cls(starts=N, sym_type=self.sym_type)
        x = nlp.variable("x", lb=-0.5, ub=1.4)[0]
        p = nlp.parameter("p")
        nlp.minimize(
            -0.3 * p * x**2
            - cs.exp(-10 * p * x**2)
            + cs.exp(-100 * p * (x - 1) ** 2)
            + cs.exp(-100 * p * (x - 1.5) ** 2)
        )
        nlp.init_solver(OPTS)
        if copy:
            nlp = deepcopy(nlp)

        # solve manually
        x0s = [0.9, 0.5, 1.1]
        xfs, fs = [], []
        for x0 in x0s:
            sol = nlp.solve(pars={"p": 1.0}, vals0={"x": x0})
            xfs.append(sol.vals["x"])
            fs.append(sol.f)

        # solve with multistart
        args = ({"p": 1.0}, [{"x": x0} for x0 in x0s])
        best_sol = nlp.solve_multi(*args)
        all_sols = nlp.solve_multi(*args, return_all_sols=True)

        for xf, f, sol in zip(xfs, fs, all_sols):
            np.testing.assert_allclose(xf, sol.vals["x"])
            np.testing.assert_allclose(xf, sol.value(x))
            np.testing.assert_allclose(f, sol.f)
            np.testing.assert_allclose(f, sol.value(nlp.f))
        np.testing.assert_allclose(best_sol.f, min(fs))
        np.testing.assert_allclose(best_sol.value(nlp.f), min(fs))

    @parameterized.expand([(cls,) for cls in MULTI_NLP_CLASSES])
    def test_is_pickleable(self, multinlp_cls: type[TMultiNlp]):
        N = 3
        nlp: MultistartNlp = multinlp_cls(starts=N, sym_type=self.sym_type)
        x = nlp.variable("x", lb=-0.5, ub=1.4)[0]
        p = nlp.parameter("p")
        nlp.minimize(
            -0.3 * p * x**2
            - cs.exp(-10 * p * x**2)
            + cs.exp(-100 * p * (x - 1) ** 2)
            + cs.exp(-100 * p * (x - 1.5) ** 2)
        )
        nlp.init_solver(OPTS)

        with cs.global_pickle_context():
            pickled = pickle.dumps(nlp)
        with cs.global_unpickle_context():
            other: MultistartNlp = pickle.loads(pickled)

        self.assertEqual(nlp.name, other.name)
        if multinlp_cls is StackedMultistartNlp:
            self.assertEqual(nlp._stacked_nlp.name, other._stacked_nlp.name)


if __name__ == "__main__":
    unittest.main()
