import math as _math
import os
import pickle
import tempfile
import unittest
from itertools import product
from typing import Any, Optional

import casadi as cs
import cvxpy as cp
import numpy as np
from parameterized import parameterized
from scipy.stats import norm

from csnlp import Nlp
from csnlp.core.solutions import subsevalf
from csnlp.util import io, math

TMPFILENAME: str = ""


class EmptyClass(io.SupportsDeepcopyAndPickle): ...


class DictClass(EmptyClass):
    def __init__(self) -> None:
        super().__init__()
        self.z, self.w = 3, 4
        self.sym2 = cs.SX.sym("y", 3, 1)


class TestIo(unittest.TestCase):
    @parameterized.expand(
        [
            (5.0, False),
            (unittest.TestCase(), False),
            (cs.DM(5), True),
            (cs.SX.sym("x"), True),
            (cs.MX.sym("x"), True),
        ]
    )
    def test_is_casadi_object__guesses_correctly(self, obj: Any, result: bool):
        self.assertEqual(io.is_casadi_object(obj), result)

    def test_is_pickleable__fails_with_casadi_obj(self):
        self.assertTrue(io.is_pickleable(5))
        self.assertTrue(io.is_pickleable({5}))
        self.assertTrue(io.is_pickleable("hello"))
        self.assertFalse(io.is_pickleable(cs.SX.sym("x")))

    @parameterized.expand(
        [
            ("pkl",),
            ("xz",),
            ("pbz2",),
            ("gz",),
            ("bt",),
            ("bl2",),
            ("mat",),
            ("npz",),
        ]
    )
    def test_save_and_load__preserve_data_correctly(self, ext: str):
        global TMPFILENAME
        TMPFILENAME = f"{next(tempfile._get_candidate_names())}.{ext}"
        data = {"x": 5, "y": "ciao", "w": {"ci": "ao"}}
        io.save(TMPFILENAME, **data)
        data2 = io.load(TMPFILENAME)
        self.assertDictEqual(data, data2)

    def test_save_and_load__one_key_dict_is_simplified(self):
        global TMPFILENAME
        TMPFILENAME = f"{next(tempfile._get_candidate_names())}.pkl"
        data = {"x": 5}
        io.save(TMPFILENAME, **data)
        data2 = io.load(TMPFILENAME)
        self.assertIsNot(data2, dict)
        self.assertEqual(data["x"], data2)

    def tearDown(self) -> None:
        try:
            os.remove(TMPFILENAME)
        finally:
            return super().tearDown()

    @parameterized.expand([(False,), (True,)])
    def test_is_pickleable_and_deepcopy_able(self, copy: bool):
        ec = EmptyClass()
        dc = DictClass()
        if copy:
            ec1 = ec.copy()
            dc1 = dc.copy()
        else:
            ec1 = pickle.loads(pickle.dumps(ec))
            dc1 = pickle.loads(pickle.dumps(dc))
        self.assertIsNot(ec, ec1)
        self.assertIsNot(dc, dc1)
        self.assertTupleEqual((dc.z, dc.w), (dc1.z, dc1.w))
        if copy:
            self.assertTupleEqual(dc.sym2.shape, dc1.sym2.shape)
        else:
            self.assertFalse(hasattr(dc1, "sym2"))


class TestMath(unittest.TestCase):
    def test_log(self):
        base = np.random.rand() * 10
        x = np.random.rand() * 10
        self.assertEqual(math.log(x), _math.log(x))
        self.assertEqual(math.log(x, base), _math.log(x, base))

    @parameterized.expand(product([(3, 1), (1, 3), (3, 4)], [-2, -1, 0, 1, None]))
    def test_prod(self, shape: tuple[int, int], axis: Optional[int]):
        x = np.random.randn(*shape) * 3
        p = np.prod(x, axis=axis, keepdims=True)
        x_sx = cs.SX.sym("X", *shape)
        p_sx = subsevalf(math.prod(x_sx, axis=axis), x_sx, x)
        x_mx = cs.MX.sym("X", *shape)
        p_mx = subsevalf(math.prod(x_mx, axis=axis), x_mx, x)
        x_dm = cs.DM(x)
        p_dm = math.prod(x_dm, axis=axis)
        np.testing.assert_allclose(p, p_sx)
        np.testing.assert_allclose(p, p_mx)
        np.testing.assert_allclose(p, p_dm)

    def test_norm_cdf(self):
        shape = (3, 4)
        x_sx = cs.SX.sym("X", *shape)
        loc_sx = cs.SX.sym("loc", *shape)
        scale_sx = cs.SX.sym("scale", *shape)
        x_mx = cs.MX.sym("X", *shape)
        loc_mx = cs.MX.sym("loc", *shape)
        scale_mx = cs.MX.sym("scale", *shape)
        x = np.random.randn(*shape)
        loc = np.random.randn(*shape)
        scale = np.random.rand(*shape)
        cdf_sx = subsevalf(
            math.normal_cdf(x_sx, loc=loc_sx, scale=scale_sx),
            [x_sx, loc_sx, scale_sx],
            [x, loc, scale],
        )
        cdf_mx = subsevalf(
            math.normal_cdf(x_mx, loc=loc_mx, scale=scale_mx),
            [x_mx, loc_mx, scale_mx],
            [x, loc, scale],
        )
        cdf = norm.cdf(x, loc=loc, scale=scale)

        np.testing.assert_allclose(cdf, cdf_sx, atol=1e-7, rtol=1e-5)
        np.testing.assert_allclose(cdf, cdf_mx, atol=1e-7, rtol=1e-5)

    def test_norm_ppf(self):
        shape = (3, 4)
        x_sx = cs.SX.sym("X", *shape)
        loc_sx = cs.SX.sym("loc", *shape)
        scale_sx = cs.SX.sym("scale", *shape)
        x_mx = cs.MX.sym("X", *shape)
        loc_mx = cs.MX.sym("loc", *shape)
        scale_mx = cs.MX.sym("scale", *shape)
        x = np.random.rand(*shape)
        loc = np.random.randn(*shape)
        scale = np.random.rand(*shape)
        cdf_sx = subsevalf(
            math.normal_ppf(x_sx, loc=loc_sx, scale=scale_sx),
            [x_sx, loc_sx, scale_sx],
            [x, loc, scale],
        )
        cdf_mx = subsevalf(
            math.normal_ppf(x_mx, loc=loc_mx, scale=scale_mx),
            [x_mx, loc_mx, scale_mx],
            [x, loc, scale],
        )
        cdf = norm.ppf(x, loc=loc, scale=scale)

        np.testing.assert_allclose(cdf, cdf_sx, atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(cdf, cdf_mx, atol=1e-5, rtol=1e-5)

    @parameterized.expand(
        [
            ((3, 4), cs.DM([3, 3, 3, 3])),
            (([1, 2, 3, 4], 3), cs.DM(sum(([i] * 3 for i in range(1, 5)), []))),
            ((cs.DM([[1, 2], [3, 4]]), (1, 2)), cs.DM([[1, 1, 2, 2], [3, 3, 4, 4]])),
        ]
    )
    def test_repeat(self, inputs, expected):
        actual = math.repeat(*inputs)
        np.testing.assert_array_equal(actual, expected)

    @parameterized.expand(product(("SX", "MX"), (1, "inf")))
    def test_norm_1_and_inf(self, sym_type, ord):
        # see Boyd & Vandenberghe, ex. 4.11 (a) and (b)
        # https://egrcc.github.io/docs/math/cvxbook-solutions.pdf
        m, n = sorted(np.random.randint(10, 20, size=2))
        A = np.random.randn(n, m)
        b = np.random.randn(n)

        x = cp.Variable(m)
        objective = cp.Minimize(cp.norm(A @ x - b, ord))
        problem = cp.Problem(objective)
        problem.solve()
        f_cp = problem.value
        x_cp = x.value

        nlp = Nlp(sym_type)
        x, _, _ = nlp.variable("x", (m, 1))
        norm_fun = math.norm_1 if ord == 1 else math.norm_inf
        nlp.minimize(norm_fun(nlp, "", A @ x - b))
        nlp.init_solver(solver="clp")
        sol = nlp.solve()
        f_csnlp = sol.f
        x_csnlp = sol.value(x).toarray().flatten()

        np.testing.assert_allclose(f_cp, f_csnlp, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(x_cp, x_csnlp, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
