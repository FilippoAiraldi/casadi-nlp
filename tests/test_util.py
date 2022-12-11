import math as _math
import os
import pickle
import tempfile
import unittest
from typing import Any, Optional, Tuple

import casadi as cs
import numpy as np
from parameterized import parameterized
from scipy.stats import norm

from csnlp.core.solutions import subsevalf
from csnlp.util import io, math

TMPFILENAME: str = ""


class EmptyClass(io.SupportsDeepcopyAndPickle):
    ...


class SlotsClass(EmptyClass):
    __slots__ = ("x", "y", "sym1")

    def __init__(self) -> None:
        super().__init__()
        self.x, self.y = 1, 2
        self.sym1 = cs.SX.sym("x", 2, 1)


class DictClass(EmptyClass):
    def __init__(self) -> None:
        super().__init__()
        self.z, self.w = 3, 4
        self.sym2 = cs.SX.sym("y", 3, 1)


class SlotsAndDictClass(SlotsClass):
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

    def test_save_and_load__preserve_data_correctly(self):
        global TMPFILENAME
        TMPFILENAME = next(tempfile._get_candidate_names())
        data = {"x": 5, "y": "ciao", "w": {"ci": "ao"}}
        io.save(TMPFILENAME, **data)
        data2 = io.load(TMPFILENAME)
        self.assertDictEqual(data, data2)

    def test_save_and_load__one_key_dict_is_simplified(self):
        global TMPFILENAME
        TMPFILENAME = next(tempfile._get_candidate_names())
        data = {"x": 5}
        io.save(TMPFILENAME, **data)
        data2 = io.load(TMPFILENAME)
        self.assertIsNot(data2, dict)
        self.assertEqual(data["x"], data2)

    def tearDown(self) -> None:
        try:
            os.remove(f"{TMPFILENAME}.pkl")
        finally:
            return super().tearDown()

    @parameterized.expand([(False,), (True,)])
    def test_is_pickleable_and_deepcopy_able(self, copy: bool):
        ec = EmptyClass()
        sc = SlotsClass()
        dc = DictClass()
        sdc = SlotsAndDictClass()
        if copy:
            ec1 = ec.copy()
            sc1 = sc.copy()
            dc1 = dc.copy()
            sdc1 = sdc.copy()
        else:
            with ec.pickleable(), sc.pickleable(), dc.pickleable(), sdc.pickleable():
                ec1 = pickle.loads(pickle.dumps(ec))
                sc1 = pickle.loads(pickle.dumps(sc))
                dc1 = pickle.loads(pickle.dumps(dc))
                sdc1 = pickle.loads(pickle.dumps(sdc))
        self.assertIsNot(ec, ec1)
        self.assertIsNot(sc, sc1)
        self.assertTupleEqual((sc.x, sc.y), (sc1.x, sc1.y))
        self.assertIsNot(dc, dc1)
        self.assertTupleEqual((dc.z, dc.w), (dc1.z, dc1.w))
        self.assertIsNot(sdc, sdc1)
        self.assertTupleEqual(
            (sdc.x, sdc.y, sdc.z, sdc.w), (sdc1.x, sdc1.y, sdc1.z, sdc1.w)
        )
        if copy:
            self.assertTupleEqual(sc.sym1.shape, sc1.sym1.shape)
            self.assertTupleEqual(dc.sym2.shape, dc1.sym2.shape)
            self.assertTupleEqual(
                (sdc.sym1.shape, sdc.sym2.shape), (sdc1.sym1.shape, sdc1.sym2.shape)
            )
        else:
            self.assertFalse(hasattr(sc1, "sym1"))
            self.assertFalse(hasattr(dc1, "sym2"))
            self.assertFalse(hasattr(sdc1, "sym1") or hasattr(sdc1, "sym2"))


class TestMath(unittest.TestCase):
    def test_log(self):
        base = np.random.rand() * 10
        x = np.random.rand() * 10
        self.assertEqual(math.log(x), _math.log(x))
        self.assertEqual(math.log(x, base), _math.log(x, base))

    @parameterized.expand([(-2,), (-1,), (0,), (1,), (None,)])
    def test_prod(self, axis: Optional[int]):
        shape = (4, 5)
        x_sx = cs.SX.sym("X", *shape)
        x_mx = cs.MX.sym("X", *shape)
        x = np.random.randn(*shape) * 2
        p_sx = subsevalf(math.prod(x_sx, axis=axis), x_sx, x)
        p_mx = subsevalf(math.prod(x_mx, axis=axis), x_mx, x)
        p = np.prod(x, axis=axis, keepdims=True)
        np.testing.assert_allclose(p, p_sx)
        np.testing.assert_allclose(p, p_mx)

    @parameterized.expand([(1,), (5,), (10,)])
    def test_quad_form(self, n: int):
        x_sx = cs.SX.sym("X", n, 1)
        x_mx = cs.MX.sym("X", n, 1)
        for m in [1, n]:
            A_sx = cs.SX.sym("A", n, m)
            A_mx = cs.MX.sym("A", n, m)
            x = np.random.randn(n, 1) * 2
            A = np.random.randn(n, m) * 2
            p_sx = subsevalf(math.quad_form(A_sx, x_sx), [x_sx, A_sx], [x, A])
            p_mx = subsevalf(math.quad_form(A_mx, x_mx), [x_mx, A_mx], [x, A])
            p = x.T @ (np.diag(A.flat) if m == 1 else A) @ x

            np.testing.assert_allclose(p, p_sx)
            np.testing.assert_allclose(p, p_mx)

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
            math.norm_cdf(x_sx, loc=loc_sx, scale=scale_sx),
            [x_sx, loc_sx, scale_sx],
            [x, loc, scale],
        )
        cdf_mx = subsevalf(
            math.norm_cdf(x_mx, loc=loc_mx, scale=scale_mx),
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
            math.norm_ppf(x_sx, loc=loc_sx, scale=scale_sx),
            [x_sx, loc_sx, scale_sx],
            [x, loc, scale],
        )
        cdf_mx = subsevalf(
            math.norm_ppf(x_mx, loc=loc_mx, scale=scale_mx),
            [x_mx, loc_mx, scale_mx],
            [x, loc, scale],
        )
        cdf = norm.ppf(x, loc=loc, scale=scale)

        np.testing.assert_allclose(cdf, cdf_sx, atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(cdf, cdf_mx, atol=1e-5, rtol=1e-5)

    @parameterized.expand(
        [
            ((1, 1), 1),
            ((5, 4), 5),
            (
                (np.arange(2, 11, 2), 4),
                np.asarray(
                    [
                        [2, 4, 6, 8],
                        [2, 4, 6, 10],
                        [2, 4, 8, 10],
                        [2, 6, 8, 10],
                        [4, 6, 8, 10],
                    ]
                ),
            ),
            ((np.asarray([10, 20, 30]), 2), np.asarray([[10, 20], [10, 30], [20, 30]])),
        ]
    )
    def test_nchoosek__computes_correct_combinations(
        self, inp: Tuple[int, int], out: int
    ):
        out_ = math.nchoosek(*inp)
        np.testing.assert_allclose(out_, out)

    @parameterized.expand([(1, 4, 4), (10, 1, np.eye(10)), (4, 3, None)])
    def test_monomial_powers__computes_correct_powers(self, n, k, out):
        p = math.monomial_powers(n, k)
        self.assertEqual(p.shape[1], n)
        np.testing.assert_allclose(p.sum(axis=1), k)
        if out is not None:
            np.testing.assert_allclose(p, out)


if __name__ == "__main__":
    unittest.main()
