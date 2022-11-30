import math as _math
import os
import tempfile
import unittest
from functools import cached_property, lru_cache
from itertools import product
from typing import Dict

import casadi as cs
import numpy as np
from scipy.stats import norm

from csnlp.solutions import subsevalf
from csnlp.util import array, data, funcs, io, math, scaling


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

    @funcs.cache_clearer(prop1, method2)
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

    @funcs.cache_clearer(prop3)
    def clear_cache(self) -> None:
        return super().clear_cache()


class TestFuncs(unittest.TestCase):
    def test_cache_clearer__raises__with_invalid_type(self):
        with self.assertRaises(TypeError):
            funcs.cache_clearer(5)

    def test_cache_clearer__clears_property_cache(self):
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

    def test_cache_clearer__accepts_new_caches_to_clear(self):
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

    def test_np_random__raises__with_invalid_seed(self):
        with self.assertRaises(ValueError):
            funcs.np_random(-1)

    def test_np_random__initializes_rng_with_correct_seed(self):
        for seed in (69, None):
            rng, actual_seed = funcs.np_random(seed)
            self.assertIsInstance(rng, np.random.Generator)
            if seed is not None:
                self.assertEqual(seed, actual_seed)
            else:
                self.assertIsInstance(actual_seed, int)


class TestArray(unittest.TestCase):
    def test_hojacobian__computes_right_derivatives(self):
        for shape in [(2, 2), (3, 1), (1, 3)]:
            x = cs.SX.sym("x", *shape)
            y = (
                (x.reshape((-1, 1)) @ x.reshape((1, -1)) + (x.T if x.is_row() else x))
                if x.is_vector()
                else (x * x.T - x)
            )
            J = array.hojacobian(y, x)
            self.assertEqual(J.ndim, 4)
            for index in np.ndindex(J.shape):
                x_ = np.random.randn(*x.shape)
                idx1, idx2 = index[:2], index[2:]
                o = cs.evalf(
                    cs.substitute(J[index] - cs.jacobian(y[idx1], x[idx2]), x, x_)
                )
                np.testing.assert_allclose(o, 0, atol=1e-9)

    def test_hohessian__computes_right_derivatives(self):
        for shape in [(2, 2), (3, 1), (1, 3)]:
            x = cs.SX.sym("x", *shape)
            y = (
                (x.reshape((-1, 1)) @ x.reshape((1, -1)) + (x.T if x.is_row() else x))
                if x.is_vector()
                else (x * x.T - x)
            )
            H, _ = array.hohessian(y, x)
            self.assertEqual(H.ndim, 6)
            for i in np.ndindex(y.shape):
                x_ = np.random.randn(*x.shape)
                H_ = data.cs2array(cs.hessian(y[i], x)[0])
                o = cs.evalf(
                    cs.substitute(H[i].reshape(H_.shape, order="F") - H_, x, x_)
                )
                np.testing.assert_allclose(o, 0, atol=1e-9)

    def test_jaggedstack__raises__with_empty_array(self):
        with self.assertRaises(ValueError):
            array.jaggedstack([])

    def test_jaggedstack__returns_correct_output(self):
        a1 = np.array([1, 2, 3]).astype(float)
        a2 = np.array([1, 2, 3, 4, 5]).astype(float)
        out_ = np.array([[1, 2, 3, np.nan, np.nan], [1, 2, 3, 4, 5]])
        out = array.jaggedstack((a1, a2), axis=0)
        np.testing.assert_allclose(out, out_)
        out = array.jaggedstack((a1, a2), axis=1)
        np.testing.assert_allclose(out, out_.T)
        with self.assertRaises(np.AxisError):
            array.jaggedstack((a1, a2), axis=2)


class TestData(unittest.TestCase):
    def test_is_casadi_object__guesses_correctly(self):
        for o, flag in [
            (5.0, False),
            (unittest.TestCase(), False),
            (cs.DM(5), True),
            (cs.SX.sym("x"), True),
            (cs.MX.sym("x"), True),
        ]:
            self.assertEqual(data.is_casadi_object(o), flag)

    def test_dict2struct__with_DM__returns_numerical_struct(self):
        d = {
            "x": cs.DM(np.random.rand(3, 2)),
            "y": cs.DM(np.random.rand(4, 1)),
        }
        s = data.dict2struct(d)
        self.assertIsInstance(s, data.DMStruct)
        for name, x in d.items():
            np.testing.assert_allclose(x, s[name])

    def test_dict2struct__with_SX__returns_sym_struct(self):
        d = {
            "x": cs.SX.sym("x", 3, 2),
            "y": cs.SX.sym("y", 4, 1),
        }
        s = data.dict2struct(d)
        self.assertIsInstance(s, data.struct_symSX)
        for name, x in d.items():
            self.assertTrue(cs.is_equal(s[name], x))

    def test_dict2struct__with_MX__returns_copy_of_dict(self):
        d = {
            "x": cs.MX.sym("x"),
            "y": cs.MX.sym("y"),
        }
        s = data.dict2struct(d)
        self.assertIsInstance(s, Dict)
        self.assertDictEqual(d, s)

    def test_cs2array_array2cs__convert_properly(self):
        for sym_type, shape in product(
            [cs.MX, cs.SX], [(1, 1), (3, 1), (1, 3), (3, 3)]
        ):
            x = sym_type.sym("x", *shape)
            a = data.cs2array(x)
            y = data.array2cs(a)
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


TMPFILENAME: str = ""


class TestIo(unittest.TestCase):
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


class TestMath(unittest.TestCase):
    def test_log(self):
        base = np.random.rand() * 10
        x = np.random.rand() * 10
        self.assertEqual(math.log(x), _math.log(x))
        self.assertEqual(math.log(x, base), _math.log(x, base))

    def test_prod(self):
        shape = (4, 5)
        for ax in (-2, -1, 0, 1, None):
            x_sx = cs.SX.sym("X", *shape)
            x_mx = cs.MX.sym("X", *shape)
            x = np.random.randn(*shape) * 2
            p_sx = subsevalf(math.prod(x_sx, axis=ax), x_sx, x)
            p_mx = subsevalf(math.prod(x_mx, axis=ax), x_mx, x)
            p = np.prod(x, axis=ax, keepdims=True)

            np.testing.assert_allclose(p, p_sx)
            np.testing.assert_allclose(p, p_mx)

    def test_quad_form(self):
        n = 5
        for m in (1, n):
            x_sx = cs.SX.sym("X", n, 1)
            x_mx = cs.MX.sym("X", n, 1)
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

        np.testing.assert_allclose(cdf, cdf_sx, atol=1e-7, rtol=1e-5)
        np.testing.assert_allclose(cdf, cdf_mx, atol=1e-7, rtol=1e-5)

    def test_nchoosek__computes_correct_combinations(self):
        for (inp, out) in [
            ((1, 1), 1),
            ((5, 4), 5),
            (
                (np.arange(2, 11, 2), 4),
                np.array(
                    [
                        [2, 4, 6, 8],
                        [2, 4, 6, 10],
                        [2, 4, 8, 10],
                        [2, 6, 8, 10],
                        [4, 6, 8, 10],
                    ]
                ),
            ),
            ((np.array([10, 20, 30]), 2), np.array([[10, 20], [10, 30], [20, 30]])),
        ]:
            out_ = math.nchoosek(*inp)
            np.testing.assert_allclose(out_, out)

    def test_monomial_powers__computes_correct_powers(self):
        for ((n, k), out) in [((1, 4), 4), ((10, 1), np.eye(10)), ((4, 3), None)]:
            p = math.monomial_powers(n, k)
            self.assertEqual(p.shape[1], n)
            np.testing.assert_allclose(p.sum(axis=1), k)
            if out is not None:
                np.testing.assert_allclose(p, out)


class TestScaling(unittest.TestCase):
    def test_str_and_repr(self):
        N = scaling.Scaling({"x": [-1, 1]})
        for S in [N.__str__(), N.__repr__()]:
            self.assertIn("scaling", S.lower())

    def test_register__raises__when_registering_duplicate_ranges(self):
        N = scaling.Scaling({"x": [-1, 1]})
        with self.assertRaises(KeyError):
            N.register("x", 0, 2)
        with self.assertRaises(KeyError):
            N.register("x", 0, 2)
        N.register("y", 0, 2)

    def test_can_scale__only_valid_ranges(self):
        N = scaling.Scaling({"x": [-1, 1]})
        self.assertTrue(N.can_scale("x"))
        self.assertFalse(N.can_scale("u"))

    def test_scaling__scale_unscale__computes_right_values(self):
        N = scaling.Scaling({"x1": (-5, 2), "x2": ([-1, 7], [2, 10])})
        x1 = 4
        y1 = N.scale("x1", x1)
        z1 = N.unscale("x1", y1)
        np.testing.assert_equal(y1, (4 + 5) / 2)
        np.testing.assert_equal(x1, z1)

        x2 = np.array([-5, 2])
        y2 = N.scale("x2", x2)
        z2 = N.unscale("x2", y2)
        np.testing.assert_equal(y2, [(-5 + 1) / 2, (2 - 7) / 10])
        np.testing.assert_equal(x2, z2)

    def test_minmaxscaling_scale_unscale__computes_right_values(self):
        N = scaling.MinMaxScaling(
            {"x1": (-5, 2), "x2": (np.array([-1, 7]), np.array([2, 10]))}
        )
        x1 = 4
        y1 = N.scale("x1", x1)
        z1 = N.unscale("x1", y1)
        np.testing.assert_equal(y1, (4 + 5) / (5 + 2))
        np.testing.assert_equal(x1, z1)

        x2 = np.array([-5, 2])
        y2 = N.scale("x2", x2)
        z2 = N.unscale("x2", y2)
        np.testing.assert_equal(y2, [(-5 + 1) / (2 + 1), (2 - 7) / (10 - 7)])
        np.testing.assert_equal(x2, z2)


if __name__ == "__main__":
    unittest.main()
