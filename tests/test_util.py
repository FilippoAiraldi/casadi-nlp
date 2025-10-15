import math as _math
import os
import pickle
import tempfile
import unittest
from copy import deepcopy
from itertools import product
from typing import Optional

import casadi as cs
import cvxpy as cp
import numpy as np
from parameterized import parameterized
from scipy.special import digamma, gammaln
from scipy.stats import norm

from csnlp import Nlp
from csnlp.core.solutions import subsevalf
from csnlp.util import docs, io, math

TMPFILENAME: str = ""


class DictClass:
    def __init__(self) -> None:
        super().__init__()
        self.z, self.w = 3, 4
        self.sym2 = cs.SX.sym("y", 3, 1)


class TestIo(unittest.TestCase):
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
        this = DictClass()
        if copy:
            other = deepcopy(this)
        else:
            with cs.global_pickle_context():
                pickled = pickle.dumps(this)
            with cs.global_unpickle_context():
                other = pickle.loads(pickled)

        self.assertIsNot(this, other)
        self.assertEqual(this.z, other.z)
        self.assertEqual(this.w, other.w)
        if copy:
            self.assertTrue(cs.is_equal(this.sym2, other.sym2))
        else:
            self.assertTupleEqual(this.sym2.shape, other.sym2.shape)


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

    @parameterized.expand(
        [
            (
                5,
                7,
                [
                    1.000000000190015,
                    76.18009172947146,
                    -86.50532032941677,
                    24.01409824083091,
                    -1.231739572450155,
                    0.1208650973866179e-2,
                    -0.5395239384953e-5,
                ],
            ),
            (
                7,
                9,
                [
                    0.99999999999980993227684700473478,
                    676.520368121885098567009190444019,
                    -1259.13921672240287047156078755283,
                    771.3234287776530788486528258894,
                    -176.61502916214059906584551354,
                    12.507343278686904814458936853,
                    -0.13857109526572011689554707,
                    9.984369578019570859563e-6,
                    1.50563273514931155834e-7,
                ],
            ),
            (
                8,
                12,
                [
                    0.9999999999999999298,
                    1975.3739023578852322,
                    -4397.3823927922428918,
                    3462.6328459862717019,
                    -1156.9851431631167820,
                    154.53815050252775060,
                    -6.2536716123689161798,
                    0.034642762454736807441,
                    -7.4776171974442977377e-7,
                    6.3041253821852264261e-8,
                    -2.7405717035683877489e-8,
                    4.0486948817567609101e-9,
                ],
            ),
            (
                4.7421875,
                15,
                [
                    0.99999999999999709182,
                    57.156235665862923517,
                    -59.597960355475491248,
                    14.136097974741747174,
                    -0.49191381609762019978,
                    0.33994649984811888699e-4,
                    0.46523628927048575665e-4,
                    -0.98374475304879564677e-4,
                    0.15808870322491248884e-3,
                    -0.21026444172410488319e-3,
                    0.21743961811521264320e-3,
                    -0.16431810653676389022e-3,
                    0.84418223983852743293e-4,
                    -0.26190838401581408670e-4,
                    0.36899182659531622704e-5,
                ],
            ),
        ]
    )
    def test_godfrey_coefficients(self, g, n, expected):
        actual = math.godfrey_coefficients(g, n).astype(float)
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

    def test_gammaln(self):
        z = cs.SX.sym("z")
        y = math.gammaln(z, 9, 12)
        cs_gammaln = cs.Function("gammaln", [z], [y])

        z = np.linspace(0.0, 10, 10000)
        expected = gammaln(z)
        actual = cs_gammaln(z).full().flatten()
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

    def test_digamma1p(self):
        z = cs.SX.sym("z")
        y = math.digamma1p(z, 3)
        cs_digamma1p = cs.Function("digamma1p", [z], [y])

        z = np.linspace(1e-6, 10, 10_000)
        expected = digamma(z + 1)
        actual = cs_digamma1p(z).full().flatten()
        np.testing.assert_allclose(actual, expected, rtol=1e-2, atol=1e-3)

    def test_digamma(self):
        z = cs.SX.sym("z")
        y = math.digamma(z, 3)
        cs_digamma = cs.Function("digamma", [z], [y])

        z = np.linspace(1e-6, 10, 10_000)
        expected = digamma(z)
        actual = cs_digamma(z).full().flatten()
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)


class TestDocs(unittest.TestCase):
    def test_get_casadi_plugins(self):
        all_plugins = docs.get_casadi_plugins()
        self.assertIsInstance(all_plugins, dict)
        self.assertGreater(len(all_plugins), 0)
        for name, plugins in all_plugins.items():
            self.assertIsInstance(name, str)
            self.assertIsInstance(plugins, list)
            for p in plugins:
                self.assertIsInstance(p, str)

    def test_list_available_solvers(self):
        solvers = docs.list_available_solvers()
        self.assertIsInstance(solvers, dict)
        self.assertIn("nlp", solvers)
        self.assertIn("qp", solvers)
        for key in ("nlp", "qp"):
            self.assertIsInstance(solvers[key], list)
            for solver in solvers[key]:
                self.assertIsInstance(solver, str)
                self.assertGreater(len(solver), 0)

    def test_get_solver_options(self):
        options = docs.get_solver_options("ipopt", False)
        self.assertIsInstance(options, dict)


if __name__ == "__main__":
    unittest.main()
