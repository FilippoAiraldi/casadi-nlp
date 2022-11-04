import unittest
import casadi as cs
import numpy as np
from scipy.stats import norm
import casadi_mpc.math as csmath
from casadi_mpc.solutions import subsevalf


class TestMath(unittest.TestCase):
    def test_prod(self):
        shape = (4, 5)
        for ax in (-2, -1, 0, 1, None):
            x_sx = cs.SX.sym('X', *shape)
            x_mx = cs.MX.sym('X', *shape)
            x = np.random.randn(*shape) * 2
            p_sx = subsevalf(csmath.prod(x_sx, axis=ax), x_sx, x)
            p_mx = subsevalf(csmath.prod(x_mx, axis=ax), x_mx, x)
            p = np.prod(x, axis=ax, keepdims=True)

            np.testing.assert_allclose(p, p_sx)
            np.testing.assert_allclose(p, p_mx)

    def test_quad_form(self):
        n = 5
        for m in (1, n):
            x_sx = cs.SX.sym('X', n, 1)
            x_mx = cs.MX.sym('X', n, 1)
            A_sx = cs.SX.sym('A', n, m)
            A_mx = cs.MX.sym('A', n, m)
            x = np.random.randn(n, 1) * 2
            A = np.random.randn(n, m) * 2
            p_sx = subsevalf(
                csmath.quad_form(A_sx, x_sx), [x_sx, A_sx], [x, A])
            p_mx = subsevalf(
                csmath.quad_form(A_mx, x_mx), [x_mx, A_mx], [x, A])
            p = x.T @ (np.diag(A.flat) if m == 1 else A) @ x

            np.testing.assert_allclose(p, p_sx)
            np.testing.assert_allclose(p, p_mx)

    def test_norm_cdf(self):
        shape = (3, 4)
        x_sx = cs.SX.sym('X', *shape)
        loc_sx = cs.SX.sym('loc', *shape)
        scale_sx = cs.SX.sym('scale', *shape)
        x_mx = cs.MX.sym('X', *shape)
        loc_mx = cs.MX.sym('loc', *shape)
        scale_mx = cs.MX.sym('scale', *shape)
        x = np.random.randn(*shape)
        loc = np.random.randn(*shape)
        scale = np.random.rand(*shape)
        cdf_sx = subsevalf(csmath.norm_cdf(x_sx, loc=loc_sx, scale=scale_sx),
                           [x_sx, loc_sx, scale_sx], [x, loc, scale])
        cdf_mx = subsevalf(csmath.norm_cdf(x_mx, loc=loc_mx, scale=scale_mx),
                           [x_mx, loc_mx, scale_mx], [x, loc, scale])
        cdf = norm.cdf(x, loc=loc, scale=scale)

        np.testing.assert_allclose(cdf, cdf_sx, atol=1e-7, rtol=1e-5)
        np.testing.assert_allclose(cdf, cdf_mx, atol=1e-7, rtol=1e-5)

    def test_norm_ppf(self):
        shape = (3, 4)
        x_sx = cs.SX.sym('X', *shape)
        loc_sx = cs.SX.sym('loc', *shape)
        scale_sx = cs.SX.sym('scale', *shape)
        x_mx = cs.MX.sym('X', *shape)
        loc_mx = cs.MX.sym('loc', *shape)
        scale_mx = cs.MX.sym('scale', *shape)
        x = np.random.rand(*shape)
        loc = np.random.randn(*shape)
        scale = np.random.rand(*shape)
        cdf_sx = subsevalf(csmath.norm_ppf(x_sx, loc=loc_sx, scale=scale_sx),
                           [x_sx, loc_sx, scale_sx], [x, loc, scale])
        cdf_mx = subsevalf(csmath.norm_ppf(x_mx, loc=loc_mx, scale=scale_mx),
                           [x_mx, loc_mx, scale_mx], [x, loc, scale])
        cdf = norm.ppf(x, loc=loc, scale=scale)

        np.testing.assert_allclose(cdf, cdf_sx, atol=1e-7, rtol=1e-5)
        np.testing.assert_allclose(cdf, cdf_mx, atol=1e-7, rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
