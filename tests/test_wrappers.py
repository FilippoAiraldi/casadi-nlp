import unittest
import casadi as cs
import numpy as np
from casadi_nlp import Nlp
from casadi_nlp.wrappers import Wrapper, DifferentiableNlp
from casadi_nlp.solutions import subsevalf


class TestWrapper(unittest.TestCase):
    def test_unwrapped__unwraps_nlp_correctly(self):
        nlp = Nlp()
        self.assertIs(nlp, nlp.unwrapped)
        wrapped = Wrapper[Nlp](nlp)
        self.assertIs(nlp, wrapped.unwrapped)


class TestDifferentiableNlp(unittest.TestCase):
    def test_h_lbx_ubx__returns_correct_indices(self):
        for flag in (True, False):
            nlp = DifferentiableNlp(Nlp(sym_type='SX'), simplify_x_bounds=flag)

            x1, lam_lbx1, lam_ubx1 = nlp.variable('x1', (2, 1))
            (h_lbx, lam_lbx), (h_ubx, lam_ubx) = nlp.h_lbx, nlp.h_ubx
            if flag:
                self.assertTrue(all(o.is_empty() 
                                for o in [h_lbx, lam_lbx, h_ubx, lam_ubx]))
            else:
                np.testing.assert_allclose(cs.evalf(h_lbx - (-np.inf - x1)), 0)
                np.testing.assert_allclose(cs.evalf(lam_lbx - lam_lbx1), 0)
                np.testing.assert_allclose(cs.evalf(h_ubx - (x1 - np.inf)), 0)
                np.testing.assert_allclose(cs.evalf(lam_ubx - lam_ubx1), 0)

            x2, lam_lbx2, lam_ubx2 = nlp.variable('x2', (2, 1), lb=0)
            (h_lbx, lam_lbx), (h_ubx, lam_ubx) = nlp.h_lbx, nlp.h_ubx
            if flag:
                self.assertTrue(all(o.is_empty() for o in [h_ubx, lam_ubx]))
                np.testing.assert_allclose(cs.evalf(h_lbx - (-x2)), 0)
                np.testing.assert_allclose(cs.evalf(lam_lbx - lam_lbx2), 0)
            else:
                np.testing.assert_allclose(cs.evalf(
                    h_lbx - cs.vertcat(-np.inf - x1, 0 - x2)), 0)
                np.testing.assert_allclose(cs.evalf(
                    lam_lbx - cs.vertcat(lam_lbx1, lam_lbx2)), 0)
                np.testing.assert_allclose(cs.evalf(
                    h_ubx - cs.vertcat(x1 - np.inf, x2 - np.inf)), 0)
                np.testing.assert_allclose(cs.evalf(
                    lam_ubx - cs.vertcat(lam_ubx1, lam_ubx2)), 0)
            
            x3, lam_lbx3, lam_ubx3 = nlp.variable('x3', (2, 1), ub=1)
            (h_lbx, lam_lbx), (h_ubx, lam_ubx) = nlp.h_lbx, nlp.h_ubx
            if flag:
                np.testing.assert_allclose(cs.evalf(h_lbx - (-x2)), 0)
                np.testing.assert_allclose(cs.evalf(lam_lbx - lam_lbx2), 0)
                np.testing.assert_allclose(cs.evalf(h_ubx - (x3 - 1)), 0)
                np.testing.assert_allclose(cs.evalf(lam_ubx - lam_ubx3), 0)
            else:
                np.testing.assert_allclose(cs.evalf(
                    h_lbx - cs.vertcat(-np.inf - x1, 0 - x2, -np.inf - x3)), 0)
                np.testing.assert_allclose(cs.evalf(
                    lam_lbx - cs.vertcat(lam_lbx1, lam_lbx2, lam_lbx3)), 0)
                np.testing.assert_allclose(cs.evalf(
                    h_ubx - cs.vertcat(x1 - np.inf, x2 - np.inf, x3 - 1)), 0)
                np.testing.assert_allclose(cs.evalf(
                    lam_ubx - cs.vertcat(lam_ubx1, lam_ubx2, lam_ubx3)), 0)

            x4, lam_lbx4, lam_ubx4 = nlp.variable('x4', (2, 1), lb=0, ub=1)
            (h_lbx, lam_lbx), (h_ubx, lam_ubx) = nlp.h_lbx, nlp.h_ubx
            if flag:
                np.testing.assert_allclose(cs.evalf(
                    h_lbx - cs.vertcat(-x2, -x4)), 0)
                np.testing.assert_allclose(
                    cs.evalf(lam_lbx - cs.vertcat(lam_lbx2, lam_lbx4)), 0)
                np.testing.assert_allclose(cs.evalf(
                    h_ubx - cs.vertcat(x3 - 1, x4 - 1)), 0)
                np.testing.assert_allclose(
                    cs.evalf(lam_ubx - cs.vertcat(lam_ubx3, lam_ubx4)), 0)
            else:
                np.testing.assert_allclose(cs.evalf(
                    h_lbx - cs.vertcat(
                        -np.inf - x1, 0 - x2, -np.inf - x3, 0 - x4)), 0)
                np.testing.assert_allclose(cs.evalf(
                    lam_lbx - cs.vertcat(
                        lam_lbx1, lam_lbx2, lam_lbx3, lam_lbx4)), 0)
                np.testing.assert_allclose(cs.evalf(
                    h_ubx - cs.vertcat(
                        x1 - np.inf, x2 - np.inf, x3 - 1, x4 - 1)), 0)
                np.testing.assert_allclose(cs.evalf(
                    lam_ubx - cs.vertcat(
                        lam_ubx1, lam_ubx2, lam_ubx3, lam_ubx4)), 0)

    def test_lagrangian__is_correct__example_1a_b(self):
        # https://en.wikipedia.org/wiki/Lagrange_multiplier#Example_1a
        # https://en.wikipedia.org/wiki/Lagrange_multiplier#Example_1b
        for sym_type in ('SX', 'MX'):
            nlp = DifferentiableNlp(Nlp(sym_type=sym_type))
            x = nlp.variable('x')[0]
            y = nlp.variable('y')[0]
            _, lam = nlp.constraint('c1', x**2 + y**2, '==', 1)
            for f in [-x - y, (x + y)**2]:
                nlp.minimize(f)
                L = f + lam * (x**2 + y**2 - 1)
                x_ = np.random.randn(*x.shape)
                y_ = np.random.randn(*y.shape)
                lam_ = np.random.randn(*lam.shape)
                np.testing.assert_allclose(
                    subsevalf(nlp.lagrangian, [x, y, lam], [x_, y_, lam_]),
                    subsevalf(L, [x, y, lam], [x_, y_, lam_])
                )

    def test_lagrangian__is_correct__example_3(self):
        # https://en.wikipedia.org/wiki/Lagrange_multiplier#Example_3:_Entropy
        n = 50
        log2 = lambda x: cs.log(x) / cs.log(2)
        lbx = 1 / n * 1e-6
        for sym_type in ('SX', 'MX'):
            nlp = DifferentiableNlp(Nlp(sym_type=sym_type))
            p, lam_lbx, _ = nlp.variable('p', (n, 1), lb=lbx)
            c1, lam_g = nlp.constraint('c1', cs.sum1(p), '==', 1)
            f = cs.sum1(p * log2(p))
            nlp.minimize(f)
            L = f + lam_g * c1 + lam_lbx.T @ (lbx - p)
            p_ = np.random.rand(*p.shape)
            lam_lbx_ = np.random.randn(*lam_lbx.shape)
            lam_g_ = np.random.randn(*lam_g.shape)
            np.testing.assert_allclose(
                subsevalf(nlp.lagrangian,
                          [p, lam_lbx, lam_g], [p_, lam_lbx_, lam_g_]),
                subsevalf(L, [p, lam_lbx, lam_g], [p_, lam_lbx_, lam_g_])
            )

    def test_lagrangian__is_correct__example_5(self):
        # https://personal.math.ubc.ca/~israel/m340/kkt2.pdf
        for sym_type in ('SX', 'MX'):
            nlp = DifferentiableNlp(Nlp(sym_type=sym_type))
            x, lam_lbx, lam_ubx = nlp.variable('x', (2, 1), lb=0, ub=1)
            c1, lam_h = nlp.constraint('c1', x[0] + x[1]**2, '<=', 2)
            f = - x[0] * x[1]
            nlp.minimize(f)
            L = f + lam_h * c1 + lam_lbx.T @ (0 - x) + lam_ubx.T @ (x - 1)
            x_ = np.random.rand(*x.shape)
            lam_lbx_ = np.random.randn(*lam_lbx.shape)
            lam_ubx_ = np.random.randn(*lam_ubx.shape)
            lam_h_ = np.random.randn(*lam_h.shape)
            np.testing.assert_allclose(
                subsevalf(nlp.lagrangian,
                          [x, lam_lbx, lam_ubx, lam_h],
                          [x_, lam_lbx_, lam_ubx_, lam_h_]),
                subsevalf(L, [x, lam_lbx, lam_ubx, lam_h],
                          [x_, lam_lbx_, lam_ubx_, lam_h_])
            )


if __name__ == '__main__':
    unittest.main()
