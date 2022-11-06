import unittest
import casadi as cs
import numpy as np
from casadi_mpc import GenericMpc
from casadi_mpc.solutions import subsevalf


def random_sym_type() -> str:
    return 'SX' if np.random.rand() > 0.5 else 'MX'


class TestGenericMpc(unittest.TestCase):
    def test_init__raises__with_invalid_sym_type(self):
        with self.assertRaises(Exception):
            GenericMpc(sym_type='a_random_sym_type')

    def test_parameter__creates_correct_parameter(self):
        shape1 = (4, 3)
        shape2 = (2, 2)
        _np = np.prod(shape1) + np.prod(shape2)
        for sym_type in ('SX', 'MX'):
            mpc = GenericMpc(sym_type=sym_type)
            p1 = mpc.parameter('p1', shape1)
            p2 = mpc.parameter('p2', shape2)
            self.assertEqual(p1.shape, shape1)
            self.assertEqual(p2.shape, shape2)
            self.assertEqual(mpc.np, _np)

            p = cs.vertcat(cs.vec(p1), cs.vec(p2))
            if sym_type == 'SX':
                # symbolically for SX
                self.assertTrue(cs.is_equal(mpc.p, p))
            else:
                # only numerically for MX
                p1_ = np.random.randn(*p1.shape)
                p2_ = np.random.randn(*p2.shape)
                np.testing.assert_allclose(
                    subsevalf(mpc.p, [p1, p2], [p1_, p2_]),
                    subsevalf(p, [p1, p2], [p1_, p2_]),
                )

            i = 0
            for name, shape in [('p1', shape1), ('p2', shape2)]:
                for _ in range(np.prod(shape)):
                    self.assertEqual(name, mpc.debug.p_describe(i).name)
                    self.assertEqual(shape, mpc.debug.p_describe(i).shape)
                    i += 1
            with self.assertRaises(IndexError):
                mpc.debug.p_describe(_np + 1)

            self.assertTrue(cs.is_equal(mpc.parameters['p1'], p1))
            self.assertTrue(cs.is_equal(mpc.parameters['p2'], p2))

    def test_parameter__raises__with_parameters_with_same_name(self):
        for sym_type in ('SX', 'MX'):
            mpc = GenericMpc(sym_type=sym_type)
            mpc.parameter('p')
            with self.assertRaises(ValueError):
                mpc.parameter('p')

    def test_variable__creates_correct_variable(self):
        shape1 = (4, 3)
        shape2 = (2, 2)
        lb1, ub1 = np.random.rand(*shape1) - 1, np.random.rand(*shape1) + 1
        lb2, ub2 = np.random.rand(*shape2) - 1, np.random.rand(*shape2) + 1
        nx = np.prod(shape1) + np.prod(shape2)
        for sym_type in ('SX', 'MX'):
            mpc = GenericMpc(sym_type=sym_type)
            x1, lam1_lb, lam1_ub = mpc.variable('x1', shape1, lb=lb1, ub=ub1)
            x2, lam2_lb, lam2_ub = mpc.variable('x2', shape2, lb=lb2, ub=ub2)
            for o in (x1, lam1_lb, lam1_ub):
                self.assertEqual(o.shape, shape1)
            for o in (x2, lam2_lb, lam2_ub):
                self.assertEqual(o.shape, shape2)
            self.assertEqual(mpc.nx, nx)

            x = cs.vertcat(cs.vec(x1), cs.vec(x2))
            if sym_type == 'SX':
                # symbolically for SX
                self.assertTrue(cs.is_equal(mpc.x, x))
            else:
                # only numerically for MX
                x1_ = np.random.randn(*x1.shape)
                x2_ = np.random.randn(*x2.shape)
                np.testing.assert_allclose(
                    subsevalf(mpc.x, [x1, x2], [x1_, x2_]),
                    subsevalf(x, [x1, x2], [x1_, x2_]),
                )
            lb = cs.vertcat(cs.vec(lb1), cs.vec(lb2))
            ub = cs.vertcat(cs.vec(ub1), cs.vec(ub2))
            np.testing.assert_allclose(mpc.lbx, lb.full().flat)
            np.testing.assert_allclose(mpc.ubx, ub.full().flat)

            i = 0
            for name, shape in [('x1', shape1), ('x2', shape2)]:
                for _ in range(np.prod(shape)):
                    self.assertEqual(name, mpc.debug.x_describe(i).name)
                    self.assertEqual(shape, mpc.debug.x_describe(i).shape)
                    i += 1
            with self.assertRaises(IndexError):
                mpc.debug.x_describe(nx + 1)

            self.assertTrue(cs.is_equal(mpc.variables['x1'], x1))
            self.assertTrue(cs.is_equal(mpc.variables['x2'], x2))

    def test_variable__raises__with_variables_with_same_name(self):
        for sym_type in ('SX', 'MX'):
            mpc = GenericMpc(sym_type=sym_type)
            mpc.variable('x')
            with self.assertRaises(ValueError):
                mpc.variable('x')

    def test_variable__raises__with_invalid_bounds(self):
        for sym_type in ('SX', 'MX'):
            mpc = GenericMpc(sym_type=sym_type)
            with self.assertRaises(ValueError):
                mpc.variable('x', lb=1, ub=0)

    def test_minimize__sets_objective_correctly(self):
        for sym_type in ('SX', 'MX'):
            x = getattr(cs, sym_type).sym('x', (5, 1))
            f = x.T @ x
            mpc = GenericMpc(sym_type=sym_type)
            mpc.minimize(f)
            self.assertTrue(cs.is_equal(mpc.f, f))


if __name__ == '__main__':
    unittest.main()
