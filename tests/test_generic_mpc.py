import unittest
import casadi as cs
import numpy as np
from casadi_mpc import GenericMpc
from casadi_mpc.solutions import subsevalf


def random_sym_type() -> str:
    return 'SX' if np.random.rand() > 0.5 else 'MX'


class TestGenericMpc(unittest.TestCase):
    def test_parameter__creates_correct_parameter(self):
        shape1 = (4, 3)
        shape2 = (2, 2)
        for sym_type in ('SX', 'MX'):
            mpc = GenericMpc(sym_type=sym_type)
            p1 = mpc.parameter('p1', shape1)
            p2 = mpc.parameter('p2', shape2)
            self.assertEqual(p1.shape, shape1)
            self.assertEqual(p2.shape, shape2)
            self.assertEqual(mpc.np, np.prod(shape1) + np.prod(shape2))
            p = cs.vertcat(cs.vec(p1), cs.vec(p2))
            if sym_type == 'SX':
                self.assertTrue(cs.is_equal(mpc.p, p))
            else:
                p1_ = np.random.randn(*p1.shape)
                p2_ = np.random.randn(*p2.shape)
                np.testing.assert_allclose(
                    subsevalf(mpc.p, [p1, p2], [p1_, p2_]),
                    subsevalf(p, [p1, p2], [p1_, p2_]),
                )

    def test_parameter__raises__with_parameters_with_same_name(self):
        for sym_type in ('SX', 'MX'):
            mpc = GenericMpc(sym_type=sym_type)
            mpc.parameter('p1')
            with self.assertRaises(ValueError):
                mpc.parameter('p1')

    def test_minimize__sets_objective_correctly(self):
        for sym_type in ('SX', 'MX'):
            x = getattr(cs, sym_type).sym('x', (5, 1))
            f = x.T @ x
            mpc = GenericMpc(sym_type=sym_type)
            mpc.minimize(f)
            self.assertTrue(cs.is_equal(mpc.f, f))


if __name__ == '__main__':
    unittest.main()
