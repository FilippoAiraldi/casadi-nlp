import unittest
from itertools import product
import casadi as cs
import numpy as np
from casadi_mpc import GenericMpc
from casadi_mpc.solutions import subsevalf


OPTS = {
    'expand': True, 'print_time': False,
    'ipopt': {
        'max_iter': 500,
        'tol': 1e-6,
        'barrier_tol_factor': 1,
        'sb': 'yes',
        # for debugging
        'print_level': 0,
        'print_user_options': 'no',
        'print_options_documentation': 'no'
    }
}


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

    def test_constraint__raises__with_constraints_with_same_name(self):
        for sym_type in ('SX', 'MX'):
            mpc = GenericMpc(sym_type=sym_type)
            x = mpc.variable('x')[0]
            mpc.constraint('c1', x, '<=', 5)
            with self.assertRaises(ValueError):
                mpc.constraint('c1', x, '<=', 5)

    def test_constraint__raises__with_unknown_operator(self):
        for sym_type in ('SX', 'MX'):
            mpc = GenericMpc(sym_type=sym_type)
            x = mpc.variable('x')[0]
            for op in ['=', '>', '<']:
                with self.assertRaises(ValueError):
                    mpc.constraint('c1', x, op, 5)

    def test_constraint__raises__with_nonsymbolic_terms(self):
        for sym_type in ('SX', 'MX'):
            mpc = GenericMpc(sym_type=sym_type)
            with self.assertRaises(TypeError):
                mpc.constraint('c1', 5, '==', 5)

    def test_constraint__creates_constraint_correctly(self):
        shape1 = (4, 3)
        shape2 = (2, 2)
        nc = np.prod(shape1) + np.prod(shape2)
        for sym_type, op in product(['SX', 'MX'], ['==', '>=']):
            mpc = GenericMpc(sym_type=sym_type)
            x = mpc.variable('x', shape1)[0]
            y = mpc.variable('y', shape2)[0]
            e1, lam1 = mpc.constraint('c1', x, op, 5)
            e2, lam2 = mpc.constraint('c2', 5, op, y)
            self.assertTrue(e1.shape == lam1.shape == shape1)
            self.assertTrue(e2.shape == lam2.shape == shape2)
            grp = 'g' if op == '==' else 'h'
            self.assertEqual(getattr(mpc, f'n{grp}'), nc)

            i = 0
            describe = getattr(mpc.debug, f'{grp}_describe')
            for name, shape in [('c1', shape1), ('c2', shape2)]:
                for _ in range(np.prod(shape)):
                    self.assertEqual(name, describe(i).name)
                    self.assertEqual(shape, describe(i).shape)
                    i += 1
            with self.assertRaises(IndexError):
                describe(nc + 1)

            e = cs.vertcat(cs.vec(e1), cs.vec(e2))
            lam = cs.vertcat(cs.vec(lam1), cs.vec(lam2))
            if sym_type == 'SX':
                # symbolically for SX
                self.assertTrue(cs.is_equal(getattr(mpc, grp), e))
                self.assertTrue(cs.is_equal(getattr(mpc, f'lam_{grp}'), lam))
            else:
                # only numerically for MX
                x_ = np.random.randn(*shape1)
                y_ = np.random.randn(*shape2)
                np.testing.assert_allclose(
                    subsevalf(getattr(mpc, grp), [x, y], [x_, y_]),
                    subsevalf(e, [x, y], [x_, y_])
                )
                lam1_ = np.random.randn(*shape1)
                lam2_ = np.random.randn(*shape2)
                np.testing.assert_allclose(
                    subsevalf(getattr(mpc, f'lam_{grp}'),
                              [lam1, lam2], [lam1_, lam2_]),
                    subsevalf(lam, [lam1, lam2], [lam1_, lam2_])
                )

    def test_set_solver__saves_options_correctly(self):
        for sym_type in ('SX', 'MX'):
            mpc = GenericMpc(sym_type=sym_type)
            opts = OPTS.copy()
            x = mpc.variable('x')[0]
            mpc.minimize(x**2)
            mpc.init_solver(opts)
            self.assertDictEqual(opts, mpc.solver_opts)

    def test_solve__raises__with_uninit_solver(self):
        for sym_type in ('SX', 'MX'):
            mpc = GenericMpc(sym_type=sym_type)
            with self.assertRaises(RuntimeError):
                mpc.solve(None)

    def test_solve__raises__with_free_parameters(self):
        for sym_type in ('SX', 'MX'):
            mpc = GenericMpc(sym_type=sym_type)
            x = mpc.variable('x')[0]
            p = mpc.parameter('p')
            mpc.f = p * (x**2)
            with self.assertRaises(RuntimeError):
                mpc.solve({})

    def test_solve__computes_correctly__small_example_1(self):
        for sym_type in ('MX', 'SX'):
            mpc = GenericMpc(sym_type=sym_type)
            x = mpc.variable('x', (2, 1))[0]
            y = mpc.variable('y', (3, 1))[0]
            p = mpc.parameter('p')
            mpc.minimize(p + (x.T @ x + y.T @ y))
            mpc.init_solver(OPTS)
            sol = mpc.solve({'p': 3})
            self.assertTrue(sol.success)
            np.testing.assert_allclose(sol.f, 3)
            for k in sol.vals.keys():
                np.testing.assert_allclose(sol.vals[k], 0)
            o = sol.value(p + (x.T @ x + y.T @ y))
            np.testing.assert_allclose(sol.f, o)


if __name__ == '__main__':
    unittest.main()
