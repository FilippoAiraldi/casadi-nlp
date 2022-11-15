import unittest
from itertools import product
import casadi as cs
import numpy as np
from csnlp import Nlp
from csnlp.wrappers import Wrapper, NlpSensitivity
from csnlp.solutions import subsevalf
from csnlp.math import log


OPTS = {
    'expand': True, 'print_time': False,
    'ipopt': {
        'max_iter': 500,
        'tol': 1e-7,
        'barrier_tol_factor': 5,
        'sb': 'yes',
        # for debugging
        'print_level': 0,
        'print_user_options': 'no',
        'print_options_documentation': 'no'
    }
}


class TestWrapper(unittest.TestCase):
    def test_unwrapped__unwraps_nlp_correctly(self):
        nlp = Nlp()
        self.assertIs(nlp, nlp.unwrapped)
        wrapped = Wrapper[Nlp](nlp)
        self.assertIs(nlp, wrapped.unwrapped)


class TestNlpSensitivity(unittest.TestCase):
    def test_lagrangian__is_correct__example_1a_b(self):
        # https://en.wikipedia.org/wiki/Lagrange_multiplier#Example_1a
        # https://en.wikipedia.org/wiki/Lagrange_multiplier#Example_1b
        for sym_type in ('SX', 'MX'):
            nlp = NlpSensitivity(Nlp(sym_type=sym_type))
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
        lbx = 1 / n * 1e-6
        for sym_type in ('SX', 'MX'):
            nlp = NlpSensitivity(Nlp(sym_type=sym_type))
            p, lam_lbx, _ = nlp.variable('p', (n, 1), lb=lbx)
            c1, lam_g = nlp.constraint('c1', cs.sum1(p), '==', 1)
            f = cs.sum1(p * log(p, 2))
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
            nlp = NlpSensitivity(Nlp(sym_type=sym_type))
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

    def test_kkt__returns_tau_correctly(self):
        for sym_type, flag in product(('MX', 'SX'), (True, False)):
            nlp = NlpSensitivity(
                Nlp(sym_type=sym_type), include_barrier_term=flag)
            x = nlp.variable('x')[0]
            nlp.constraint('c', x, '<=', 1)
            nlp.minimize(x)
            _, tau = nlp.kkt
            if flag:
                self.assertIsInstance(tau, nlp._csXX)
            else:
                self.assertIsNone(tau)

    def test_kkt__computes_kkt_conditions_correctly__example_1a(self):
        # https://en.wikipedia.org/wiki/Lagrange_multiplier#Example_1a
        for sym_type in ('MX', 'SX'):
            nlp = NlpSensitivity(Nlp(sym_type=sym_type))
            x = nlp.variable('x')[0]
            y = nlp.variable('y')[0]
            nlp.constraint('c1', x**2 + y**2, '==', 1)
            nlp.minimize(-x - y)
            nlp.init_solver(OPTS)
            sol = nlp.solve()
            kkt, _ = nlp.kkt
            np.testing.assert_allclose(sol.value(kkt), 0, atol=1e-9)

    def test_kkt__computes_kkt_conditions_correctly__example_1b(self):
        # https://en.wikipedia.org/wiki/Lagrange_multiplier#Example_1b
        for sym_type in ('SX', 'MX'):
            nlp = NlpSensitivity(Nlp(sym_type=sym_type))
            x = nlp.variable('x')[0]
            y = nlp.variable('y')[0]
            nlp.constraint('c1', x**2 + y**2, '==', 1)
            nlp.minimize((x + y)**2)
            nlp.init_solver(OPTS)
            sol = nlp.solve(vals0={
                'x': 0.5,
                'y': np.sqrt(1 - 0.5**2)
            })
            kkt, _ = nlp.kkt
            np.testing.assert_allclose(sol.value(kkt), 0, atol=1e-6)

    def test_kkt__computes_kkt_conditions_correctly__example_2(self):
        # https://en.wikipedia.org/wiki/Lagrange_multiplier#Example_2
        for sym_type in ('SX', 'MX'):
            nlp = NlpSensitivity(Nlp(sym_type=sym_type))
            x = nlp.variable('x')[0]
            y = nlp.variable('y')[0]
            nlp.constraint('c1', x**2 + y**2, '==', 3)
            nlp.minimize(x**2 * y)
            nlp.init_solver(OPTS)
            sol = nlp.solve(vals0={
                'x': np.sqrt(3 - 0.8**2),
                'y': -0.8
            })
            kkt, _ = nlp.kkt
            np.testing.assert_allclose(sol.value(kkt), 0, atol=1e-6)

    def test_kkt__computes_kkt_conditions_correctly__example_3(self):
        # https://en.wikipedia.org/wiki/Lagrange_multiplier#Example_3:_Entropy
        n = 50
        for sym_type in ('SX', 'MX'):
            nlp = NlpSensitivity(Nlp(sym_type=sym_type))
            p = nlp.variable('p', (n, 1), lb=1 / n * 1e-6)[0]
            nlp.constraint('c1', cs.sum1(p), '==', 1)
            nlp.minimize(cs.sum1(p * log(p, 2)))
            nlp.init_solver(OPTS)
            sol = nlp.solve(vals0={'p': np.random.rand(n)})
            kkt, tau = nlp.kkt
            kkt = subsevalf(kkt, tau, sol.barrier_parameter, eval=False)
            np.testing.assert_allclose(sol.value(kkt), 0, atol=1e-6)

    def test_kkt__computes_kkt_conditions_correctly__example_4(self):
        # https://en.wikipedia.org/wiki/Lagrange_multiplier#Example_4:_Numerical_optimization
        for sym_type in ('SX', 'MX'):
            nlp = NlpSensitivity(Nlp(sym_type=sym_type))
            x = nlp.variable('x')[0]
            nlp.constraint('c1', x**2, '==', 1)
            nlp.minimize(x**2)
            nlp.init_solver(OPTS)
            sol = nlp.solve(vals0={'x': 1 + 0.1 * np.random.rand()})
            kkt, _ = nlp.kkt
            np.testing.assert_allclose(sol.value(kkt), 0, atol=1e-6)

    def test_kkt__computes_kkt_conditions_correctly__example_5(self):
        # https://personal.math.ubc.ca/~israel/m340/kkt2.pdf
        for sym_type in ('SX', 'MX'):
            nlp = NlpSensitivity(Nlp(sym_type=sym_type))
            x = nlp.variable('x', (2, 1), lb=0)[0]
            nlp.constraint('c1', x[0] + x[1]**2, '<=', 2)
            nlp.minimize(- x[0] * x[1])
            nlp.init_solver(OPTS)
            sol = nlp.solve(vals0={'x': [1, 1]})
            kkt, tau = nlp.kkt
            kkt = subsevalf(kkt, tau, sol.barrier_parameter, eval=False)
            np.testing.assert_allclose(sol.value(kkt), 0, atol=1e-7)

    def test_kkt__computes_sensitivity_correctly__example_7(self):
        #  Example 4.2 from [1]
        #
        # References
        # ----------
        # [1] Buskens, C. and Maurer, H. (2001). Sensitivity analysis and
        #     real-time optimization of parametric nonlinear programming
        #     problems. In M. Grotschel, S.O. Krumke, and J. Rambau (eds.),
        #     Online Optimization of Large Scale Systems, 3–16. Springer,
        #     Berlin, Heidelberg.
        for sym_type in ('SX', 'MX'):
            nlp = NlpSensitivity(Nlp(sym_type=sym_type))
            z = nlp.variable('z', (2, 1))[0]
            p = nlp.parameter('p')
            nlp.minimize(-((0.5 + p) * cs.sqrt(z[0]) + (0.5 - p) * z[1]))
            _, lam = nlp.constraint('c1', cs.sum1(z), '<=', 1)
            nlp.constraint('c2', z[0], '>=', 0.1)
            nlp.init_solver(OPTS)
            sol = nlp.solve(pars={'p': 0}, vals0={'z': [0.1, 0.9]})

            np.testing.assert_allclose(
                sol.vals['z'], [[0.25], [0.75]], rtol=1e-5)
            np.testing.assert_allclose(sol.value(lam), 0.5)

            kkt, tau = nlp.kkt
            kkt = subsevalf(kkt, tau, sol.barrier_parameter, eval=False)
            np.testing.assert_allclose(sol.value(kkt), 0, atol=1e-7)

            S1 = sol.value(nlp.parametric_sensitivity()[0]).full().flatten()
            S2 = nlp.parametric_sensitivity(solution=sol)[0].flatten()
            for S in (S1, S2):
                np.testing.assert_allclose(S, [2, -2, -1, 0], atol=1e-5)

    def test_kkt__computes_sensitivity_correctly__example_8(self):
        #  Example 4.5 from [1]
        #
        # References
        # ----------
        # [1] Buskens, C. and Maurer, H. (2001). Sensitivity analysis and
        #     real-time optimization of parametric nonlinear programming
        #     problems. In M. Grotschel, S.O. Krumke, and J. Rambau (eds.),
        #     Online Optimization of Large Scale Systems, 3–16. Springer,
        #     Berlin, Heidelberg.
        for sym_type in ('SX', 'MX'):
            nlp = NlpSensitivity(Nlp(sym_type=sym_type))
            z = nlp.variable('z', (2, 1))[0]
            p = nlp.parameter('p')
            nlp.minimize(cs.sumsqr(z + [1, -2]))
            _, lam1 = nlp.constraint('c1', -z[0] + p, '<=', 0)
            _, lam2 = nlp.constraint('c2', 2 * z[0] + z[1], '<=', 6)
            nlp.init_solver(OPTS)
            sol = nlp.solve(pars={'p': 1})

            np.testing.assert_allclose(sol.f, 4)
            np.testing.assert_allclose(sol.vals['z'], [[1], [2]], rtol=1e-7)
            np.testing.assert_allclose(sol.value(lam1), 4, atol=1e-7)
            np.testing.assert_allclose(sol.value(lam2), 0, atol=1e-7)

            kkt, tau = nlp.kkt
            kkt = subsevalf(kkt, tau, sol.barrier_parameter, eval=False)
            np.testing.assert_allclose(sol.value(kkt), 0, atol=1e-7)

            S1 = nlp.parametric_sensitivity()[0]
            S2 = nlp.parametric_sensitivity(solution=sol)[0]
            np.testing.assert_allclose(
                sol.value(S1).full().flat, [1, 0, 2, 0], atol=1e-5)
            np.testing.assert_allclose(S2.flat, [1, 0, 2, 0], atol=1e-5)

            Lp = nlp.jacobians['L-p']
            Lpp = nlp.hessians['L-pp']
            Lpz = nlp.hessians['L-px']
            Lzz = nlp.hessians['L-xx']
            np.testing.assert_allclose(sol.value(Lp), 4, atol=1e-7)

            dzdp = S1[:nlp.nx]
            d2Fdp2 = dzdp.T @ Lzz @ dzdp + 2 * (Lpz @ dzdp).T + Lpp
            np.testing.assert_allclose(sol.value(d2Fdp2), 2, atol=1e-7)
            dzdp = S2[:nlp.nx, None]
            d2Fdp2 = dzdp.T @ Lzz @ dzdp + 2 * (Lpz @ dzdp).T + Lpp
            np.testing.assert_allclose(sol.value(d2Fdp2), 2, atol=1e-7)

    def test_licq__computes_qualification_correctly__example_1(self):
        # https://de.wikipedia.org/wiki/Linear_independence_constraint_qualification#LICQ
        for sym_type in ('SX', 'MX'):
            nlp = NlpSensitivity(Nlp(sym_type=sym_type))
            x = nlp.variable('x', (2, 1))[0]
            nlp.constraint('c1', cs.sum1(x), '<=', 1)
            nlp.constraint('c2', cs.sumsqr(x), '<=', 1)
            nlp.minimize(x[0]**2 + (x[1] - 1)**2)  # any objective

            x_ = cs.DM([[0], [1]])
            d = subsevalf(nlp.licq, x, x_).full()
            np.testing.assert_allclose(d, [[1, 1], [0, 2]])
            self.assertEqual(np.linalg.matrix_rank(d), d.shape[0])

    def test_licq__computes_qualification_correctly__example_2(self):
        # https://de.wikipedia.org/wiki/Linear_independence_constraint_qualification#MFCQ_ohne_LICQ
        for sym_type in ('SX', 'MX'):
            nlp = NlpSensitivity(Nlp(sym_type=sym_type))
            x = nlp.variable('x', (2, 1))[0]
            nlp.constraint('c1', x[1], '>=', 0)
            nlp.constraint('c2', x[0]**4 - x[1], '<=', 1)
            nlp.minimize(x[0]**2 + x[1]**2)  # any objective

            x_ = cs.DM([[0], [0]])
            d = subsevalf(nlp.licq, x, x_).full()
            np.testing.assert_allclose(d, [[0, -1], [0, -1]])
            self.assertNotEqual(np.linalg.matrix_rank(d), d.shape[0])

    def test_parametric_sensitivity__is_correct(self):
        # https://web.casadi.org/blog/nlp_sens/

        p_values_and_solutions = [
            (1.25, (-0.442401495488, 0.400036517837, 3.516850030791)),
            (1.4, (-0.350853791920, 0.766082818877, 1.401822559490)),
            (2, (-0.000000039328, 0, 0))
        ]

        def z(x):
            return x[1, :] - x[0, :]

        for sym_type in ('SX', 'MX'):
            nlp = NlpSensitivity(Nlp(sym_type=sym_type))
            x = nlp.variable('x', (2, 1), lb=[[0], [-np.inf]])[0]
            p = nlp.parameter('p', (2, 1))
            nlp.minimize((1 - x[0])**2 + p[0] * (x[1] - x[0]**2)**2)
            g = (x[0] + 0.5)**2 + x[1]**2
            nlp.constraint('c1', (p[1] / 2)**2, '<=', g)
            nlp.constraint('c2', g, '<=', p[1]**2)
            nlp.init_solver(OPTS)

            Z1_ = z(x)
            J1_, H1_ = nlp.parametric_sensitivity(expr=Z1_)
            for p, (Z, J, H) in p_values_and_solutions:
                sol = nlp.solve(pars={'p': [0.2, p]})
                Z1 = sol.value(Z1_)
                J1 = sol.value(J1_)
                H1 = sol.value(H1_)
                J2, H2 = nlp.parametric_sensitivity(expr=Z1_, solution=sol)
                np.testing.assert_allclose(J1, J2, atol=1e-7)
                np.testing.assert_allclose(H1, H2, atol=1e-7)
                np.testing.assert_allclose(Z, Z1, atol=1e-4)
                np.testing.assert_allclose(J, J1[1], atol=1e-4)
                np.testing.assert_allclose(J, J2[1], atol=1e-4)
                np.testing.assert_allclose(H, H1[1, 1], atol=1e-4)
                np.testing.assert_allclose(H, H2[1, 1], atol=1e-4)


if __name__ == '__main__':
    unittest.main()
