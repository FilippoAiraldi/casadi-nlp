import pickle
import unittest
from itertools import product
from math import ceil, floor
from random import choice
from unittest.mock import Mock

import casadi as cs
import numpy as np
from parameterized import parameterized

from csnlp import Nlp
from csnlp.core.solutions import subsevalf
from csnlp.wrappers import Mpc, PwaMpc, PwaRegion
from csnlp.wrappers import MultiScenarioMpc as MSMPC
from csnlp.wrappers import ScenarioBasedMpc as SCMPC
from csnlp.wrappers.mpc.scenario_based_mpc import _n

OPTS = {
    "expand": True,
    "print_time": False,
    "ipopt": {
        "max_iter": 500,
        "tol": 1e-7,
        "barrier_tol_factor": 5,
        "sb": "yes",
        # for debugging
        "print_level": 0,
        "print_user_options": "no",
        "print_options_documentation": "no",
    },
}


class TestMpc(unittest.TestCase):
    def test_init__raises__with_invalid_args(self):
        nlp = Nlp()
        with self.assertRaisesRegex(ValueError, "Invalid shooting method."):
            Mpc(nlp, shooting="ciao", prediction_horizon=1)
        with self.assertRaisesRegex(
            ValueError, "Prediction horizon must be positive and > 0."
        ):
            Mpc(nlp, prediction_horizon=0)
        with self.assertRaisesRegex(
            ValueError, "Control horizon must be positive and > 0."
        ):
            Mpc(nlp, prediction_horizon=10, control_horizon=-2)

    def test_init__initializes_control_horizon_properly(self):
        N = 10
        nlp = Nlp()
        mpc1 = Mpc[cs.SX](nlp, prediction_horizon=N)
        mpc2 = Mpc[cs.SX](nlp, prediction_horizon=N, control_horizon=N * 2)
        self.assertEqual(mpc1.control_horizon, N)
        self.assertEqual(mpc2.control_horizon, N * 2)

    @parameterized.expand([("single",), ("multi",)])
    def test_state__constructs_state_correctly(self, shooting: str):
        N = 10
        nlp = Nlp(sym_type="MX")
        mpc = Mpc[cs.MX](nlp=nlp, prediction_horizon=N, shooting=shooting)
        x1, x1_0 = mpc.state("x1", 2)
        if shooting == "multi":
            self.assertEqual(x1.shape, (2, N + 1))
            self.assertEqual(x1.shape, mpc.states["x1"].shape)
            self.assertEqual(mpc.constraints["x1_0"].shape, (2, 1))
        else:
            self.assertIsNone(x1)
        self.assertEqual(x1_0.shape, (2, 1))
        self.assertEqual(x1_0.shape, mpc.initial_states["x1_0"].shape)
        self.assertEqual(mpc.ns, x1_0.shape[0])
        x2, x2_0 = mpc.state("x2", 1)
        if shooting == "multi":
            self.assertEqual(x2.shape, (1, N + 1))
            self.assertEqual(x2.shape, mpc.states["x2"].shape)
            self.assertEqual(mpc.constraints["x2_0"].shape, (1, 1))
        else:
            self.assertIsNone(x2)
        self.assertEqual(x2_0.shape, (1, 1))
        self.assertEqual(x2_0.shape, mpc.initial_states["x2_0"].shape)
        self.assertEqual(mpc.ns, x1_0.shape[0] + x2_0.shape[0])

    @parameterized.expand(product((False, True), (False, True)))
    def test_state__removes_bounds_properly(self, initial: bool, terminal: bool):
        N = 10
        nlp = Nlp(sym_type="MX")
        mpc = Mpc[cs.MX](nlp=nlp, prediction_horizon=N, shooting="multi")

        shape = (2, N + 1)
        nlp.variable = Mock(return_value=cs.MX.sym("x", *shape))
        mpc.state("x", 2, lb=0, ub=1, bound_initial=initial, bound_terminal=terminal)

        lb, ub = np.full(shape, 0.0), np.full(shape, 1.0)
        if not initial:
            lb[:, 0], ub[:, 0] = -np.inf, np.inf
        if not terminal:
            lb[:, -1], ub[:, -1] = -np.inf, np.inf
        nlp.variable.assert_called_once()
        lb_actual, ub_actual = nlp.variable.call_args[0][-2:]
        np.testing.assert_array_equal(lb_actual, lb)
        np.testing.assert_array_equal(ub_actual, ub)

    @parameterized.expand([(0,), (1,), (2,)])
    def test_state__raises__in_singleshooting_with_state_bounds(self, i: int):
        nlp = Nlp(sym_type="MX")
        mpc = Mpc[cs.MX](nlp=nlp, prediction_horizon=10, shooting="single")
        with self.assertRaises(RuntimeError):
            if i == 0:
                mpc.state("x1", 2, lb=0)
            elif i == 1:
                mpc.state("x1", 2, ub=1)
            else:
                mpc.state("x1", 2, lb=0, ub=1)

    @parameterized.expand(product((1, 2), (1, 3)))
    def test_action__constructs_action_correctly(self, divider: int, space: int):
        Np = 10
        Nc = Np // divider
        nlp = Nlp(sym_type="SX")
        mpc = Mpc[cs.SX](
            nlp=nlp, prediction_horizon=Np, control_horizon=Nc, input_spacing=space
        )
        u1, u1_exp = mpc.action("u1", 2)
        self.assertEqual(u1.shape, (2, ceil(Nc / space)))
        self.assertEqual(u1.shape, mpc.actions["u1"].shape)
        self.assertEqual(u1_exp.shape, (2, Np))
        self.assertEqual(u1_exp.shape, mpc.actions_expanded["u1"].shape)
        self.assertEqual(mpc.na, u1.shape[0])
        u2, u2_exp = mpc.action("u2", 1)
        self.assertEqual(u2.shape, (1, ceil(Nc / space)))
        self.assertEqual(u2.shape, mpc.actions["u2"].shape)
        self.assertEqual(u2_exp.shape, (1, Np))
        self.assertEqual(u2_exp.shape, mpc.actions_expanded["u2"].shape)
        for i in range(Nc):
            self.assertTrue(cs.is_equal(u1[:, floor(i / space)], u1_exp[:, i]))
            self.assertTrue(cs.is_equal(u2[:, floor(i / space)], u2_exp[:, i]))
        for i in range(Nc - 1, Np):
            self.assertTrue(cs.is_equal(u1[:, -1], u1_exp[:, i]))
            self.assertTrue(cs.is_equal(u2[:, -1], u2_exp[:, i]))
        self.assertEqual(mpc.na, u1.shape[0] + u2.shape[0])

    def test_constraint__constructs_slack_correctly(self):
        nlp = Nlp(sym_type="MX")
        mpc = Mpc[cs.MX](nlp=nlp, prediction_horizon=10)
        x, _ = mpc.state("x1", 3)
        _, _, slack = mpc.constraint("c0", x, ">=", 5, soft=True)
        self.assertIn(slack.name(), mpc.slacks)
        self.assertIn(slack.name(), mpc.slacks)
        self.assertEqual(slack.shape, mpc.slacks[slack.name()].shape)
        self.assertEqual(mpc.nslacks, x.shape[0])
        mpc.constraint("c1", x, "<=", 10, soft=False)
        self.assertEqual(mpc.nslacks, x.shape[0])

    def test_disturbance__constructs_disturbance_correctly(self):
        N = 10
        nlp = Nlp(sym_type="MX")
        mpc = Mpc[cs.MX](nlp=nlp, prediction_horizon=N)
        d1 = mpc.disturbance("d1", 2)
        self.assertEqual(d1.shape, (2, N))
        self.assertEqual(d1.shape, mpc.disturbances["d1"].shape)
        self.assertEqual(mpc.nd, 2)
        d2 = mpc.disturbance("d2", 20)
        self.assertEqual(d2.shape, (20, N))
        self.assertEqual(d2.shape, mpc.disturbances["d2"].shape)
        self.assertEqual(mpc.nd, 22)

    def test_nonlinear_dynamics__raises__if_dynamics_already_set(self):
        nlp = Nlp(sym_type="MX")
        mpc = Mpc(nlp=nlp, prediction_horizon=10)
        mpc._dynamics_already_set = True
        with self.assertRaises(RuntimeError):
            mpc.set_nonlinear_dynamics(object())

    def test_nonlinear_dynamics__raises__if_dynamics_arguments_are_invalid(self):
        x1 = cs.SX.sym("x1", 2)
        x2 = cs.SX.sym("x2", 3)
        x = cs.vertcat(x1, x2)
        u1 = cs.SX.sym("u1", 3)
        u2 = cs.SX.sym("u2", 1)
        u = cs.vertcat(u1, u2)
        d = cs.SX.sym("d")
        p = cs.SX.sym("p")
        x_next = x + cs.vertcat(u, u2)
        F1 = cs.Function("F", [x], [x_next], {"allow_free": True})
        F2 = cs.Function("F", [x, u, d, p], [x_next + d + p])
        nlp = Nlp(sym_type="MX")
        mpc = Mpc(nlp=nlp, prediction_horizon=10, control_horizon=5)
        for F in (F1, F2):
            with self.assertRaises(ValueError):
                mpc.set_nonlinear_dynamics(F)

    @parameterized.expand([(0,), (1,)])
    def test_nonlinear_dynamics__in_multishooting__creates_dynamics_eq_constraints(
        self, i: int
    ):
        N = 10
        nlp = Nlp(sym_type="SX")
        mpc = Mpc[cs.SX](
            nlp=nlp, prediction_horizon=N, control_horizon=N // 2, shooting="multi"
        )
        x1, _ = mpc.state("x1", 2)
        x2, _ = mpc.state("x2", 3)
        u1, _ = mpc.action("u1", 3)
        u2, _ = mpc.action("u2", 1)
        x = cs.vertcat(x1[:, 0], x2[:, 0])
        u = cs.vertcat(u1[:, 0], u2[:, 0])
        x_next = x + cs.vertcat(u, u2[:, 0])
        if i == 0:
            F = cs.Function("F", [x, u], [x_next], ["x", "u"], ["x+"])
        else:
            d = mpc.disturbance("d")
            x_next += d[:, 0]
            F = cs.Function("F", [x, u, d], [x_next], ["x", "u", "d"], ["x+"])
        mpc.set_nonlinear_dynamics(F)
        self.assertIn("dyn", mpc.constraints.keys())
        self.assertEqual(mpc.ng, (1 + N) * 5)

    @parameterized.expand([(0,), (1,)])
    def test_nonlinear_dynamics__in_singleshooting__creates_states(self, i: int):
        N = 10
        nlp = Nlp(sym_type="SX")
        mpc = Mpc[cs.SX](
            nlp=nlp, prediction_horizon=N, control_horizon=N // 2, shooting="single"
        )
        mpc.state("x1", 2)
        x1 = cs.SX.sym("x1", 2)
        mpc.state("x2", 3)
        x2 = cs.SX.sym("x2", 3)
        u1, _ = mpc.action("u1", 3)
        u2, _ = mpc.action("u2", 1)
        x = cs.vertcat(x1[:, 0], x2[:, 0])
        u = cs.vertcat(u1[:, 0], u2[:, 0])
        x_next = x + cs.vertcat(u, u2[:, 0])
        if i == 0:
            F = cs.Function("F", [x, u], [x_next], ["x", "u"], ["x+"])
        else:
            d = mpc.disturbance("d")
            x_next += d[:, 0]
            F = cs.Function("F", [x, u, d], [x_next], ["x", "u", "d"], ["x+"])
        mpc.set_nonlinear_dynamics(F)
        for k in range(N):
            self.assertNotIn(f"dyn_{k}", mpc.constraints.keys())
        self.assertIn("x1", mpc.states.keys())
        self.assertIn("x2", mpc.states.keys())
        self.assertEqual(mpc.states["x1"].shape, (2, N + 1))
        self.assertEqual(mpc.states["x2"].shape, (3, N + 1))

    def test_affine_dynamics__raises__if_dynamics_already_set(self):
        nlp = Nlp(sym_type="MX")
        mpc = Mpc(nlp=nlp, prediction_horizon=10)
        mpc._dynamics_already_set = True
        with self.assertRaises(RuntimeError):
            mpc.set_affine_dynamics(object(), object())

    @parameterized.expand([("A",), ("B",), ("D",), ("c",)])
    def test_affine_dynamics__raises__if_matrices_have_wrong_shapes(self, wrong_mat):
        ns, na, nd = np.random.randint(4, 20, size=3)
        nlp = Nlp(sym_type="MX")
        mpc = Mpc(nlp=nlp, prediction_horizon=10)
        mpc.state("x", ns)
        mpc.action("u", na)
        mpc.disturbance("d", nd)
        shapes = {"A": (ns, ns), "B": (ns, na), "D": (ns, nd), "c": (ns,)}

        wrong_shapes = shapes.copy()
        wrong_shapes[wrong_mat] = np.add(shapes[wrong_mat], choice((1, -1)))

        A = np.random.randn(*wrong_shapes["A"])
        B = np.random.randn(*wrong_shapes["B"])
        D = np.random.randn(*wrong_shapes["D"])
        c = np.random.randn(*wrong_shapes["c"])
        with self.assertRaisesRegex(ValueError, f"{wrong_mat} must have shape"):
            mpc.set_affine_dynamics(A, B, D, c)

    @parameterized.expand([(0,), (6,)])
    def test_affine_dynamics__raises__when_disturbances_are_misspecified(self, nd: int):
        ns, na = np.random.randint(4, 20, size=2)
        nlp = Nlp(sym_type="MX")
        mpc = Mpc(nlp=nlp, prediction_horizon=10)
        mpc.state("x", ns)
        mpc.action("u", na)
        A = np.random.randn(ns, ns)
        B = np.random.randn(ns, na)

        if nd == 0:
            D = np.random.randn(ns, 123)
            with self.assertRaisesRegex(ValueError, "Expected D to be `None`"):
                mpc.set_affine_dynamics(A, B, D)
        else:
            mpc.disturbance("d", nd)
            D = None
            with self.assertRaisesRegex(ValueError, "D must be provided"):
                mpc.set_affine_dynamics(A, B, D)

    @parameterized.expand(product((False, True), (False, True)))
    def test_affine_dynamics__in_multishooting__creates_dynamics_eq_constraints(
        self, include_d: bool, include_c: bool
    ):
        ns, na, nd, N = np.random.randint(4, 20, size=4)
        nlp = Nlp(sym_type="MX")
        mpc = Mpc(nlp=nlp, prediction_horizon=N)
        mpc.state("x", ns)
        mpc.action("u", na)
        A = np.random.randn(ns, ns)
        B = np.random.randn(ns, na)
        if include_d:
            mpc.disturbance("d", nd)
            D = np.random.randn(ns, nd)
        else:
            D = None
        c = np.random.randn(ns) if include_c else None

        mpc.set_affine_dynamics(A, B, D, c)

        self.assertIn("dyn", mpc.constraints.keys())
        self.assertEqual(mpc.ng, (N + 1) * ns)

    @parameterized.expand(product((False, True), (False, True)))
    def test_affine_dynamics__in_singleshooting__creates_states(
        self, include_d: bool, include_c: bool
    ):
        ns, na, nd, N = np.random.randint(4, 20, size=4)
        nlp = Nlp(sym_type="MX")
        mpc = Mpc(nlp=nlp, prediction_horizon=N)
        mpc.state("x", ns)
        mpc.action("u", na)
        A = np.random.randn(ns, ns)
        B = np.random.randn(ns, na)
        if include_d:
            mpc.disturbance("d", nd)
            D = np.random.randn(ns, nd)
        else:
            D = None
        c = np.random.randn(ns) if include_c else None

        mpc.set_affine_dynamics(A, B, D, c)

        for k in range(N):
            self.assertNotIn(f"dyn_{k}", mpc.constraints.keys())
        self.assertIn("x", mpc.states.keys())
        self.assertEqual(mpc.states["x"].shape, (ns, N + 1))

    @parameterized.expand([("SX",), ("MX",)])
    def test_can_be_pickled(self, sym_type: str):
        N = 10
        mpc = Mpc(
            nlp=Nlp(sym_type=sym_type), prediction_horizon=N, control_horizon=N // 2
        )
        mpc.state("x1", 2)
        mpc.state("x2", 3)
        mpc.action("u1", 3)
        mpc.action("u2", 1)

        with cs.global_pickle_context():
            pickled = pickle.dumps(mpc)
        with cs.global_unpickle_context():
            other = pickle.loads(pickled)

        self.assertIn(repr(mpc), repr(other))


class TestPwaMpc(unittest.TestCase):
    def test_pwa_dynamics__raises__if_dynamics_already_set(self):
        nlp = Nlp(sym_type="MX")
        mpc = PwaMpc(nlp=nlp, prediction_horizon=10)
        mpc._dynamics_already_set = True
        with self.assertRaises(RuntimeError):
            mpc.set_pwa_dynamics(object(), object(), object())

    @parameterized.expand(product(("SX", "MX"), range(3), (False, True)))
    def test_pwa_dynamics__raises__if_lower_or_upper_bounds_are_set(
        self, sym_type: str, set_lower: bool, set_state: bool
    ):
        mpc = PwaMpc(Nlp(sym_type=sym_type), prediction_horizon=2)
        kwargs = {"lb" if set_lower else "ub": 0} if set_state else {}
        mpc.state("x", 1, **kwargs)
        kwargs = {"lb" if set_lower else "ub": 0} if not set_state else {}
        mpc.action("u", 1, **kwargs)

        regex = (
            "Cannot set lower and upper bounds on the "
            f"{'states' if set_state else 'actions'} in PWA systems; use "
            "arguments `D` and `E` of `set_pwa_dyanmics` instead."
        )
        with self.assertRaisesRegex(RuntimeError, regex):
            mpc.set_pwa_dynamics(None, None, None)

    def test_pwa_linear_dynamics__raises__if_matrices_have_wrong_shapes(self):
        nx, nu, n_ineq_r, n_ineq_xu = np.random.randint(4, 20, size=4)
        A = np.random.randn(nx, nx)
        B = np.random.randn(nx, nu)
        c = np.random.randn(nx)
        S = np.random.randn(n_ineq_r, nx + nu)
        T = np.random.randn(n_ineq_r)
        D = np.random.randn(n_ineq_xu, nx + nu)
        E = np.random.randn(n_ineq_xu)
        args = {"A": A, "B": B, "c": c, "S": S, "T": T, "D": D, "E": E}

        for key in args:
            shape = args[key].shape
            new_args = args.copy()
            new_args[key] = np.random.randn(*np.add(shape, choice((1, -1))))

            region = PwaRegion(
                new_args["A"],
                new_args["B"],
                new_args["c"],
                new_args["S"],
                new_args["T"],
            )
            nlp = Nlp(sym_type="MX")
            mpc = PwaMpc(nlp=nlp, prediction_horizon=3, control_horizon=5)
            mpc.state("x", nx)
            mpc.action("u", nu)
            regions = [region]
            with self.assertRaises(ValueError):
                mpc.set_pwa_dynamics(regions, new_args["D"], new_args["E"])

    def test_pwa_dynamics__in_multishooting__creates_dynamics_eq_constraints(self):
        tau, k1, k2, d, m = 0.5, 10, 1, 4, 10
        A1 = np.array([[1, tau], [-((tau * 2 * k1) / m), 1 - (tau * d) / m]])
        A2 = np.array([[1, tau], [-((tau * 2 * k2) / m), 1 - (tau * d) / m]])
        B1 = B2 = np.array([[0], [tau / m]])
        x_bnd = (5, 5)
        u_bnd = 20
        pwa_regions = (
            PwaRegion(A1, B1, np.zeros(2), np.array([[1, 0, 0]]), np.zeros(1)),
            PwaRegion(A2, B2, np.zeros(2), np.array([[-1, 0, 0]]), np.zeros(1)),
        )
        D1 = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
        E1 = np.array([x_bnd[0], x_bnd[0], x_bnd[1], x_bnd[1]])
        D2 = np.array([[1], [-1]])
        E2 = np.array([u_bnd, u_bnd])
        D = cs.diagcat(D1, D2).sparse()
        E = np.concatenate((E1, E2))
        N, ns, na = 4, 2, 1
        mpc = PwaMpc(Nlp(), prediction_horizon=N, shooting="multi")
        mpc.state("x", ns)
        mpc.action("u", na)
        mpc.set_pwa_dynamics(pwa_regions, D, E, parallelization="inline")

        nr = len(pwa_regions)
        constraints = {
            "delta_sum": (1, N),
            "z_ub": (ns * nr, N),
            "z_lb": (ns * nr, N),
            "z_x_ub": (ns * nr, N),
            "z_x_lb": (ns * nr, N),
            "region": (nr, N),
        }
        for con, shape in constraints.items():
            self.assertIn(con, mpc.constraints)
            self.assertEqual(mpc.constraints[con].shape, shape, msg=con)

    def test_pwa_dynamics__in_singleshooting__creates_states(self):
        tau, k1, k2, d, m = 0.5, 10, 1, 4, 10
        A1 = np.array([[1, tau], [-((tau * 2 * k1) / m), 1 - (tau * d) / m]])
        A2 = np.array([[1, tau], [-((tau * 2 * k2) / m), 1 - (tau * d) / m]])
        B1 = B2 = np.array([[0], [tau / m]])
        x_bnd = (5, 5)
        u_bnd = 20
        pwa_regions = (
            PwaRegion(A1, B1, np.zeros(2), np.array([[1, 0, 0]]), np.zeros(1)),
            PwaRegion(A2, B2, np.zeros(2), np.array([[-1, 0, 0]]), np.zeros(1)),
        )
        D1 = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
        E1 = np.array([x_bnd[0], x_bnd[0], x_bnd[1], x_bnd[1]])
        D2 = np.array([[1], [-1]])
        E2 = np.array([u_bnd, u_bnd])
        D = cs.diagcat(D1, D2).sparse()
        E = np.concatenate((E1, E2))
        N, ns, na = 4, 2, 1
        mpc = PwaMpc(Nlp(), prediction_horizon=N, shooting="single")
        mpc.state("x", ns)
        mpc.action("u", na)
        mpc.set_pwa_dynamics(pwa_regions, D, E, parallelization="inline")

        self.assertIn("x", mpc.states)
        self.assertEqual(mpc.states["x"].shape, (ns, N + 1))


class TestScenarioBasedMpc(unittest.TestCase):
    @parameterized.expand([("multi",), ("single",)])
    def test_state__creates_as_many_states_as_scenarios(self, shooting: str):
        K, N, ns = np.random.randint(2, 20, size=3)
        ny = ns // 2
        nx = ns - ny
        scmpc = SCMPC[cs.MX](Nlp(sym_type="MX"), K, N, shooting=shooting)
        x, xs, _ = scmpc.state("x", nx, bound_initial=False, bound_terminal=False)
        y, ys, _ = scmpc.state("y", ny, bound_initial=False, bound_terminal=False)

        self.assertEqual(len(xs), K)
        self.assertEqual(len(ys), K)
        self.assertEqual(scmpc.ns, ns)
        self.assertEqual(scmpc.ns_all, ns * K)

        xshape = (nx, N + 1)
        yshape = (ny, N + 1)
        if shooting == "multi":
            self.assertTrue(all(x.shape == x_i.shape == xshape for x_i in xs))
            self.assertTrue(all(y.shape == y_i.shape == yshape for y_i in ys))
        else:
            self.assertEqual(x.shape, xshape)
            self.assertEqual(y.shape, yshape)
            self.assertTrue(all(x_i is None for x_i in xs))
            self.assertTrue(all(y_i is None for y_i in ys))

    def test_disturbance__creates_as_many_disturbances_as_scenarios(self):
        K = np.random.randint(2, 20)
        scmpc = SCMPC[cs.MX](Nlp(sym_type="MX"), K, 10)
        size = 5
        d, ds = scmpc.disturbance("d", size)
        self.assertEqual(len(ds), K)
        self.assertEqual(scmpc.nd, size)
        self.assertEqual(scmpc.nd_all, size * K)
        self.assertEqual(d.shape, ds[0].shape)

    @parameterized.expand([("SX",), ("MX",)])
    def test_dynamics__in_multishooting__creates_dynamics_eq_constraints(self, sym_tpe):
        nx, nu, nd, N, K = np.random.randint(4, 20, size=5)
        x, u, d = cs.SX.sym("x", nx), cs.SX.sym("u", nu), cs.SX.sym("d", nd)
        x_next = cs.repmat(cs.sum1(x) + cs.sum1(u) + cs.sum1(d), nx, 1)
        F = cs.Function("F", [x, u, d], [x_next])
        scmpc = SCMPC[cs.SX](Nlp(sym_type=sym_tpe), K, N, N // 2, shooting="multi")
        scmpc.state("x1", nx // 2)
        scmpc.state("x2", nx - (nx // 2))
        scmpc.action("u1", nu // 3)
        scmpc.action("u2", nu - (nu // 3))
        scmpc.disturbance("d", nd)
        scmpc.set_nonlinear_dynamics(F)

        self.assertIn("dyn", scmpc.constraints.keys())
        self.assertEqual(scmpc.nlp.ng, (1 + N) * nx * K)

    @parameterized.expand([("SX",), ("MX",)])
    def test_dynamics__in_singleshooting__creates_state_trajectories(self, sym_type):
        nx, nu, nd, N, K = np.random.randint(4, 20, size=5)
        x, u, d = cs.SX.sym("x", nx), cs.SX.sym("u", nu), cs.SX.sym("d", nd)
        x_next = cs.repmat(cs.sum1(x) + cs.sum1(u) + cs.sum1(d), nx, 1)
        F = cs.Function("F", [x, u, d], [x_next])
        scmpc = SCMPC[cs.SX](Nlp(sym_type=sym_type), K, N, N // 2, shooting="single")
        scmpc.state("x1", nx // 2)
        scmpc.state("x2", nx - (nx // 2))
        scmpc.action("u1", nu // 3)
        scmpc.action("u2", nu - (nu // 3))
        scmpc.disturbance("d", nd)
        scmpc.set_nonlinear_dynamics(F)

        self.assertNotIn("dyn", scmpc.constraints.keys())
        self.assertEqual(scmpc.nlp.ng, 0)
        for i in range(K):
            self.assertIn(_n("x1", i), scmpc.states.keys())
            self.assertIn(_n("x2", i), scmpc.states.keys())
            self.assertEqual(scmpc.states[_n("x1", i)].shape, (nx // 2, N + 1))
            self.assertEqual(scmpc.states[_n("x2", i)].shape, (nx - (nx // 2), N + 1))

    @parameterized.expand(product((True, False), ("multi", "single")))
    def test_constraint_from_single__creates_constraint_for_all_scenarios(
        self, soft: bool, shooting: str
    ):
        N, K = 10, np.random.randint(2, 20)
        scmpc = SCMPC[cs.SX](Nlp(sym_type="SX"), K, N, shooting=shooting)
        x, _, _ = scmpc.state("x", 2)
        u, _ = scmpc.action("u", 2)
        d, _ = scmpc.disturbance("d", 4)
        scmpc.set_nonlinear_dynamics(lambda x, u, d: x + u + d[:2] - d[2:])

        lhs = cs.vertcat(cs.sum2(cs.exp(x)), cs.sum2(cs.log(u)))
        rhs = cs.sum2(cs.sin(d))
        expr = lhs - rhs
        out = scmpc.constraint_from_single("c", lhs, ">=", rhs, soft=soft)

        self.assertEqual(scmpc.nh, expr.shape[0] * K)
        exprs, lams = out[:2]
        self.assertTrue(len(exprs) == K)
        self.assertTrue(len(lams) == K)
        self.assertTrue(all(o.shape == expr.shape for o in exprs))
        self.assertTrue(all(o.shape == expr.shape for o in lams))
        if soft:
            slack, slacks = out[2:]
            self.assertTrue(len(slacks) == K)
            self.assertTrue(all(o.shape == expr.shape for o in slacks))
            self.assertTrue(slack.shape == expr.shape)

        for i, expr_i in enumerate(out[0]):
            x_i = scmpc.states_i(i)["x"]
            d_i = scmpc.disturbances_i(i)["d"]
            vars_i = [x_i, d_i, u]
            if soft:
                s_i = scmpc.slacks_i(i)["slack_c"]
                vars_i.append(s_i)

            for var in cs.symvar(expr_i):
                found = False
                for arr in vars_i:
                    for j, k in product(range(arr.shape[0]), range(arr.shape[1])):
                        if cs.is_equal(var, arr[j, k]):
                            found = True
                            break
                self.assertTrue(found, f"Variable {var} not found in scenario {i}.")

    def test_minimize_from_single__creates_objective_for_all_scenarios(self):
        N, K = 10, np.random.randint(2, 20)
        scmpc = SCMPC[cs.SX](Nlp(sym_type="SX"), K, N, shooting="multi")
        x, _, _ = scmpc.state("x", 2)
        u, _ = scmpc.action("u", 2)
        d, _ = scmpc.disturbance("d", 4)
        lhs = cs.vertcat(cs.sum2(cs.exp(x)), cs.sum2(cs.log(u)))
        rhs = cs.sum2(cs.sin(d))
        _, _, s, _ = scmpc.constraint_from_single("c", lhs, ">=", rhs, soft=True)

        def objective(x, u, d, s):
            return (
                cs.sumsqr(x)
                + cs.sum1(cs.sum2(cs.log(cs.fabs(u))))
                + cs.sum1(cs.sum2(cs.sqrt(cs.exp(d))))
                + cs.sum1(cs.sum2(s))
            )

        scmpc.minimize_from_single(objective(x, u, d, s))

        # symbolical check
        for var in cs.symvar(scmpc.nlp.f):
            found = any(
                cs.is_equal(var, arr[j, k])
                for i in range(K)
                for arr in (
                    scmpc.states_i(i)["x"],
                    u,
                    scmpc.disturbances_i(i)["d"],
                    scmpc.slacks_i(i)["slack_c"],
                )
                for j, k in product(range(arr.shape[0]), range(arr.shape[1]))
            )
            self.assertTrue(found, f"Variable {var} not found.")

        # numerical check
        numerical_J = 0.0
        u_ = np.random.randn(*u.shape)
        symbolical_J = subsevalf(scmpc.nlp.f, u, u_, eval=False)
        for i in range(K):
            x_i = scmpc.states_i(i)["x"]
            d_i = scmpc.disturbances_i(i)["d"]
            s_i = scmpc.slacks_i(i)["slack_c"]
            x_i_ = np.random.randn(*x_i.shape)
            d_i_ = np.random.randn(*d_i.shape)
            s_i_ = np.random.randn(*s_i.shape)
            numerical_J += objective(x_i_, u_, d_i_, s_i_) / K
            symbolical_J = subsevalf(
                symbolical_J, [x_i, d_i, s_i], [x_i_, d_i_, s_i_], eval=i == K - 1
            )
        np.testing.assert_almost_equal(symbolical_J, numerical_J)


class TestMultiScenarioMpc(unittest.TestCase):
    @parameterized.expand([(True,), (False,)])
    def test_init__raises__with_wrong_input_sharing(self, neg: bool):
        nlp = Nlp()
        K = np.random.randint(2, 20)
        N = 14
        input_spacing = 5
        input_sharing = -1 if neg else 4  # only 3 free us, but we ask 4 to be shared!
        with self.assertRaises(ValueError):
            MSMPC[cs.MX](
                nlp, K, N, input_spacing=input_spacing, input_sharing=input_sharing
            )

    def test_parameter__creates_as_many_parameters_as_scenarios(self):
        K, N, np1_1, np1_2, np2_1, np2_2 = np.random.randint(2, 20, size=6)
        shape1 = (np1_1, np1_2)
        shape2 = (np2_1, np2_2)
        msmpc = MSMPC[cs.MX](Nlp(sym_type="MX"), K, N)
        p1, p1s = msmpc.parameter("p1", shape1)
        p2, p2s = msmpc.parameter("p2", shape2)

        self.assertEqual(len(p1s), K)
        self.assertEqual(len(p2s), K)
        np_single = np1_1 * np1_2 + np2_1 * np2_2
        self.assertEqual(msmpc.np, np_single)
        self.assertEqual(msmpc.np_all, np_single * K)
        self.assertTrue(all(p1.shape == p1_i.shape == shape1 for p1_i in p1s))
        self.assertTrue(all(p2.shape == p2_i.shape == shape2 for p2_i in p2s))

    @parameterized.expand([(i,) for i in range(4)])
    def test_action(self, sharing: int):
        K, nu = np.random.randint(2, 20, size=2)
        Np = 21
        Nc = 14
        spacing = 5
        msmpc = MSMPC[cs.MX](Nlp(sym_type="SX"), K, Np, Nc, spacing, sharing)
        nu_free = ceil(Nc / spacing)
        assert nu_free == 3, f"Expected `nu_free` to be 3; got {nu_free} instead."
        nu2 = nu // 2
        nu1 = nu - nu2
        u1, u1_exp, u1s, u1s_exp = msmpc.action("u1", nu1)
        u2, u2_exp, u2s, u2s_exp = msmpc.action("u2", nu2)

        self.assertEqual(msmpc.na, nu)
        self.assertEqual(msmpc.na_all, nu * K)

        self.assertEqual(len(u1s), K)
        self.assertTrue(all(u1.shape == u.shape == (nu1, nu_free) for u in u1s))
        self.assertEqual(len(u1s_exp), K)
        self.assertTrue(all(u1_exp.shape == u.shape == (nu1, Np) for u in u1s_exp))
        self.assertEqual(len(u2s), K)
        self.assertTrue(all(u2.shape == u.shape == (nu2, nu_free) for u in u2s))
        self.assertEqual(len(u2s_exp), K)
        self.assertTrue(all(u2_exp.shape == u.shape == (nu2, Np) for u in u2s_exp))

        for i in range(K):
            name1 = _n("u1", i)
            self.assertIn(name1, msmpc.actions)
            self.assertIn(name1, msmpc.actions_expanded)
            name2 = _n("u2", i)
            self.assertIn(name2, msmpc.actions)
            self.assertIn(name2, msmpc.actions_expanded)

        self.assertEqual(msmpc.nx, (sharing + (nu_free - sharing) * K) * nu)

    @parameterized.expand([("SX",), ("MX",)])
    def test_dynamics__in_multishooting__creates_dynamics_eq_constraints(self, sym_tpe):
        nx, nu, nd, N, K = np.random.randint(4, 20, size=5)
        x, u, d = cs.SX.sym("x", nx), cs.SX.sym("u", nu), cs.SX.sym("d", nd)
        x_next = cs.repmat(cs.sum1(x) + cs.sum1(u) + cs.sum1(d), nx, 1)
        F = cs.Function("F", [x, u, d], [x_next])
        msmpc = MSMPC[cs.SX](Nlp(sym_type=sym_tpe), K, N, N // 2, shooting="multi")
        msmpc.state("x1", nx // 2)
        msmpc.state("x2", nx - (nx // 2))
        msmpc.action("u1", nu // 3)
        msmpc.action("u2", nu - (nu // 3))
        msmpc.disturbance("d", nd)
        msmpc.set_nonlinear_dynamics(F)

        self.assertIn("dyn", msmpc.constraints.keys())
        self.assertEqual(msmpc.nlp.ng, (1 + N) * nx * K)

    @parameterized.expand([("SX",), ("MX",)])
    def test_dynamics__in_singleshooting__creates_state_trajectories(self, sym_type):
        nx, nu, nd, N, K = np.random.randint(4, 20, size=5)
        x, u, d = cs.SX.sym("x", nx), cs.SX.sym("u", nu), cs.SX.sym("d", nd)
        x_next = cs.repmat(cs.sum1(x) + cs.sum1(u) + cs.sum1(d), nx, 1)
        F = cs.Function("F", [x, u, d], [x_next])
        msmpc = MSMPC[cs.SX](Nlp(sym_type=sym_type), K, N, N // 2, shooting="single")
        msmpc.state("x1", nx // 2)
        msmpc.state("x2", nx - (nx // 2))
        msmpc.action("u1", nu // 3)
        msmpc.action("u2", nu - (nu // 3))
        msmpc.disturbance("d", nd)
        msmpc.set_nonlinear_dynamics(F)

        self.assertNotIn("dyn", msmpc.constraints.keys())
        self.assertEqual(msmpc.nlp.ng, 0)
        for i in range(K):
            self.assertIn(_n("x1", i), msmpc.states.keys())
            self.assertIn(_n("x2", i), msmpc.states.keys())
            self.assertEqual(msmpc.states[_n("x1", i)].shape, (nx // 2, N + 1))
            self.assertEqual(msmpc.states[_n("x2", i)].shape, (nx - (nx // 2), N + 1))


if __name__ == "__main__":
    unittest.main()
