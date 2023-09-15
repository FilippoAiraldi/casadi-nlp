import pickle
import unittest
from itertools import product

import casadi as cs
import numpy as np
from parameterized import parameterized

from csnlp import Nlp
from csnlp.core.solutions import subsevalf
from csnlp.wrappers import Mpc
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

    @parameterized.expand([(2,), (1,)])
    def test_action__constructs_action_correctly(self, divider: int):
        Np = 10
        Nc = Np // divider
        nlp = Nlp(sym_type="SX")
        mpc = Mpc[cs.SX](nlp=nlp, prediction_horizon=Np, control_horizon=Nc)
        u1, u1_exp = mpc.action("u1", 2)
        self.assertEqual(u1.shape, (2, Nc))
        self.assertEqual(u1.shape, mpc.actions["u1"].shape)
        self.assertEqual(u1_exp.shape, (2, Np))
        self.assertEqual(u1_exp.shape, mpc.actions_expanded["u1"].shape)
        self.assertEqual(mpc.na, u1.shape[0])
        u2, u2_exp = mpc.action("u2", 1)
        self.assertEqual(u2.shape, (1, Nc))
        self.assertEqual(u2.shape, mpc.actions["u2"].shape)
        self.assertEqual(u2_exp.shape, (1, Np))
        self.assertEqual(u2_exp.shape, mpc.actions_expanded["u2"].shape)
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

    def test_dynamics__raises__if_dynamics_already_set(self):
        nlp = Nlp(sym_type="MX")
        mpc = Mpc(nlp=nlp, prediction_horizon=10)
        mpc._dynamics = 5
        with self.assertRaises(RuntimeError):
            mpc.set_dynamics(6)

    def test_dynamics__raises__if_dynamics_arguments_are_invalid(self):
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
                mpc.set_dynamics(F)

    @parameterized.expand([(0,), (1,)])
    def test_dynamics__in_multishooting__creates_dynamics_eq_constraints(self, i: int):
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
        mpc.set_dynamics(F)
        self.assertIn("dyn", mpc.constraints.keys())
        self.assertEqual(mpc.ng, (1 + N) * 5)

    @parameterized.expand([(0,), (1,)])
    def test_dynamics__in_singleshooting__creates_states(self, i: int):
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
        mpc.set_dynamics(F)
        for k in range(N):
            self.assertNotIn(f"dyn_{k}", mpc.constraints.keys())
        self.assertIn("x1", mpc.states.keys())
        self.assertIn("x2", mpc.states.keys())
        self.assertEqual(mpc.states["x1"].shape, (2, N + 1))
        self.assertEqual(mpc.states["x2"].shape, (3, N + 1))

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

        mpc2 = pickle.loads(pickle.dumps(mpc))

        self.assertIn(repr(mpc), repr(mpc2))


class TestScenarioBasedMpc(unittest.TestCase):
    @parameterized.expand([(False,), (True,)])
    def test_state__creates_as_many_states_as_scenarios(self, multishooting: bool):
        K = np.random.randint(2, 20)
        shooting = "multi" if multishooting else "single"
        scmpc = SCMPC[cs.MX](Nlp(sym_type="MX"), K, 10, shooting=shooting)
        size = 4
        x, xs, _ = scmpc.state("x", size, remove_bounds_on_initial=True)
        self.assertEqual(len(xs), K)
        self.assertEqual(scmpc.ns, size)
        self.assertEqual(scmpc.ns_all, size * K)
        if multishooting:
            self.assertEqual(x.shape, xs[0].shape)
        else:
            self.assertIsNone(x)
            self.assertTrue(all(x_i is None for x_i in xs))

    def test_disturbance__creates_as_many_disturbances_as_scenarios(self):
        K = np.random.randint(2, 20)
        scmpc = SCMPC[cs.MX](Nlp(sym_type="MX"), K, 10)
        size = 5
        d, ds = scmpc.disturbance("d", size)
        self.assertEqual(len(ds), K)
        self.assertEqual(scmpc.nd, size)
        self.assertEqual(scmpc.nd_all, size * K)
        self.assertEqual(d.shape, ds[0].shape)

    @parameterized.expand([("multi",), ("single",)])
    def test_dynamics__in_multishooting__creates_dynamics_eq_constraints(
        self, shooting: str
    ):
        x, u, d = cs.SX.sym("x", 5), cs.SX.sym("u", 4), cs.SX.sym("d", 1)
        F = cs.Function("F", [x, u, d], [x + cs.vertcat(u, u[-1]) + d])

        N, K = 10, np.random.randint(2, 20)
        scmpc = SCMPC[cs.SX](Nlp(sym_type="SX"), K, N, N // 2, shooting=shooting)
        scmpc.state("x1", 2)
        scmpc.state("x2", 3)
        scmpc.action("u1", 3)
        scmpc.action("u2", 1)
        scmpc.disturbance("d")
        scmpc.set_dynamics(F)

        if shooting == "multi":
            self.assertEqual(scmpc.nlp.ng, (1 + N) * 5 * K)
            for i in range(K):
                self.assertIn(_n("dyn", i), scmpc.constraints.keys())
        else:
            self.assertEqual(scmpc.nlp.ng, 0)
            for i in range(K):
                self.assertIn(_n("x1", i), scmpc.states.keys())
                self.assertIn(_n("x2", i), scmpc.states.keys())
                self.assertEqual(scmpc.states[_n("x1", i)].shape, (2, N + 1))
                self.assertEqual(scmpc.states[_n("x2", i)].shape, (3, N + 1))

    @parameterized.expand([(True,), (False,)])
    def test_constraint_from_single__creates_constraint_for_all_scenarios(
        self, soft: bool
    ):
        N, K = 10, np.random.randint(2, 20)
        scmpc = SCMPC[cs.SX](Nlp(sym_type="SX"), K, N)
        x, _, _ = scmpc.state("x", 2)
        u, _ = scmpc.action("u", 2)
        d, _ = scmpc.disturbance("d", 4)

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
        scmpc = SCMPC[cs.SX](Nlp(sym_type="SX"), K, N)
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


if __name__ == "__main__":
    unittest.main()
