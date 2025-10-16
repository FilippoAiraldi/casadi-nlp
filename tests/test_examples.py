import contextlib
import os
import sys
import unittest
from copy import deepcopy
from typing import TYPE_CHECKING

import casadi as cs
import numpy as np
from parameterized import parameterized, parameterized_class
from scipy import io

from csnlp import Nlp, Solution, scaling, wrappers
from csnlp.multistart import ParallelMultistartNlp, StackedMultistartNlp

if TYPE_CHECKING:
    from csnlp.multistart.multistart_nlp import MultistartNlp

QRQP_OPTS = {
    "error_on_fail": True,
    "expand": True,
    "verbose": False,
    "print_time": False,
    "print_info": False,
    "print_header": False,
    "print_iter": False,
}
IPOPT_OPTS = {
    "expand": True,
    "print_time": False,
    "ipopt": {
        "max_iter": 500,
        "sb": "yes",
        # for debugging
        "print_level": 0,
        "print_user_options": "no",
        "print_options_documentation": "no",
    },
}
EXAMPLES_DATA_FILENAME = r"tests/examples_data.mat"
RESULTS = io.loadmat(EXAMPLES_DATA_FILENAME, simplify_cells=True)
# io.savemat(EXAMPLES_DATA_FILENAME, {**RESULTS, "multistart_nlp_fs": fs})


@contextlib.contextmanager
def nostdout(suppress: bool = True):
    if suppress:
        save_stdout = sys.stdout
        with open(os.devnull, "w") as f:
            sys.stdout = f
            try:
                yield
            finally:
                sys.stdout = save_stdout
    else:
        yield


@parameterized_class("sym_type", [("SX",), ("MX",)])
class TestExamples(unittest.TestCase):
    def test__chain(self):
        N = 40
        m = 40 / N
        D = 70 * N / 2
        g = 9.81
        L = 1
        nlp = Nlp(sym_type=self.sym_type)
        p = nlp.variable("p", (2, N))[0]
        x, y = p[0, :], p[1, :]
        V = D * cs.sum2(
            (cs.sqrt(cs.diff(x) ** 2 + cs.diff(y) ** 2) - L / N) ** 2
        ) + g * m * cs.sum2(y)
        nlp.minimize(V)
        nlp.init_solver(IPOPT_OPTS)
        nlp.constraint("c1", p[:, 0], "==", [-2, 1])
        nlp.constraint("c2", p[:, -1], "==", [2, 1])
        nlp.constraint("c3", y, ">=", cs.cos(0.1 * x) - 0.5)
        nlp = deepcopy(nlp)
        sol = nlp(vals0={"p": np.vstack((np.linspace(-2, 2, N), np.ones(y.shape)))})
        np.testing.assert_allclose(sol.vals["p"], RESULTS["chain_p"])

    def test__multistart_nlp(self):
        def func(x):
            return (
                -0.3 * x**2
                - np.exp(-10 * x**2)
                + np.exp(-100 * (x - 1) ** 2)
                + np.exp(-100 * (x - 1.5) ** 2)
            )

        N = 3
        LB, UB = -0.5, 1.4
        x0s = [0.9, 0.5, 1.1]
        args = ([{"p0": 0, "p1": 1} for _ in x0s], [{"x": x0} for x0 in x0s])

        nlps: list[MultistartNlp] = [
            StackedMultistartNlp(starts=N, sym_type=self.sym_type),
            ParallelMultistartNlp(
                starts=N, sym_type=self.sym_type, parallel_kwargs={"n_jobs": N}
            ),
        ]
        sols = []
        for nlp in nlps:
            x = nlp.variable("x", lb=LB, ub=UB)[0]
            nlp.parameter("p0")
            nlp.parameter("p1")
            nlp.minimize(func(x))
            nlp.init_solver(IPOPT_OPTS)
            nlp = deepcopy(nlp)

            best_sol: Solution = nlp.solve_multi(*args)
            all_sols: list[Solution] = nlp.solve_multi(*args, return_all_sols=True)
            all_sols.sort(key=Solution.cmp_key)
            fs = [all_sol.f for all_sol in all_sols]
            self.assertEqual(best_sol.f, fs[0])
            np.testing.assert_allclose(fs, np.sort(RESULTS["multistart_nlp_fs"]))
            sols.append(all_sols)

        for sol1, sol2 in zip(*sols):
            self.assertEqual(sol1.success, sol2.success)
            np.testing.assert_allclose(sol1.f, sol2.f)
            np.testing.assert_allclose(sol1.vals["x"], sol2.vals["x"])
            np.testing.assert_allclose(
                sol1.value(nlps[0].lam_lbx), sol2.value(nlps[1].lam_lbx), atol=1e-6
            )
            np.testing.assert_allclose(
                sol1.value(nlps[0].lam_ubx), sol2.value(nlps[1].lam_ubx), atol=1e-6
            )

    @parameterized.expand([("single",), ("multi",)])
    def test__optimal_ctrl(self, shooting: str):
        x = cs.MX.sym("x", 2)
        u = cs.MX.sym("u")
        ode = cs.vertcat((1 - x[1] ** 2) * x[0] - x[1] + u, x[0])
        f = cs.Function("f", [x, u], [ode], ["x", "u"], ["ode"])
        T = 10
        N = 20
        intg_options = {"simplify": True, "number_of_finite_elements": 4}
        dae = {"x": x, "p": u, "ode": f(x, u)}
        intg = cs.integrator("intg", "rk", dae, 0.0, T / N, intg_options)
        res = intg(x0=x, p=u)
        x_next = res["xf"]
        F = cs.Function("F", [x, u], [x_next], ["x", "u"], ["x_next"])
        mpc = wrappers.Mpc(
            nlp=Nlp(sym_type=self.sym_type),
            prediction_horizon=N,
            shooting=shooting,
        )
        u, _ = mpc.action("u", lb=-1, ub=+1)
        if shooting == "single":
            mpc.state("x", 2)
            mpc.set_nonlinear_dynamics(F)
            x = mpc.states["x"]  # only accessible after dynamics have been set
            mpc.constraint("c0", x, ">=", -0.2)
        else:
            x, _ = mpc.state("x", 2, lb=-0.2)  # must be created before dynamics
            mpc.set_nonlinear_dynamics(F)
        mpc.minimize(cs.sumsqr(x) + cs.sumsqr(u))
        mpc.init_solver(IPOPT_OPTS)
        mpc = deepcopy(mpc)
        sol = mpc.solve(pars={"x_0": [0, 1]})
        u_opt = sol.vals["u"].full().flat
        np.testing.assert_allclose(
            u_opt, RESULTS["optimal_ctrl_u"], rtol=1e-6, atol=1e-6
        )

    def test__rosenbrock(self):
        nlp = Nlp(sym_type=self.sym_type)
        x = nlp.variable("x", (2, 1))[0]
        r = nlp.parameter("r")
        f = (1 - x[0]) ** 2 + (x[1] - x[0] ** 2) ** 2
        nlp.minimize(f)
        nlp.constraint("con1", cs.sumsqr(x), "<=", r)
        nlp.init_solver(IPOPT_OPTS)
        nlp = deepcopy(nlp)
        r_values = np.linspace(1, 3, 25)
        f_values = []
        lam_values = []
        for r_value in r_values:
            sol = nlp.solve(pars={"r": r_value})
            f_values.append(sol.f)
            lam_values.append(sol.dual_vals["lam_h_con1"])
        f_values = np.asarray(f_values).squeeze()
        lam_values = np.asarray(lam_values).squeeze()
        np.testing.assert_allclose(f_values, RESULTS["rosenbrock_f"])
        np.testing.assert_allclose(lam_values, RESULTS["rosenbrock_lam"])

    def test__sensitivity(self):
        def z1(x, lam, p):
            return (x[1, :] ** p[1] - x[0, :]) * cs.exp(-10 * lam + p[0]) / p[1]

        def z2(x):
            return x[1, :] - x[0, :]

        def z3(x, p):
            return x[1, :] ** (1 / p[1]) - x[0, :]

        def z4(lam, p):
            return cs.exp(-10 * lam + p[0]) / p[1]

        nlp = wrappers.NlpSensitivity(Nlp(sym_type=self.sym_type))
        x = nlp.variable("x", (2, 1), lb=[[0], [-np.inf]])[0]
        p = nlp.parameter("p", (2, 1))
        nlp.minimize((1 - x[0]) ** 2 + p[0] * (x[1] - x[0] ** 2) ** 2)
        g = (x[0] + 0.5) ** 2 + x[1] ** 2
        nlp.constraint("c1", (p[1] / 2) ** 2, "<=", g)
        _, lam = nlp.constraint("c2", g, "<=", p[1] ** 2)
        opts = {"print_time": False, "ipopt": {"sb": "yes", "print_level": 0}}
        nlp.init_solver(opts)
        nlp = deepcopy(nlp)
        Z = cs.blockcat(
            [
                [z1(x, lam, p) ** 2, z2(x)],
                [1 / (1 + z3(x, p)), z4(lam, p)],
                [z3(x, p) * (-1 - 10 * z1(x, lam, p)), z4(lam, p) / (1 + z2(x))],
            ]
        )
        z0, j0, h0 = [], [], []
        J, H = nlp.parametric_sensitivity(expr=Z, second_order=True)
        for p0 in (1.2, 1.45, 1.9):
            sol = nlp.solve(pars={"p": [0.2, p0]})
            z0.append(sol.value(Z))
            j0.append(sol.value(J))
            h0.append(sol.value(H))
        z0 = np.asarray(z0).squeeze()
        j0 = np.asarray(j0).squeeze()
        h0 = np.asarray(h0).squeeze()
        np.testing.assert_allclose(RESULTS["sensitivity_z"], z0, rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(RESULTS["sensitivity_j"], j0, rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(RESULTS["sensitivity_h"], h0, rtol=1e-6, atol=1e-7)

    @parameterized.expand([(StackedMultistartNlp,), (ParallelMultistartNlp,)])
    def test__scaling(self, multinlp_cls: type):
        def get_dynamics(g: float, alpha: float, dt: float) -> cs.Function:
            x, u = cs.SX.sym("x", 3), cs.SX.sym("u")
            x_next = x + cs.vertcat(x[1], u / x[2] - g, -alpha * u) * dt
            return cs.Function("F", [x, u], [x_next], ["x", "u"], ["x+"])

        N = 100
        T = 100
        K = 3
        dt = T / N
        m0 = 500000
        yT = 100000
        g = 9.81
        alpha = 1 / (300 * g)
        seed = 69
        rng = np.random.default_rng(seed)
        kwargs = (
            {"parallel_kwargs": {"n_jobs": -1}}
            if multinlp_cls is ParallelMultistartNlp
            else {}
        )
        nlp = multinlp_cls(sym_type="SX", starts=K, **kwargs)

        y_nom = 1e5
        v_nom = 2e3
        m_nom = 3e5
        x_nom = cs.vertcat(y_nom, v_nom, m_nom)
        u_nom = 1e8
        scaler = scaling.Scaler()
        scaler.register("x", scale=x_nom)
        scaler.register("x_0", scale=x_nom)
        scaler.register("u", scale=u_nom)
        nlp = wrappers.NlpScaling(nlp, scaler=scaler)

        mpc = wrappers.Mpc(nlp, prediction_horizon=N)
        x, _ = mpc.state("x", 3, lb=cs.DM([-cs.inf, -cs.inf, 0]))
        y = x[0, :]
        m = x[2, :]
        u, _ = mpc.action("u", lb=0, ub=5e7)
        mpc.set_nonlinear_dynamics(get_dynamics(g, alpha, dt))
        x0 = cs.vertcat(0, 0, m0)
        mpc.constraint("yT", y[-1], "==", yT)
        mpc.minimize(m[0] - m[-1])
        mpc.init_solver(IPOPT_OPTS)
        mpc = deepcopy(mpc)

        x_init = cs.repmat([0, 0, 1e5], 1, N + 1)

        pars = [{"x_0": x0}] * K
        vals0 = [
            {
                "x": x_init + rng.random(x_init.shape) * 1e4,
                "u": rng.random() * 1e8,
            }
            for _ in range(K)
        ]
        us, fs = [], []
        for i in range(K + 1):
            sol = mpc.solve(pars[i], vals0[i]) if i != K else mpc(pars, vals0)
            fs.append(sol.f)
            us.append(sol.value(u))
        us, fs = np.asarray(us).squeeze(), np.asarray(fs).squeeze()

        np.testing.assert_almost_equal(fs[-1], fs[:-1].min(), decimal=2)
        np.testing.assert_allclose(RESULTS["scaling_fs"], fs, rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(RESULTS["scaling_us"], us, rtol=1e-6, atol=1e-7)

    def test__milp(self):
        nlp = Nlp(sym_type=self.sym_type)
        z = nlp.variable("z", (2, 1), lb=-0.5, discrete=True)[0]
        x, y = z[0], z[1]
        nlp.minimize(-y)
        _, _ = nlp.constraint("con1", -x + y, "<=", 1)
        _, _ = nlp.constraint("con2", 3 * x + 2 * y, "<=", 12)
        _, _ = nlp.constraint("con3", 2 * x + 3 * y, "<=", 12)
        nlp.init_solver({"cbc": {"logLevel": 0}}, solver="cbc")
        sol = nlp.solve()

        self.assertTrue(sol.success)
        z_opt = sol.vals["z"].full().flatten()
        self.assertTrue(np.array_equal(z_opt, [1, 2]) or np.array_equal(z_opt, [2, 2]))

    def test__miqp(self):
        np_random = np.random.default_rng(42)
        m, n = 10, 5
        A = np_random.random(size=(m, n))
        b = np_random.normal(size=m)

        nlp = Nlp(sym_type=self.sym_type)
        x = nlp.variable("x", (n, 1), discrete=True)[0]
        nlp.minimize(cs.sumsqr(A @ x - b))
        nlp.init_solver(solver="bonmin")
        with nostdout():
            sol = nlp.solve()

        self.assertTrue(sol.success)
        x_opt = sol.vals["x"].full().flatten()
        np.testing.assert_array_equal(x_opt, [1, 0, -1, 1, -1])

    @parameterized.expand([("single",), ("multi",)])
    def test__pwa_mpc(self, shooting: str):
        np_random = np.random.default_rng(42)

        tau, k1, k2, d, m = 0.5, 10, 1, 4, 10
        A1 = np.array([[1, tau], [-((tau * 2 * k1) / m), 1 - (tau * d) / m]])
        A2 = np.array([[1, tau], [-((tau * 2 * k2) / m), 1 - (tau * d) / m]])
        B1 = B2 = np.array([[0], [tau / m]])
        C1, C2 = np_random.normal(scale=0.01, size=(2, A1.shape[0]))
        S1 = np.array([[1, 0, 0]])
        S2 = -S1
        T1, T2 = np_random.normal(scale=0.01, size=(2, S1.shape[0]))
        x_bnd = (5, 5)
        u_bnd = 20
        pwa_regions = (
            wrappers.PwaRegion(A1, B1, C1, S1, T1),
            wrappers.PwaRegion(A2, B2, C2, S2, T2),
        )
        D1 = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
        E1 = np.array([x_bnd[0], x_bnd[0], x_bnd[1], x_bnd[1]])
        D2 = np.array([[1], [-1]])
        E2 = np.array([u_bnd, u_bnd])
        D = cs.diagcat(D1, D2).sparse()
        E = np.concatenate((E1, E2))
        mpc = wrappers.PwaMpc(
            nlp=Nlp(sym_type=self.sym_type), prediction_horizon=4, shooting=shooting
        )
        x, _ = mpc.state("x", 2)
        u, _ = mpc.action("u")
        with nostdout():
            mpc.set_pwa_dynamics(pwa_regions, D, E, parallelization="serial")
        if shooting == "single":
            x = mpc.states["x"]  # previous `x` is None if in single shooting
        mpc.constraint("state_constraints", D1 @ x - E1, "<=", 0)
        mpc.constraint("input_constraints", D2 @ u - E2, "<=", 0)
        mpc.minimize(cs.sumsqr(x) + cs.sumsqr(u))
        mpc.init_solver(solver="bonmin")
        with nostdout():
            sol = mpc.solve(pars={"x_0": [-3, 0]})

        tols = (1e-6, 1e-6)
        expected = {
            "u": RESULTS["pwa_mpc_u"].reshape(u.shape[0], -1),
            "x": RESULTS["pwa_mpc_x"],
            "delta": RESULTS["pwa_mpc_delta"],
        }
        actual = {"u": sol.vals["u"], "x": sol.value(x), "delta": sol.vals["delta"]}
        for name, val in expected.items():
            np.testing.assert_allclose(actual[name], val, *tols, err_msg=name)

    @parameterized.expand([("single",), ("multi",)])
    def test__pwa_mpc__with_sequence(self, shooting: str):
        np_random = np.random.default_rng(42)

        tau, k1, k2, d, m = 0.5, 10, 1, 4, 10
        A1 = np.array([[1, tau], [-((tau * 2 * k1) / m), 1 - (tau * d) / m]])
        A2 = np.array([[1, tau], [-((tau * 2 * k2) / m), 1 - (tau * d) / m]])
        B1 = B2 = np.array([[0], [tau / m]])
        C1, C2 = np_random.normal(scale=0.01, size=(2, A1.shape[0]))
        S1 = np.array([[1, 0, 0]])
        S2 = -S1
        T1, T2 = np_random.normal(scale=0.01, size=(2, S1.shape[0]))
        x_bnd = (5, 5)
        u_bnd = 20
        pwa_regions = (
            wrappers.PwaRegion(A1, B1, C1, S1, T1),
            wrappers.PwaRegion(A2, B2, C2, S2, T2),
        )
        D1 = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
        E1 = np.array([x_bnd[0], x_bnd[0], x_bnd[1], x_bnd[1]])
        D2 = np.array([[1], [-1]])
        E2 = np.array([u_bnd, u_bnd])
        mpc = wrappers.PwaMpc(
            nlp=Nlp(sym_type=self.sym_type), prediction_horizon=4, shooting=shooting
        )
        x, _ = mpc.state("x", 2)
        u, _ = mpc.action("u")
        mpc.set_affine_time_varying_dynamics(pwa_regions)
        if shooting == "single":
            x = mpc.states["x"]  # previous `x` is None if in single shooting
        mpc.constraint("state_constraints", D1 @ x - E1, "<=", 0)
        mpc.constraint("input_constraints", D2 @ u - E2, "<=", 0)
        mpc.minimize(cs.sumsqr(x) + cs.sumsqr(u))
        mpc.init_solver(QRQP_OPTS, "qrqp")
        mpc.set_switching_sequence([0, 0, 0, 1])
        sol = mpc.solve(pars={"x_0": [-3, 0]})

        tols = (1e-6, 1e-6)
        expected = {
            "u": RESULTS["pwa_mpc_u"].reshape(u.shape[0], -1),
            "x": RESULTS["pwa_mpc_x"],
        }
        actual = {"u": sol.vals["u"], "x": sol.value(x)}
        for name, val in expected.items():
            np.testing.assert_allclose(actual[name], val, *tols, err_msg=name)

    @parameterized.expand([("single",), ("multi",)])
    def test__linear_mpc(self, shooting: str):
        A = np.asarray(
            [
                [0.763, 0.460, 0.115, 0.020],
                [-0.899, 0.763, 0.420, 0.115],
                [0.115, 0.020, 0.763, 0.460],
                [0.420, 0.115, -0.899, 0.763],
            ]
        )
        B = np.asarray([[0.014], [0.063], [0.221], [0.367]])
        c = A.sum(axis=1) * 0.1  # also add an affine term to spice things up
        ns, na = B.shape
        N = 7
        mpc = wrappers.Mpc(Nlp[cs.SX](), prediction_horizon=N, shooting=shooting)
        mpc.state("x", ns)
        u, _ = mpc.action("u", na, lb=-0.5, ub=0.5)
        mpc.set_affine_dynamics(A, B, c=c)
        x = mpc.states["x"]
        x_bound = np.asarray([[4.0], [10.0], [4.0], [10.0]])
        mpc.constraint("x_lb", x, ">=", -x_bound)
        mpc.constraint("x_ub", x, "<=", x_bound)
        delta_u = cs.diff(u, 1, 1)
        mpc.minimize(cs.sumsqr(x) + 1e-4 * cs.sumsqr(delta_u))
        mpc.init_solver(QRQP_OPTS, "qrqp", type="conic")
        x = RESULTS["lti_mpc_xs"][0]
        X, U = [x], []
        for _ in range(50):
            sol = mpc.solve(pars={"x_0": x})
            u_opt = sol.vals["u"][:, 0].full().reshape(na)
            x = A @ x + B @ u_opt
            X.append(x)
            U.append(u_opt)
        X = np.squeeze(X)
        U = np.squeeze(U)

        np.testing.assert_allclose(X, RESULTS["lti_mpc_xs"], atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(U, RESULTS["lti_mpc_us"], atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
