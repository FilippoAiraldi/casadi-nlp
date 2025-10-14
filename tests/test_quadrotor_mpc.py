import unittest
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Optional, TypeVar, Union

import casadi as cs
import numpy as np
from parameterized import parameterized

from csnlp import Nlp, Solution
from csnlp.wrappers import Mpc


@dataclass
class QuadRotorEnvConfig:
    T: float = 0.1
    g: float = 9.81
    thrust_coeff: float = 1.4
    pitch_d: float = 10
    pitch_dd: float = 8
    pitch_gain: float = 10
    roll_d: float = 10
    roll_dd: float = 8
    roll_gain: float = 10
    winds: dict[float, float] = field(default_factory=lambda: {1: 1.0, 2: 0.7, 3: 0.85})
    x0: np.ndarray = field(
        default_factory=lambda: np.array([0, 0, 3.5, 0, 0, 0, 0, 0, 0, 0])
    )
    xf: np.ndarray = field(
        default_factory=lambda: np.array([3, 3, 0.2, 0, 0, 0, 0, 0, 0, 0])
    )
    soft_constraints: bool = True
    x_bounds: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                [-0.5, 3.5],
                [-0.5, 3.5],
                [-0.175, 4],
                [-np.inf, np.inf],
                [-np.inf, np.inf],
                [-np.inf, np.inf],
                [np.deg2rad(-30), np.deg2rad(30)],
                [np.deg2rad(-30), np.deg2rad(30)],
                [-np.inf, np.inf],
                [-np.inf, np.inf],
            ]
        )
    )
    u_bounds: np.ndarray = field(
        default_factory=lambda: np.array(
            [[-np.pi, np.pi], [-np.pi, np.pi], [0, 2 * 9.81]]
        )
    )

    # termination conditions
    termination_N: int = 5
    termination_error: float = 0.5


class QuadRotorEnv:
    nx: int = 10
    nu: int = 3

    def __init__(self, config: Union[dict, QuadRotorEnvConfig] = None) -> None:
        config = init_config(config, QuadRotorEnvConfig)
        self.config = config

        # create dynamics matrices
        self._A, self._B, self._C, self._e = self.get_dynamics(
            g=config.g,
            thrust_coeff=config.thrust_coeff,
            pitch_d=config.pitch_d,
            pitch_dd=config.pitch_dd,
            pitch_gain=config.pitch_gain,
            roll_d=config.roll_d,
            roll_dd=config.roll_dd,
            roll_gain=config.roll_gain,
            winds=config.winds,
        )
        # weight for positional, control action usage and violation errors
        self._Wx = np.ones(self.nx)
        self._Wu = np.ones(self.nu)
        self._Wv = np.array([1e2, 1e2, 3e2, 3e2])

    @property
    def A(self) -> np.ndarray:
        return self._A.copy()

    @property
    def B(self) -> np.ndarray:
        return self._B.copy()

    @property
    def C(self) -> np.ndarray:
        return self._C.copy()

    @property
    def e(self) -> np.ndarray:
        return self._e.copy()

    @property
    def x(self) -> np.ndarray:
        return self._x.copy()

    @x.setter
    def x(self, val: np.ndarray) -> None:
        self._x = val.copy()

    def position_error(self, x: np.ndarray) -> float:
        return (np.square(x - self.config.xf) * self._Wx).sum(axis=-1)

    def control_usage(self, u: np.ndarray) -> float:
        return (np.square(u) * self._Wu).sum(axis=-1)

    def constraint_violations(self, x: np.ndarray, u: np.ndarray) -> float:
        W = self._Wv
        return (
            W[0] * np.maximum(0, self.config.x_bounds[:, 0] - x).sum(axis=-1)
            + W[1] * np.maximum(0, x - self.config.x_bounds[:, 1]).sum(axis=-1)
            + W[2] * np.maximum(0, self.config.u_bounds[:, 0] - u).sum(axis=-1)
            + W[3] * np.maximum(0, u - self.config.u_bounds[:, 1]).sum(axis=-1)
        )

    def phi(self, alt: Union[float, np.ndarray]) -> np.ndarray:
        if isinstance(alt, np.ndarray):
            alt = alt.squeeze()
            assert alt.ndim == 1, "Altitudes must be a vector"

        return np.vstack([np.exp(-np.square(alt - h)) for h in self.config.winds])

    def reset(
        self, seed: int = None, x0: np.ndarray = None, xf: np.ndarray = None
    ) -> np.ndarray:
        self.np_random = np.random.default_rng(seed)
        if x0 is None:
            x0 = self.config.x0
        if xf is None:
            xf = self.config.xf
        self.x = x0
        self.config.x0 = x0
        self.config.xf = xf
        self._n_within_termination = 0
        return self.x

    def step(self, u: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        u = u.squeeze()  # in case a row or col was passed
        wind = (
            self._C
            @ self.phi(self.x[2])
            * self.np_random.uniform(
                low=[0, 0, -1, 0, 0, 0, -1, -1, 0, 0],
                high=[1, 1, 0, 0, 0, 0, 1, 1, 0, 0],
            ).reshape(self.nx, 1)
        )
        self.x = (
            self._A @ self.x.reshape((-1, 1))
            + self._B @ u.reshape((-1, 1))
            + self._e
            + wind
        ).flatten()

        # compute cost
        error = self.position_error(self.x)
        usage = self.control_usage(u)
        violations = self.constraint_violations(self.x, u)
        cost = float(error + usage + violations)

        # check if terminated
        within_bounds = (
            (self.config.x_bounds[:, 0] <= self._x)
            & (self._x <= self.config.x_bounds[:, 1])
        ).all()
        if error <= self.config.termination_error and within_bounds:
            self._n_within_termination += 1
        else:
            self._n_within_termination = 0
        terminated = self._n_within_termination >= self.config.termination_N
        return self.x, cost, terminated, False, {"error": error}

    def render(self):
        raise NotImplementedError("Render method unavailable.")

    def get_dynamics(
        self,
        g: Union[float, cs.SX],
        thrust_coeff: Union[float, cs.SX],
        pitch_d: Union[float, cs.SX],
        pitch_dd: Union[float, cs.SX],
        pitch_gain: Union[float, cs.SX],
        roll_d: Union[float, cs.SX],
        roll_dd: Union[float, cs.SX],
        roll_gain: Union[float, cs.SX],
        winds: dict[float, float] = None,
    ) -> Union[
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        tuple[cs.SX, cs.SX, cs.SX],
    ]:
        T = self.config.T
        is_casadi = any(
            isinstance(o, (cs.SX, cs.MX, cs.DM))
            for o in [
                g,
                thrust_coeff,
                pitch_d,
                pitch_dd,
                pitch_gain,
                roll_d,
                roll_dd,
                roll_gain,
            ]
        )
        if is_casadi:
            diag = lambda o: cs.diag(cs.vertcat(*o))  # noqa: E731
            block = cs.blockcat
        else:
            diag = np.diag
            block = np.block
            assert winds is not None, "Winds are required to compute matrix C."
            nw = len(winds)
            wind_mag = np.array(list(winds.values()))
        A = T * block(
            [
                [np.zeros((3, 3)), np.eye(3), np.zeros((3, 4))],
                [np.zeros((2, 6)), np.eye(2) * g, np.zeros((2, 2))],
                [np.zeros((1, 10))],
                [np.zeros((2, 6)), -diag((pitch_d, roll_d)), np.eye(2)],
                [np.zeros((2, 6)), -diag((pitch_dd, roll_dd)), np.zeros((2, 2))],
            ]
        ) + np.eye(10)
        B = T * block(
            [
                [np.zeros((5, 3))],
                [0, 0, thrust_coeff],
                [np.zeros((2, 3))],
                [pitch_gain, 0, 0],
                [0, roll_gain, 0],
            ]
        )
        if not is_casadi:
            C = T * block(
                [
                    [wind_mag],
                    [wind_mag],
                    [wind_mag],
                    [np.zeros((3, nw))],
                    [wind_mag],
                    [wind_mag],
                    [np.zeros((2, nw))],
                ]
            )
        e = block([[np.zeros((5, 1))], [-T * g], [np.zeros((4, 1))]])
        return (A, B, e) if is_casadi else (A, B, C, e)


@dataclass(frozen=True)
class QuadRotorSolution:
    f: float
    vars: dict[str, cs.SX]
    vals: dict[str, np.ndarray]
    stats: dict[str, Any]
    get_value: partial

    @property
    def status(self) -> str:
        return self.stats["return_status"]

    @property
    def success(self) -> bool:
        return self.stats["success"]

    def value(self, x: cs.SX) -> np.ndarray:
        return self.get_value(x)


class GenericMPC:
    def __init__(self, name: str = None) -> None:
        self.name = f"MPC{np.random.random()}" if name is None else name
        self.f: cs.SX = None  # objective
        self.vars: dict[str, cs.SX] = {}
        self.pars: dict[str, cs.SX] = {}
        self.cons: dict[str, cs.SX] = {}
        self.p = cs.SX()
        self.x, self.lbx, self.ubx = cs.SX(), np.array([]), np.array([])
        self.lam_lbx, self.lam_ubx = cs.SX(), cs.SX()
        self.g, self.lbg, self.ubg = cs.SX(), np.array([]), np.array([])
        self.lam_g = cs.SX()
        self.h, self.lbh, self.ubh = cs.SX(), np.array([]), np.array([])
        self.lam_h = cs.SX()
        self.solver: cs.Function = None
        self.opts: dict = None

    @property
    def ng(self) -> int:
        return self.g.shape[0]

    def add_par(self, name: str, *dims: int) -> cs.SX:
        assert name not in self.pars, f"Parameter {name} already exists."
        par = cs.SX.sym(name, *dims)
        self.pars[name] = par
        self.p = cs.vertcat(self.p, cs.vec(par))
        return par

    def add_var(
        self,
        name: str,
        *dims: int,
        lb: np.ndarray = -np.inf,
        ub: np.ndarray = np.inf,
    ) -> tuple[cs.SX, cs.SX, cs.SX]:
        assert name not in self.vars, f"Variable {name} already exists."
        lb, ub = np.broadcast_to(lb, dims), np.broadcast_to(ub, dims)
        assert np.all(lb < ub), "Improper variable bounds."

        var = cs.SX.sym(name, *dims)
        self.vars[name] = var
        self.x = cs.vertcat(self.x, cs.vec(var))
        self.lbx = np.concatenate((self.lbx, cs.vec(lb).full().flatten()))
        self.ubx = np.concatenate((self.ubx, cs.vec(ub).full().flatten()))

        # create also the multiplier associated to the variable
        lam_lb = cs.SX.sym(f"lam_lb_{name}", *dims)
        self.lam_lbx = cs.vertcat(self.lam_lbx, cs.vec(lam_lb))
        lam_ub = cs.SX.sym(f"lam_ub_{name}", *dims)
        self.lam_ubx = cs.vertcat(self.lam_ubx, cs.vec(lam_ub))
        return var, lam_lb, lam_ub

    def add_con(
        self, name: str, expr1: cs.SX, op: str, expr2: cs.SX
    ) -> tuple[cs.SX, cs.SX]:
        assert name not in self.cons, f"Constraint {name} already exists."
        expr = expr1 - expr2
        dims = expr.shape
        if op in {"=", "=="}:
            is_eq = True
            lb, ub = np.zeros(dims), np.zeros(dims)
        elif op in {"<", "<="}:
            is_eq = False
            lb, ub = np.full(dims, -np.inf), np.zeros(dims)
        elif op in {">", ">="}:
            is_eq = False
            expr = -expr
            lb, ub = np.full(dims, -np.inf), np.zeros(dims)
        else:
            raise ValueError(f"Unrecognized operator {op}.")
        expr = cs.simplify(expr)
        lb, ub = cs.vec(lb).full().flatten(), cs.vec(ub).full().flatten()
        self.cons[name] = expr
        group = "g" if is_eq else "h"
        setattr(self, group, cs.vertcat(getattr(self, group), cs.vec(expr)))
        setattr(self, f"lb{group}", np.concatenate((getattr(self, f"lb{group}"), lb)))
        setattr(self, f"ub{group}", np.concatenate((getattr(self, f"ub{group}"), ub)))
        lam = cs.SX.sym(f"lam_{group}_{name}", *dims)
        setattr(
            self, f"lam_{group}", cs.vertcat(getattr(self, f"lam_{group}"), cs.vec(lam))
        )
        return expr, lam

    def minimize(self, objective: cs.SX) -> None:
        self.f = objective

    def init_solver(self, opts: dict) -> None:
        g = cs.vertcat(self.g, self.h)
        nlp = {"x": self.x, "p": self.p, "g": g, "f": self.f}
        self.solver = cs.nlpsol(f"nlpsol_{self.name}", "ipopt", nlp, opts)
        self.opts = opts

    def solve(
        self, pars: dict[str, np.ndarray], vals0: dict[str, np.ndarray] = None
    ) -> QuadRotorSolution:
        assert self.solver is not None, "Solver uninitialized."
        assert len(self.pars.keys() - pars.keys()) == 0, (
            "Trying to solve the MPC with unspecified parameters: "
            + ", ".join(self.pars.keys() - pars.keys())
            + "."
        )
        p = subsevalf(self.p, self.pars, pars)
        kwargs = {
            "p": p,
            "lbx": self.lbx,
            "ubx": self.ubx,
            "lbg": np.concatenate((self.lbg, self.lbh)),
            "ubg": np.concatenate((self.ubg, self.ubh)),
        }
        if vals0 is not None:
            kwargs["x0"] = np.clip(
                subsevalf(self.x, self.vars, vals0), self.lbx, self.ubx
            )
        sol: dict[str, cs.DM] = self.solver(**kwargs)
        lam_lbx = -np.minimum(sol["lam_x"], 0)
        lam_ubx = np.maximum(sol["lam_x"], 0)
        lam_g = sol["lam_g"][: self.ng, :]
        lam_h = sol["lam_g"][self.ng :, :]
        S = cs.vertcat(
            self.p, self.x, self.lam_g, self.lam_h, self.lam_lbx, self.lam_ubx
        )
        D = cs.vertcat(p, sol["x"], lam_g, lam_h, lam_lbx, lam_ubx)
        get_value = partial(subsevalf, old=S, new=D)
        vals = {name: get_value(var) for name, var in self.vars.items()}
        return QuadRotorSolution(
            f=float(sol["f"]),
            vars=self.vars.copy(),
            vals=vals,
            get_value=get_value,
            stats=self.solver.stats().copy(),
        )

    def __str__(self) -> str:
        msg = "not initialized" if self.solver is None else "initialized"
        C = len(self.cons)
        return (
            f"{type(self).__name__} {{\n"
            f"  name: {self.name}\n"
            f"  #variables: {len(self.vars)} (nx={self.nx})\n"
            f"  #parameters: {len(self.pars)} (np={self.np})\n"
            f"  #constraints: {C} (ng={self.ng}, nh={self.nh})\n"
            f"  CasADi solver {msg}.\n}}"
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}: {self.name}"


def subsevalf(
    expr: cs.SX,
    old: Union[cs.SX, dict[str, cs.SX], list[cs.SX], tuple[cs.SX]],
    new: Union[cs.SX, dict[str, cs.SX], list[cs.SX], tuple[cs.SX]],
    eval: bool = True,
) -> Union[cs.SX, np.ndarray]:
    if isinstance(old, dict):
        for name, o in old.items():
            expr = cs.substitute(expr, o, new[name])
    elif isinstance(old, (tuple, list)):
        for o, n in zip(old, new):
            expr = cs.substitute(expr, o, n)
    else:
        expr = cs.substitute(expr, old, new)

    if eval:
        expr = cs.evalf(expr).full().squeeze()
    return expr


ConfigType = TypeVar("ConfigType")


def init_config(
    config: Optional[Union[ConfigType, dict]], cls: type[ConfigType]
) -> ConfigType:
    if config is None:
        return cls()
    if isinstance(config, cls):
        return config
    if isinstance(config, dict):
        if not hasattr(cls, "__dataclass_fields__"):
            raise ValueError("Configiration class must be a dataclass.")
        keys = cls.__dataclass_fields__.keys()
        return cls(**{k: config[k] for k in keys if k in config})
    raise ValueError(
        "Invalid configuration type; expected None, dict or "
        f"a dataclass, got {cls} instead."
    )


@dataclass
class QuadRotorMPCConfig:
    N: int = 15
    solver_opts: dict = field(
        default_factory=lambda: {
            "expand": True,
            "print_time": False,
            "ipopt": {
                "max_iter": 500,
                "tol": 1e-6,
                "barrier_tol_factor": 1,
                "sb": "yes",
                # for debugging
                "print_level": 0,
                "print_user_options": "no",
                "print_options_documentation": "no",
            },
        }
    )


class QuadRotorMPC(GenericMPC):
    def __init__(
        self,
        env: QuadRotorEnv,
        config: Union[dict, QuadRotorMPCConfig] = None,
        mpctype: str = "V",
    ) -> None:
        assert mpctype in {
            "V",
            "Q",
        }, "MPC must be either V (state value func) or Q (action value func)"
        super().__init__(name=mpctype)
        self.config = init_config(config, QuadRotorMPCConfig)
        N = self.config.N

        # ======================= #
        # Variable and Parameters #
        # ======================= #
        lbx, ubx = env.config.x_bounds[:, 0], env.config.x_bounds[:, 1]
        not_red = ~(np.isneginf(lbx) & np.isposinf(ubx))
        not_red_idx = np.where(not_red)[0]
        lbx, ubx = lbx[not_red].reshape(-1, 1), ubx[not_red].reshape(-1, 1)
        nx, nu = env.nx, env.nu
        x, _, _ = self.add_var("x", nx, N)
        u, _, _ = self.add_var("u", nu, N)
        ns = not_red_idx.size + nu
        s, _, _ = self.add_var("slack", ns * N - not_red_idx.size, 1, lb=0)
        sx: cs.SX = s[: not_red_idx.size * (N - 1)].reshape((-1, N - 1))
        su: cs.SX = s[-nu * N :].reshape((-1, N))

        # 2) create model parameters
        for name in (
            "g",
            "thrust_coeff",
            "pitch_d",
            "pitch_dd",
            "pitch_gain",
            "roll_d",
            "roll_dd",
            "roll_gain",
        ):
            self.add_par(name, 1, 1)

        # =========== #
        # Constraints #
        # =========== #

        # 1) constraint on initial conditions
        x0 = self.add_par("x0", env.nx, 1)
        x_ = cs.horzcat(x0, x)

        # 2) constraints on dynamics
        A, B, e = env.get_dynamics(
            g=self.pars["g"],
            thrust_coeff=self.pars["thrust_coeff"],
            pitch_d=self.pars["pitch_d"],
            pitch_dd=self.pars["pitch_dd"],
            pitch_gain=self.pars["pitch_gain"],
            roll_d=self.pars["roll_d"],
            roll_dd=self.pars["roll_dd"],
            roll_gain=self.pars["roll_gain"],
        )
        self.add_con("dyn", x_[:, 1:], "==", A @ x_[:, :-1] + B @ u + e)

        # 3) constraint on state (soft, backed off, without infinity in g, and
        # removing redundant entries, no constraint on first state)
        # constraint backoff parameter and bounds
        bo = self.add_par("backoff", 1, 1)

        # set the state constraints as
        #  - soft-backedoff minimum constraint: (1+back)*lb - slack <= x
        #  - soft-backedoff maximum constraint: x <= (1-back)*ub + slack
        # NOTE: there is a mistake here in the old code, since we are excluding the
        # first state from constraints which is actually the second.
        self.add_con("x_min", (1 + bo) * lbx - sx, "<=", x[not_red_idx, 1:])
        self.add_con("x_max", x[not_red_idx, 1:], "<=", (1 - bo) * ubx + sx)
        self.add_con("u_min", env.config.u_bounds[:, 0] - su, "<=", u)
        self.add_con("u_max", u, "<=", env.config.u_bounds[:, 1] + su)

        # ========= #
        # Objective #
        # ========= #
        J = 0  # (no initial state cost not required since it is not economic)
        s = cs.blockcat([[cs.SX.zeros(sx.size1(), 1), sx], [su]])
        xf = self.add_par("xf", nx, 1)
        uf = cs.vertcat(0, 0, self.pars["g"])
        w_x = self.add_par("w_x", nx, 1)  # weights for stage/final state
        w_u = self.add_par("w_u", nu, 1)  # weights for stage/final control
        w_s = self.add_par("w_s", ns, 1)  # weights for stage/final slack
        J += sum(
            (
                cs.dot(w_x, (x[:, k] - xf) ** 2)
                + cs.dot(w_u, (u[:, k] - uf) ** 2)
                + cs.dot(w_s, s[:, k])
            )
            for k in range(N - 1)
        )
        J += (
            cs.dot(w_x, (x[:, -1] - xf) ** 2)
            + cs.dot(w_u, (u[:, -1] - uf) ** 2)
            + cs.dot(w_s, s[:, -1])
        )
        self.minimize(J)

        # ====== #
        # Others #
        # ====== #
        if mpctype == "Q":
            u0 = self.add_par("u0", nu, 1)
            self.add_con("init_action", u[:, 0], "==", u0)
        else:
            perturbation = self.add_par("perturbation", nu, 1)
            self.f += cs.dot(perturbation, u[:, 0])
        self.init_solver(self.config.solver_opts)


# =================================== end old code =================================== #


class QuadRotorMpcActual(Mpc[cs.SX]):
    def __init__(self, env: QuadRotorEnv, mpctype: str = "V") -> None:
        N = QuadRotorMPCConfig.N
        super().__init__(Nlp(sym_type="SX"), prediction_horizon=N, shooting="multi")

        # ======================= #
        # Variable and Parameters #
        # ======================= #
        lbx, ubx = env.config.x_bounds[:, 0], env.config.x_bounds[:, 1]
        not_red = ~(np.isneginf(lbx) & np.isposinf(ubx))
        not_red_idx = np.where(not_red)[0]
        lbx, ubx = lbx[not_red].reshape(-1, 1), ubx[not_red].reshape(-1, 1)
        nx, nu = env.nx, env.nu
        x, _ = self.state("x", nx)
        u, _ = self.action("u", nu)
        ns = not_red_idx.size + nu
        s, _, _ = self.variable("slack", (ns * N - not_red_idx.size, 1), lb=0)
        sx: cs.SX = s[: not_red_idx.size * (N - 1)].reshape((-1, N - 1))
        su: cs.SX = s[-nu * N :].reshape((-1, N))

        # 2) create model parameters
        for name in (
            "g",
            "thrust_coeff",
            "pitch_d",
            "pitch_dd",
            "pitch_gain",
            "roll_d",
            "roll_dd",
            "roll_gain",
        ):
            self.parameter(name, (1, 1))

        # =========== #
        # Constraints #
        # =========== #
        A, B, e = env.get_dynamics(
            g=self.parameters["g"],
            thrust_coeff=self.parameters["thrust_coeff"],
            pitch_d=self.parameters["pitch_d"],
            pitch_dd=self.parameters["pitch_dd"],
            pitch_gain=self.parameters["pitch_gain"],
            roll_d=self.parameters["roll_d"],
            roll_dd=self.parameters["roll_dd"],
            roll_gain=self.parameters["roll_gain"],
        )
        self.set_nonlinear_dynamics(lambda x, u: A @ x + B @ u + e)

        # 3) constraint on state
        bo = self.parameter("backoff", (1, 1))
        self.constraint("x_min", (1 + bo) * lbx - sx, "<=", x[not_red_idx, 2:])
        self.constraint("x_max", x[not_red_idx, 2:], "<=", (1 - bo) * ubx + sx)
        self.constraint("u_min", env.config.u_bounds[:, 0] - su, "<=", u)
        self.constraint("u_max", u, "<=", env.config.u_bounds[:, 1] + su)

        # ========= #
        # Objective #
        # ========= #
        J = 0  # (no initial state cost not required since it is not economic)
        s = cs.blockcat([[cs.SX.zeros(sx.size1(), 1), sx], [su]])
        xf = self.parameter("xf", (nx, 1))
        uf = cs.vertcat(0, 0, self.parameters["g"])
        w_x = self.parameter("w_x", (nx, 1))  # weights for stage/final state
        w_u = self.parameter("w_u", (nu, 1))  # weights for stage/final control
        w_s = self.parameter("w_s", (ns, 1))  # weights for stage/final slack
        J += sum(
            (
                cs.dot(w_x, (x[:, k + 1] - xf) ** 2)
                + cs.dot(w_u, (u[:, k] - uf) ** 2)
                + cs.dot(w_s, s[:, k])
            )
            for k in range(N - 1)
        )
        J += (
            cs.dot(w_x, (x[:, -1] - xf) ** 2)
            + cs.dot(w_u, (u[:, -1] - uf) ** 2)
            + cs.dot(w_s, s[:, -1])
        )

        # ====== #
        # Others #
        # ====== #
        if mpctype == "Q":
            u0 = self.parameter("u0", (nu, 1))
            self.constraint("init_action", u[:, 0], "==", u0)
        else:
            perturbation = self.parameter("perturbation", (nu, 1))
            J += cs.dot(perturbation, u[:, 0])
        self.minimize(J)
        self.init_solver(
            QuadRotorMPCConfig.__dataclass_fields__["solver_opts"].default_factory()
        )


class TestQuadRotorMpc(unittest.TestCase):
    @parameterized.expand([("V", 10), ("Q", 10)])
    def test(self, mpctype: str, N: int):
        env = QuadRotorEnv()
        mpc_expected = QuadRotorMPC(mpctype=mpctype, env=env)
        mpc_actual = QuadRotorMpcActual(mpctype=mpctype, env=env)
        sol_expected = sol_actual = None
        kwargs = {"rtol": 1e-7, "atol": 1e-7}
        for _ in range(N):
            if sol_expected is None or sol_actual is None:
                vals_expected = vals_actual = {
                    "x": 0,
                    "u": 0,
                    "slack": 0,
                }
            else:
                vals_expected = sol_expected.vals
                vals_actual = sol_actual.vals
            pars = {
                "g": np.random.rand() * 3 + 9,
                "thrust_coeff": np.random.rand() * 2 + 0.5,
                "pitch_d": np.random.rand() * 3 + 9,
                "pitch_dd": np.random.rand() * 4 + 8,
                "pitch_gain": np.random.rand() * 5 + 7,
                "roll_d": np.random.rand() * 3 + 9,
                "roll_dd": np.random.rand() * 4 + 8,
                "roll_gain": np.random.rand() * 5 + 7,
                "x0": np.array([0, 0, 3.5, 0, 0, 0, 0, 0, 0, 0]).reshape(-1, 1)
                + np.random.rand(10, 1) * 1,
                "u0": np.random.rand(3, 1) * 1,
                "backoff": np.random.rand() * 0.1,
                "xf": np.array([3, 3, 0.2, 0, 0, 0, 0, 0, 0, 0]).reshape(-1, 1)
                + np.random.rand(10, 1) * 0.1,
                "w_x": np.random.rand(10, 1) * 5 + 10,
                "w_u": np.random.rand(3, 1) * 5 + 10,
                "w_s": np.random.rand(8, 1) * 5 + 10,
                "perturbation": np.random.randn(3, 1) * 0.5,
            }
            pars["x_0"] = pars["x0"]
            sol_expected = mpc_expected.solve(pars, vals_expected)
            sol_actual: Solution[cs.SX] = mpc_actual.solve(pars, vals_actual)

            self.assertTrue(sol_expected.success)
            self.assertTrue(sol_actual.success)
            np.testing.assert_allclose(sol_actual.f, sol_expected.f, **kwargs)
            np.testing.assert_allclose(
                sol_actual.vals["x"][:, 1:], sol_expected.vals["x"], **kwargs
            )
            np.testing.assert_allclose(
                sol_actual.vals["u"], sol_expected.vals["u"], **kwargs
            )
            np.testing.assert_allclose(
                sol_actual.vals["slack"].full().flatten(),
                sol_expected.vals["slack"],
                **kwargs,
            )


if __name__ == "__main__":
    unittest.main()
