from inspect import signature
from typing import Callable, Literal, Optional, TypeVar, Union

import casadi as cs
import numpy as np
import numpy.typing as npt

from csnlp.multistart.multistart_nlp import _chained_substitute, _n

from ..wrapper import Nlp
from .mpc import Mpc, _callable2csfunc, _create_ati_mats
from .mpc import _n as _name_init_state

SymType = TypeVar("SymType", cs.SX, cs.MX)
MatType = TypeVar("MatType", SymType, cs.DM, np.ndarray)


class ScenarioBasedMpc(Mpc[SymType]):
    """Implementation of the Scenario-based Model Predictive Control
    :cite:`schildbach_scenario_2014`, here referred to as SCMPC, a well-known stochastic
    MPC formulation.

    Parameters
    ----------
    nlp : Nlp
        NLP scheme to be wrapped
    n_scenarios : int
        Number of scenarios to be considered in the scenario-based MPC formulation. Must
        be a positive integer.
    prediction_horizon : int
        A positive integer for the prediction horizon of the MPC controller.
    control_horizon : int, optional
        A positive integer for the control horizon of the MPC controller. If not given,
        it is set equal to the control horizon.
    input_spacing : int, optional
        Spacing between independent input actions. This argument allows to reduce the
        number of free actions along the control horizon by allowing only the first
        action every ``n`` to be free, and the following ``n-1`` to be fixed equal to
        that action (where ``n`` is given by ``input_spacing``). By default, no spacing
        is allowed, i.e., ``1``.
    shooting : 'single' or 'multi', optional
        Type of approach in the direct shooting for parametrizing the control
        trajectory. See Section 8.5 in :cite:`rawlings_model_2017`. By default, direct
        shooting is used.

    Raises
    ------
    ValueError
        Raises if the shooting method is invalid; or if any of the horizons are invalid;
        or if the number of scenarios is not a positive integer.
    """

    def __init__(
        self,
        nlp: Nlp[SymType],
        n_scenarios: int,
        prediction_horizon: int,
        control_horizon: Optional[int] = None,
        input_spacing: int = 1,
        shooting: Literal["single", "multi"] = "multi",
    ) -> None:
        if n_scenarios < 1:
            raise ValueError("The number of scenarios must be a positive integer.")
        super().__init__(
            nlp, prediction_horizon, control_horizon, input_spacing, shooting
        )
        self._n_scenarios = n_scenarios
        self.single_states: dict[str, SymType] = {}
        self.single_disturbances: dict[str, SymType] = {}
        self.single_slacks: dict[str, SymType] = {}

    @property
    def n_scenarios(self) -> int:
        """Gets the number of scenarios."""
        return self._n_scenarios

    @property
    def ns_all(self) -> int:
        """Gets the number of states of the SCMPC controller considering all
        scenarios."""
        return self.ns * self._n_scenarios

    @property
    def nd(self) -> int:
        return super().nd // self._n_scenarios

    @property
    def nd_all(self) -> int:
        """Gets the number of disturbances of the SCMPC controller considering all
        scenarios."""
        return super().nd

    def name_i(self, base_name: str, i: int) -> str:
        """Gets the name of the i-th scenario."""
        return _n(base_name, i)

    def states_i(self, i: int) -> dict[str, SymType]:
        """Gets the symbolic states belonging to the i-th scenario."""
        return {n: self.states[_n(n, i)] for n in self.single_states}

    def slacks_i(self, i: int) -> list[SymType]:
        """Gets the symbolic slack variables belonging to the i-th scenario."""
        return {n: self.slacks[_n(n, i)] for n in self.single_slacks}

    def disturbances_i(self, i: int) -> dict[str, SymType]:
        """Gets the symbolic disturbances belonging to the i-th scenario."""
        return {n: self.disturbances[_n(n, i)] for n in self.single_disturbances}

    def state(
        self,
        name: str,
        size: int = 1,
        discrete: bool = False,
        lb: Union[npt.ArrayLike, cs.DM] = -np.inf,
        ub: Union[npt.ArrayLike, cs.DM] = +np.inf,
        bound_initial: bool = True,
        bound_terminal: bool = True,
    ) -> tuple[SymType, list[Optional[SymType]], SymType]:
        """Adds one state variable per scenario to the SCMPC controller. Automatically
        creates the (shared) constraint on the initial conditions for these states.

        Parameters
        ----------
        name : str
            Name of the state.
        size : int
            Size of the state (assumed to be a vector).
        discrete : bool, optional
            Flag indicating if the state is discrete. Defaults to ``False``.
        lb : array_like, casadi.DM, optional
            Hard lower bound of the state, by default ``-np.inf``.
        ub : array_like, casadi.DM, optional
            Hard upper bound of the state, by default ``+np.inf``.
        bound_initial : bool, optional
            If ``False``, then the upper and lower bounds on the initial state are not
            imposed, i.e., set to ``+/- np.inf`` (since the initial state is constrained
            to be equal to the current state of the system, it is sometimes advantageous
            to remove its bounds). By default ``True``.
        bound_terminal : bool, optional
            Same as above, but for the terminal state. By default ``True``.

        Returns
        -------
        single state : casadi.SX or MX
            Symbol corresponding to the state of a single scenario. This is useful for
            automatically defining, e.g., the objective and constraints over the various
            scenarios of the SCMPC controller, but it is not used in the actual NLP
            solver.
        states : list of casadi.SX or MX or None
            The list of the state symbolic variable. If `shooting=single`, then
            `None` is returned since the states will only be available once the dynamics
            are set.
        initial state : casadi.SX or MX
            The initial state symbolic parameter.

        Raises
        ------
        ValueError
            Raises if there exists already a state with the same name.
        RuntimeError
            Raises in single shooting if lower or upper bounds have been specified,
            since these can only be set after the dynamics have been set via the
            `constraint` method.
        """
        N = self._prediction_horizon
        if self._is_multishooting:
            shape = (size, N + 1)
            lb = np.broadcast_to(lb, shape).astype(float)
            ub = np.broadcast_to(ub, shape).astype(float)
            if not bound_initial:
                lb[:, 0] = -np.inf
                ub[:, 0] = +np.inf
            if not bound_terminal:
                lb[:, -1] = -np.inf
                ub[:, -1] = +np.inf
        elif np.any(lb != -np.inf) or np.any(ub != +np.inf):
            raise RuntimeError(
                "in single shooting, lower and upper state bounds can only be "
                "created after the dynamics have been set"
            )

        # create as many states as scenarions, but only one initial state
        x0_name = _name_init_state(name)
        x0 = self.nlp.parameter(x0_name, (size, 1))
        self._initial_states[x0_name] = x0

        xs = []
        for i in range(self._n_scenarios):
            name_i = _n(name, i)
            if self._is_multishooting:
                x_i = self.nlp.variable(name_i, shape, discrete, lb, ub)[0]
                self.nlp.constraint(_n(x0_name, i), x_i[:, 0], "==", x0)
            else:
                x_i = None
            xs.append(x_i)
            self._states[name_i] = x_i

        # create also a single symbol for the state
        x_single = self.nlp._sym_type.sym(name, size, N + 1)
        self.single_states[name] = x_single
        return x_single, xs, x0

    def disturbance(self, name: str, size: int = 1) -> tuple[SymType, list[SymType]]:
        """Adds one disturbance parameter per scenario to the SCMPC controller along the
        whole prediction horizon.

        Parameters
        ----------
        name : str
            Name of the disturbance.
        size : int, optional
            Size of the disturbance (assumed to be a vector). Defaults to ``1``.

        Returns
        -------
        single disturbance : casadi.SX or MX
            Symbol corresponding to the disturbance of a single scenario. See the note
            for :meth:``state``.
        disturbances : list of casadi.SX or MX
            The symbols for the new disturbances of each scenario in the SCMPC
            controller.
        """
        shape = (size, self._prediction_horizon)
        ds = []
        for i in range(self._n_scenarios):
            name_i = _n(name, i)
            d_i = self.nlp.parameter(name_i, shape)
            ds.append(d_i)
            self._disturbances[name_i] = d_i
        d_single = self.nlp._sym_type.sym(name, shape)
        self.single_disturbances[name] = d_single
        return d_single, ds

    def constraint_from_single(
        self,
        name: str,
        lhs: Union[SymType, np.ndarray, cs.DM],
        op: Literal["==", ">=", "<="],
        rhs: Union[SymType, np.ndarray, cs.DM],
        soft: bool = False,
        simplify: bool = True,
    ) -> tuple[list[SymType], ...]:
        """Similarly to :meth:`csnlp.wrappers.Mpc.constraint`, adds a constraint to the
        MPC scheme. However, instead of manually creating the constraint for each
        scenario, this method allows to define only one constraint expression for a
        single scenario, which is then automatically declined for all scenarios. The
        symbolical expression must be made up of the single scenario states and
        disturbances, returned as first output by the methods :meth:`state` and
        :meth:`disturbance`, respectively. Note that the return types are lists of
        symbolical variables.

        Returns
        -------
        exprs : list of casadi.SX or MX
            The constraint expression in canonical form, i.e., :math:`g(x,u) = 0` or
            :math:`h(x,u) <= 0`, for each scenario.
        lams : list of casadi.SX or MX
            The symbol corresponding to the constraint's multipliers, for each scenario.
        single slack : casadi.SX or MX
            Symbol corresponding to the slack from a single scenario. This is useful for
            automatically defining, e.g., the objective over the various scenarios of
            the SCMPC controller, but it is not used in the actual NLP
            solver. Only returned if `soft=True`; otherwise, only a 2-tuple is returned.
        slacks : list of casadi.SX or MX, optional
            Each scenario's slack variable in case of `soft=True`; otherwise, only a
            2-tuple is returned.
        """
        expr = lhs - rhs
        if simplify:
            expr = cs.cse(cs.simplify(expr))

        cons = []
        lams = []
        if soft:
            slacks = []
        for i in range(self._n_scenarios):
            expr_i = self._chained_substitution_for_scenario_i(expr, i)
            out = self.constraint(_n(name, i), expr_i, op, 0, soft)
            cons.append(out[0])
            lams.append(out[1])
            if soft:
                slacks.append(out[2])

        if soft:
            slack_name = f"slack_{name}"
            single_slack = self.nlp.sym_type.sym(slack_name, expr.shape)
            self.single_slacks[slack_name] = single_slack
            return cons, lams, single_slack, slacks
        return cons, lams

    def minimize_from_single(self, objective: SymType) -> None:
        """Similarly to :meth:`csnlp.Nlp.minimize`, adds the objective to be minimized
        to the NLP scheme. However, instead of manually creating the objective for each
        scenario, this method allows to define only one expression for a single
        scenario, which is then automatically declined and summed for all scenarios. The
        symbolical expression must be made up of the single scenario states,
        disturbances, and slacks, returned as first output by the methods :meth:`state`,
        :meth:`disturbance`, and :meth:`constraint_from_single`, respectively.
        """
        return self.nlp.minimize(
            sum(
                self._chained_substitution_for_scenario_i(objective, i)
                for i in range(self._n_scenarios)
            )
            / self._n_scenarios
        )

    def set_affine_dynamics(
        self,
        A: MatType,
        B: MatType,
        D: MatType,
        c: Optional[MatType] = None,
        parallelization: Literal[
            "serial", "unroll", "inline", "thread", "openmp"
        ] = "thread",
        max_num_threads: Optional[int] = None,
    ) -> tuple[
        Optional[MatType], Optional[MatType], Optional[MatType], Optional[MatType]
    ]:
        if D is None:
            raise ValueError(
                "The dynamics matrix D must be given, as SCMPC is a tool to account for"
                " stochastic disturbances, and if there are none, a nominal MPC should "
                "suffice (see `Mpc` wrapper)."
            )
        if max_num_threads is None:
            max_num_threads = max(self._prediction_horizon, self._n_scenarios)
        return super().set_affine_dynamics(A, B, D, c, parallelization, max_num_threads)

    def set_nonlinear_dynamics(
        self,
        F: Union[
            cs.Function,
            Callable[[tuple[npt.ArrayLike, ...]], tuple[npt.ArrayLike, ...]],
        ],
        parallelization: Literal[
            "serial", "unroll", "inline", "thread", "openmp"
        ] = "thread",
        max_num_threads_or_unrolling_base: Optional[int] = None,
    ) -> None:
        n_in = F.n_in() if isinstance(F, cs.Function) else len(signature(F).parameters)
        nd = self.nd
        if n_in != 3 or nd == 0:
            raise ValueError(
                "The dynamics function must have 3 arguments: the state, the action, "
                "and the disturbance. This is because SCMPC is a tool to account for "
                "stochastic disturbances, and if there are none, a nominal MPC should "
                "suffice (see `Mpc` wrapper)."
            )

        if not isinstance(F, cs.Function):
            F = _callable2csfunc(F, self.nlp.sym_type, (self.ns, self.na, nd))

        if max_num_threads_or_unrolling_base is None:
            max_num_threads_or_unrolling_base = max(
                self._prediction_horizon, self._n_scenarios
            )

        return super().set_nonlinear_dynamics(
            F, parallelization, max_num_threads_or_unrolling_base
        )

    def _set_singleshooting_affine_dynamics(
        self, A: MatType, B: MatType, D: MatType, c: Optional[MatType]
    ) -> tuple[MatType, MatType, Optional[MatType], Optional[MatType]]:
        disturbance_names = self.single_disturbances.keys()
        X0 = cs.vcat(self._initial_states.values())
        U = cs.vec(cs.vcat(self._actions_exp.values()))  # NOTE: different from vvcat!
        D_all = cs.hcat(
            [
                cs.vec(
                    cs.vcat([self._disturbances[_n(n, i)] for n in disturbance_names])
                )
                for i in range(self._n_scenarios)
            ]
        )

        F, G, H, L = _create_ati_mats(self._prediction_horizon, A, B, D, c)
        X_next_pred = F @ X0 + G @ U + H @ D_all
        if L is not None:
            X_next_pred += L

        state_names = self.single_states.keys()
        N = self._prediction_horizon
        ns = A.shape[0]
        cumsizes = np.cumsum([0] + [s.shape[0] for s in self._initial_states.values()])
        for i in range(self._n_scenarios):
            X_i = cs.vertcat(X0, X_next_pred[:, i]).reshape((ns, N + 1))
            X_i_split = cs.vertsplit(X_i, cumsizes)
            for n, x in zip(state_names, X_i_split):
                self._states[_n(n, i)] = x
        return F, G, H, L

    def _set_multishooting_nonlinear_dynamics(
        self,
        F: cs.Function,
        _: int,
        parallelization: Literal["serial", "unroll", "inline", "thread", "openmp"],
        max_num_threads: int,
    ) -> None:
        state_names = self.single_states.keys()
        disturbance_names = self.single_disturbances.keys()
        U = cs.vcat(self._actions_exp.values())
        X_all_ = []
        X_all_next_ = []
        D_all_ = []
        for i in range(self._n_scenarios):
            X_i = cs.vcat([self._states[_n(n, i)] for n in state_names])
            X_all_.append(X_i[:, :-1])
            X_all_next_.append(X_i[:, 1:])
            D_all_.append(
                cs.vcat([self._disturbances[_n(n, i)] for n in disturbance_names])
            )
        X_all = cs.hcat(X_all_)
        X_all_next = cs.hcat(X_all_next_)
        D_all = cs.hcat(D_all_)

        Fmap = F.map(
            self._prediction_horizon * self._n_scenarios,
            parallelization,
            max_num_threads,
        )
        X_all_next_pred = Fmap(X_all, U, D_all)
        self.constraint("dyn", X_all_next, "==", X_all_next_pred)

    def _set_singleshooting_nonlinear_dynamics(
        self, F: cs.Function, _: int, base: int
    ) -> None:
        disturbance_names = self.single_disturbances.keys()
        X0 = cs.vcat(self._initial_states.values())
        U = cs.vcat(self._actions_exp.values())
        D_all = cs.hcat(
            [
                cs.vcat([self._disturbances[_n(n, i)] for n in disturbance_names])
                for i in range(self._n_scenarios)
            ]
        )

        N = self._prediction_horizon
        Fmapaccum = F.mapaccum(N, {"base": base, "allow_free": True})
        X_next_hcat = Fmapaccum(X0, U, D_all)

        state_names = self.single_states.keys()
        cumsizes = np.cumsum([0] + [s.shape[0] for s in self._initial_states.values()])
        X_next_split = cs.horzsplit(X_next_hcat, N)
        for i in range(self._n_scenarios):
            X_i = cs.horzcat(X0, X_next_split[i])
            X_i_split = cs.vertsplit(X_i, cumsizes)
            for n, x in zip(state_names, X_i_split):
                self._states[_n(n, i)] = x

    def _chained_substitution_for_scenario_i(
        self, expr: SymType, i: int
    ) -> Union[SymType, cs.DM]:
        """Iternal utility to perform substitutions in chain for the i-th scenario."""
        return _chained_substitute(
            expr,
            (self.single_states, self.states_i(i)),
            (self.single_disturbances, self.disturbances_i(i)),
            (self.single_slacks, self.slacks_i(i)),
        )
