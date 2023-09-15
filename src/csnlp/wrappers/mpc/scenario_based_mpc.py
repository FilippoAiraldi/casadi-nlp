from itertools import chain
from typing import Callable, Dict, List, Literal, Optional, Tuple, TypeVar, Union

import casadi as cs
import numpy as np
import numpy.typing as npt

from csnlp.multistart.multistart_nlp import _chained_subevalf, _n
from csnlp.wrappers.mpc.mpc import Mpc
from csnlp.wrappers.mpc.mpc import _n as _name_init_state
from csnlp.wrappers.wrapper import Nlp

SymType = TypeVar("SymType", cs.SX, cs.MX)


class ScenarioBasedMpc(Mpc[SymType]):
    """Implementation of the Scenario-based Model Predictive Control [1], here referred
    to as SCMPC, a well-known stochastic MPC formulation.

    References
    ----------
    [1] Schildbach, G., Fagiano, L., Frei, C. and Morari, M., 2014. The scenario
        approach for stochastic model predictive control with bounds on closed-loop
        constraint violations. Automatica, 50(12), pp.3009-3018.
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
        """Initializes the Scenarion-based MPC (SCMPC) wrapper around the NLP instance.

        Parameters
        ----------
        nlp : Nlp
            NLP scheme to be wrapped
        n_scenarios : int
            Number of scenarios to be considered in the scenario-based MPC formulation.
            Must be a positive integer.
        prediction_horizon : int
            A positive integer for the prediction horizon of the MPC controller.
        control_horizon : int, optional
            A positive integer for the control horizon of the MPC controller. If not
            given, it is set equal to the control horizon.
        input_spacing : int, optional
            Spacing between independent input actions. This argument allows to reduce
            the number of free actions along the control horizon by allowing only the
            first action every `n` to be free, and the following `n-1` to be fixed equal
            to that action (where `n` is given by `input_spacing`). By default, no
            spacing is allowed, i.e., 1.
        shooting : 'single' or 'multi', optional
            Type of approach in the direct shooting for parametrizing the control
            trajectory. See [1, Section 8.5]. By default, direct shooting is used.

        Raises
        ------
        ValueError
            Raises if the shooting method is invalid; or if any of the horizons are
            invalid; or if the number of scenarios is not a positive integer.

        References
        ----------
        [1] Rawlings, J.B., Mayne, D.Q. and Diehl, M., 2017. Model Predictive Control:
            theory, computation, and design (Vol. 2). Madison, WI: Nob Hill Publishing.
        """
        if n_scenarios < 1:
            raise ValueError("The number of scenarios must be a positive integer.")
        super().__init__(
            nlp, prediction_horizon, control_horizon, input_spacing, shooting
        )
        self._n_scenarios = n_scenarios
        self.single_states: Dict[str, SymType] = {}
        self.single_disturbances: Dict[str, SymType] = {}
        self.single_slacks: Dict[str, SymType] = {}

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

    def states_i(self, i: int) -> Dict[str, SymType]:
        """Gets the symbolic states belonging to the i-th scenario."""
        return {n: self.states[_n(n, i)] for n in self.single_states}

    def slacks_i(self, i: int) -> List[SymType]:
        """Gets the symbolic slack variables belonging to the i-th scenario."""
        return {n: self.slacks[_n(n, i)] for n in self.single_slacks}

    def disturbances_i(self, i: int) -> Dict[str, SymType]:
        """Gets the symbolic disturbances belonging to the i-th scenario."""
        return {n: self.disturbances[_n(n, i)] for n in self.single_disturbances}

    def state(
        self,
        name: str,
        size: int = 1,
        lb: Union[npt.ArrayLike, cs.DM] = -np.inf,
        ub: Union[npt.ArrayLike, cs.DM] = +np.inf,
        remove_bounds_on_initial: bool = False,
    ) -> Tuple[SymType, List[Optional[SymType]], SymType]:
        """Adds one state variable per scenario to the SCMPC controller. Automatically
        creates the (shared) constraint on the initial conditions for these states.

        Parameters
        ----------
        name : str
            Name of the state.
        size : int
            Size of the state (assumed to be a vector).
        lb : array_like, casadi.DM, optional
            Hard lower bound of the state, by default -np.inf.
        ub : array_like, casadi.DM, optional
            Hard upper bound of the state, by default +np.inf.
        remove_bounds_on_initial : bool, optional
            If `True`, then the upper and lower bounds on the initial state are removed,
            i.e., set to `+/- np.inf` (since the initial state is constrained to be
            equal to the current state of the system, it is sometimes advantageous to
            remove its bounds). By default `False`.

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
        if self._is_multishooting:
            shape = (size, self._prediction_horizon + 1)
            if remove_bounds_on_initial:
                lb = np.broadcast_to(lb, shape).astype(float)
                lb[:, 0] = -np.inf
                ub = np.broadcast_to(ub, shape).astype(float)
                ub[:, 0] = +np.inf
        elif np.any(lb != -np.inf) or np.any(ub != +np.inf):
            raise RuntimeError(
                "in single shooting, lower and upper state bounds can only be "
                "created after the dynamics have been set"
            )
        else:
            shape = (size, 1)

        # create as many states as scenarions, but only one initial state
        x0_name = _name_init_state(name)
        x0 = self.nlp.parameter(x0_name, (size, 1))
        self._initial_states[x0_name] = x0

        xs = []
        for i in range(self._n_scenarios):
            name_i = _n(name, i)
            if self._is_multishooting:
                x_i = self.nlp.variable(name_i, shape, lb, ub)[0]
                self.nlp.constraint(_n(x0_name, i), x_i[:, 0], "==", x0)
            else:
                x_i = None
            xs.append(x_i)
            self._states[name_i] = x_i

        # create also a single symbol for the state
        x_single = (
            self.nlp._sym_type.sym(name, shape) if self._is_multishooting else None
        )
        self.single_states[name] = x_single
        return x_single, xs, x0

    def disturbance(self, name: str, size: int = 1) -> Tuple[SymType, List[SymType]]:
        """Adds one disturbance parameter per scenario to the SCMPC controller along the
        whole prediction horizon.

        Parameters
        ----------
        name : str
            Name of the disturbance.
        size : int, optional
            Size of the disturbance (assumed to be a vector). Defaults to 1.

        Returns
        -------
        single disturbance : casadi.SX or MX
            Symbol corresponding to the disturbance of a single scenario. See the note
            for method `state`.
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
    ) -> Tuple[List[SymType], ...]:
        """Similarly to `Nlp.constraint`, adds a constraint to the NLP scheme. However,
        instead of manually creating the constraint for each scenario, this method
        allows to define only one constraint expression for a single scenario, which is
        then automatically declined for all scenarios. The symbolical expression must
        be made up of the single scenario states and disturbances, returned as first
        output by the methods `state` and `disturbance`, respectively.

        Note that the return types are list of symbolical variables.
        Returns
        -------
        exprs : list of casadi.SX or MX
            The constraint expression in canonical form, i.e., `g(x,u) = 0` or
            `h(x,u) <= 0`, for each scenario.
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
            expr_i = _chained_subevalf(
                expr,
                self.single_states,
                self.states_i(i),
                self.single_disturbances,
                self.disturbances_i(i),
                eval=False,
            )

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
        """Similarly to `Nlp.minimize`, adds the objective to be minimized to the NLP
        scheme. However, instead of manually creating the objective for each scenario,
        this method allows to define only one expression for a single scenario, which is
        then automatically declined and summed for all scenarios. The symbolical
        expression must be made up of the single scenario states, disturbances, and
        slacks, returned as first output by the methods `state`, `disturbance`, and
        `constraint_from_single`, respectively.
        """
        objective = cs.simplify(cs.cse(objective)) / self._n_scenarios
        return self.nlp.minimize(
            sum(
                _chained_subevalf(
                    objective,
                    self.single_states,
                    self.states_i(i),
                    self.single_disturbances,
                    self.disturbances_i(i),
                    self.single_slacks,
                    self.slacks_i(i),
                    eval=False,
                )
                for i in range(self._n_scenarios)
            )
        )

    def set_dynamics(
        self,
        F: Union[
            cs.Function,
            Callable[[Tuple[npt.ArrayLike, ...]], Tuple[npt.ArrayLike, ...]],
        ],
        n_in: Optional[int] = None,
        n_out: Optional[int] = None,
    ) -> None:
        if isinstance(F, cs.Function):
            n_in = F.n_in()
        if n_in != 3 or self.nd == 0:
            raise ValueError(
                "The dynamics function must have 3 arguments: the state, the action, "
                "and the disturbance. This is because SCMPC is a tool to account for "
                "stochastic disturbances, and if there are none, a nominal MPC should "
                "suffice (see `Mpc` wrapper)."
            )
        return super().set_dynamics(F, n_in, n_out)

    def _multishooting_dynamics(self, F: cs.Function, _: int, n_out: int) -> None:
        state_names = self.single_states.keys()
        disturbance_names = self.single_disturbances.keys()
        U = cs.vcat(self._actions_exp.values())
        for i in range(self._n_scenarios):
            X_i = cs.vcat([self._states[_n(n, i)] for n in state_names])
            D_i = cs.vcat([self._disturbances[_n(n, i)] for n in disturbance_names])
            xs_i_next = []
            for k in range(self._prediction_horizon):
                x_i_next = F(X_i[:, k], U[:, k], D_i[:, k])
                if n_out != 1:
                    x_i_next = x_i_next[0]
                xs_i_next.append(x_i_next)
            self.constraint(_n("dyn", i), cs.hcat(xs_i_next), "==", X_i[:, 1:])

    def _singleshooting_dynamics(self, F: cs.Function, _: int, n_out: int) -> None:
        disturbance_names = self.single_disturbances.keys()
        Xk_shared = cs.vcat(self._initial_states.values())
        cumsizes = np.cumsum([0] + [s.shape[0] for s in self._initial_states.values()])
        U = cs.vcat(self._actions_exp.values())
        state_names = self.single_states.keys()
        propagated_states = []
        for i in range(self._n_scenarios):
            Xk_i = Xk_shared
            D_i = cs.vcat([self._disturbances[_n(n, i)] for n in disturbance_names])
            X_i = [Xk_i]
            for k in range(self._prediction_horizon):
                Xk_i = F(Xk_i, U[:, k], D_i[:, k])
                if n_out != 1:
                    Xk_i = Xk_i[0]
                X_i.append(Xk_i)
            X_i = cs.vertsplit(cs.hcat(X_i), cumsizes)
            propagated_states.append([(_n(n, i), x) for n, x in zip(state_names, X_i)])
        self._states = dict(chain.from_iterable(propagated_states))
