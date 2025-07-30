from math import ceil
from typing import Callable, Literal, Optional, TypeVar, Union

import casadi as cs
import numpy as npy
import numpy.typing as npt

from csnlp.multistart.multistart_nlp import _chained_substitute, _n

from ...util.math import repeat
from ..wrapper import Nlp
from .mpc import Mpc, _create_ati_mats
from .scenario_based_mpc import ScenarioBasedMpc

SymType = TypeVar("SymType", cs.SX, cs.MX)
MatType = TypeVar("MatType", SymType, cs.DM, npy.ndarray)


class MultiScenarioMpc(ScenarioBasedMpc[SymType]):
    """Multi-scenario model predictive control (MSMPC) wrapper for a :class:`csnlp.Nlp`
    scheme. It generlizes :class:`csnlp.wrappers.ScenarioBasedMpc` to allow for multiple
    scenarios, each with its own state trajectory, action sequence (a number of these
    actions can be shared across all scenarios; see ``input_sharing``), and parameters.

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
    input_sharing : int, optional
        Number of actions that are shared across all scenarios. If ``0``, each scenario
        has its own action trajectory; if ``n``, all scenarios share the same first
        ``n`` actions. Must not be larger than the number of free actions in the
        control horizon. By default, ``1``, so only the first action is shared across
        the scenarios.
    shooting : 'single' or 'multi', optional
        Type of approach in the direct shooting for parametrizing the control
        trajectory. See Section 8.5 in :cite:`rawlings_model_2017`. By default, direct
        shooting is used.

    Raises
    ------
    ValueError
        Raises if the shooting method is invalid; or if any of the horizons are invalid;
        or if the number of scenarios is not a positive integer; or if
        ``input_sharing`` is not a valid integer (nonnegative and not larger than the
        number of free actions in the control horizon).
    """

    def __init__(
        self,
        nlp: Nlp[SymType],
        n_scenarios: int,
        prediction_horizon: int,
        control_horizon: Optional[int] = None,
        input_spacing: int = 1,
        input_sharing: int = 1,
        shooting: Literal["single", "multi"] = "multi",
    ) -> None:
        super().__init__(
            nlp,
            n_scenarios,
            prediction_horizon,
            control_horizon,
            input_spacing,
            shooting,
        )

        nu_free = ceil(self.control_horizon / self._input_spacing)
        if not isinstance(input_sharing, (int, npy.integer, npy.int_)) or not (
            0 <= input_sharing <= nu_free
        ):
            raise ValueError(
                f"`input_sharing` in range [0, {nu_free}]; got {input_sharing} instead."
            )
        self._input_sharing = input_sharing

        self.single_parameters: dict[str, SymType] = {}
        self.single_actions: dict[str, SymType] = {}
        self.single_actions_exp: dict[str, SymType] = {}

    @property
    def na(self) -> int:
        """Number of actions in a single scenario in the NLP scheme."""
        return super().na // self._n_scenarios

    @property
    def na_all(self) -> int:
        """Gets the number of actions of the MSMPC controller considering all
        scenarios."""
        return super().na

    @property
    def np(self) -> int:
        """Number of parameters in a single scenario in the NLP scheme."""
        return self.nlp.np // self._n_scenarios

    @property
    def np_all(self) -> int:
        """Gets the number of parameters of the MSMPC controller considering all
        scenarios."""
        return self.nlp.np

    def parameters_i(self, i: int) -> dict[str, SymType]:
        """Gets the symbolic parameters belonging to the i-th scenario."""
        return {n: self.parameters[_n(n, i)] for n in self.single_parameters}

    def actions_i(self, i: int) -> dict[str, SymType]:
        """Gets the symbolic actions belonging to the i-th scenario."""
        return {n: self.actions[_n(n, i)] for n in self.single_actions_exp}

    def actions_expanded_i(self, i: int) -> dict[str, SymType]:
        """Gets the symbolic expanded actions belonging to the i-th scenario."""
        return {n: self.actions_expanded[_n(n, i)] for n in self.single_actions_exp}

    def parameter(
        self, name: str, shape: tuple[int, int] = (1, 1)
    ) -> tuple[SymType, list[SymType]]:
        """Adds one parameter per scenario to the MSMPC controller.

        Parameters
        ----------
        name : str
            Name of the new parameter. Must not be already in use.
        shape : tuple of 2 ints, optional
            Shape of the new parameter. By default a scalar, i.e., ``(1, 1)``.

        Returns
        -------
        single_parameter : casadi.SX or MX
            Symbol corresponding to the parameter of a single scenario. See the note
            for :meth:``state``
        parameters : list of casadi.SX or MX
            The symbols for the new parameters of each scenario in the MSMPC controller.

        Raises
        ------
        ValueError
            Raises if there is already another parameter with the same name ``name``.
        """
        ps = []
        for i in range(self._n_scenarios):
            name_i = _n(name, i)
            p_i = self.nlp.parameter(name_i, shape)
            ps.append(p_i)
        p_single = self.nlp._sym_type.sym(name, shape)
        self.single_parameters[name] = p_single
        return p_single, ps

    def action(
        self,
        name: str,
        size: int = 1,
        discrete: bool = False,
        lb: Union[npt.ArrayLike, cs.DM] = -npy.inf,
        ub: Union[npt.ArrayLike, cs.DM] = +npy.inf,
    ) -> tuple[SymType, SymType, list[SymType], list[SymType]]:
        """Adds one control action variable per scenario to the MSMPC controller.
        Automatically handles sharing of actions across scenarios, generating less free
        actions if some of these are to be shared across scenarios.

        Parameters
        ----------
        name : str
            Name of the control action.
        size : int, optional
            Size of the control action (assumed to be a vector). Defaults to ``1``.
        discrete : bool, optional
            Flag indicating if the action is discrete. Defaults to ``False``.
        lb : array_like, casadi.DM, optional
            Hard lower bound of the control action, by default ``-np.inf``.
        ub : array_like, casadi.DM, optional
            Hard upper bound of the control action, by default ``+np.inf``.

        Returns
        -------
        single action : casadi.SX or MX
            Symbol corresponding to the action of a single scenario. This is useful for
            automatically defining, e.g., the objective and constraints over the various
            scenarios of the MSMPC controller, but it is not used in the actual NLP
            solver.
        single action expanded : casadi.SX or MX
            Same as above, but the action is expanded to the same length of the
            prediction horizon.
        actions : list of casadi.SX or MX
            The list of the action symbolic variable.
        actions expanded : list of casadi.SX or MX
            The list of the same action symbolic variable, but expanded to the same
            length of the prediction horizon.
        """
        Np = self._prediction_horizon
        Nc = self._control_horizon
        spacing = self._input_spacing
        nu_free = ceil(Nc / spacing)
        sharing = self._input_sharing
        shape = (size, nu_free)

        def _expand(u: SymType) -> SymType:
            u_exp = u if spacing == 1 else repeat(u, (1, spacing))[:, :Nc]
            if gap := Np - u_exp.shape[1]:
                last = u_exp[:, -1]
                u_exp = cs.horzcat(u_exp, *(last for _ in range(gap)))
            return u_exp

        us = []
        us_exp = []
        if sharing == 0:
            # no sharing, each scenario has its own action trajectory
            for i in range(self._n_scenarios):
                name_i = _n(name, i)
                u_i = self.nlp.variable(name_i, shape, discrete, lb, ub)[0]
                u_exp_i = _expand(u_i)
                us.append(u_i)
                us_exp.append(u_exp_i)
                self._actions[name_i] = u_i
                self._actions_exp[name_i] = u_exp_i

        elif sharing == nu_free:
            # full sharing, all scenarios share the same action trajectory
            u_unique = self.nlp.variable(_n(name, "shared"), shape, discrete, lb, ub)[0]
            u_exp_unique = _expand(u_unique)
            for i in range(self._n_scenarios):
                name_i = _n(name, i)
                us.append(u_unique)
                us_exp.append(u_exp_unique)
                self._actions[name_i] = u_unique
                self._actions_exp[name_i] = u_exp_unique

        else:
            # partial sharing, each scenario has its own action trajectory, but
            # the first `sharing` actions are shared
            u_shared = self.nlp.variable(
                _n(name, "shared"), (size, sharing), discrete, lb, ub
            )[0]
            shape_not_shared = (size, nu_free - sharing)
            for i in range(self._n_scenarios):
                name_i = _n(name, i)
                u_i = self.nlp.variable(name_i, shape_not_shared, discrete, lb, ub)[0]
                u_i = cs.horzcat(u_shared, u_i)
                u_exp_i = _expand(u_i)
                us.append(u_i)
                us_exp.append(u_exp_i)
                self._actions[name_i] = u_i
                self._actions_exp[name_i] = u_exp_i

        # create also a single symbol for the action and its expanded version
        u_single = self.nlp._sym_type.sym(name, shape)
        u_exp_single = _expand(u_single)
        self.single_actions[name] = u_single
        self.single_actions_exp[name] = u_exp_single
        return u_single, u_exp_single, us, us_exp

    def set_affine_dynamics(
        self,
        A: MatType,
        B: MatType,
        D: Optional[MatType] = None,
        c: Optional[MatType] = None,
        parallelization: Literal[
            "serial", "unroll", "inline", "thread", "openmp"
        ] = "thread",
        max_num_threads: Optional[int] = None,
    ) -> tuple[
        Optional[MatType], Optional[MatType], Optional[MatType], Optional[MatType]
    ]:
        # NOTE: contrary to `ScenarioBasedMpc`, `D` can be optional here. Also, no need
        # to take the number of scenarios into account for threading, as we will run the
        # dynamics once and then chain-substitute for each scenario.
        return Mpc.set_affine_dynamics(
            self, A, B, D, c, parallelization, max_num_threads
        )

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
        # NOTE: contrary to `ScenarioBasedMpc`, `F` is allowed to have no disturbances.
        # Also, no need to take the number of scenarios into account for threading, as
        # we will run the dynamics once and then chain-substitute for each scenario.
        return Mpc.set_nonlinear_dynamics(
            self, F, parallelization, max_num_threads_or_unrolling_base
        )

    def _set_singleshooting_affine_dynamics(
        self, A: MatType, B: MatType, D: Optional[MatType], c: Optional[MatType]
    ) -> tuple[MatType, MatType, Optional[MatType], Optional[MatType]]:
        # NOTE: conversely to `ScenarioBasedMpc`, here first we create the dynamics with
        # the single states/actions/disturbances (we suppose single parameters have been
        # also used in A,B,D,c), and then chain-substitute to get each scenario's
        # dynamics
        ns = A.shape[0]
        N = self._prediction_horizon
        F, G, H, L = _create_ati_mats(self._prediction_horizon, A, B, D, c)
        x_0 = cs.vcat(self._initial_states.values())
        U = cs.vec(
            cs.vcat(self.single_actions_exp.values())  # NOTE: different from vvcat!
        )
        X_next = F @ x_0 + G @ U
        if H is not None:
            X_next += H @ cs.vec(cs.vcat(self.single_disturbances.values()))
        if L is not None:
            X_next += L
        X = cs.vertcat(x_0, X_next).reshape((ns, N + 1))

        state_names = self.single_states.keys()
        cumsizes = npy.cumsum([0] + [s.shape[0] for s in self._initial_states.values()])
        for i in range(self._n_scenarios):
            X_i = self._chained_substitution_for_scenario_i(X, i, True)
            X_i_split = cs.vertsplit(X_i, cumsizes)
            for n, x in zip(state_names, X_i_split):
                self._states[_n(n, i)] = x
        return F, G, H, L

    def _set_multishooting_nonlinear_dynamics(
        self,
        F: cs.Function,
        n_in: int,
        parallelization: Literal["serial", "unroll", "inline", "thread", "openmp"],
        max_num_threads: int,
    ) -> None:
        # NOTE: see note in `_set_singleshooting_affine_dynamics`
        X = cs.vcat(self.single_states.values())
        U = cs.vcat(self.single_actions_exp.values())
        if n_in < 3:
            args = (X[:, :-1], U)
        else:
            D = cs.vcat(self.single_disturbances.values())
            args = (X[:, :-1], U, D)
        Fmap = F.map(self._prediction_horizon, parallelization, max_num_threads)
        X_next = Fmap(*args)

        constraints = []
        for i in range(self._n_scenarios):
            X_i = _chained_substitute(X[:, 1:], (self.single_states, self.states_i(i)))
            X_next_i = self._chained_substitution_for_scenario_i(X_next, i)
            constraints.append(X_i - X_next_i)
        self.constraint("dyn", cs.vvcat(constraints), "==", 0.0)

    def _set_singleshooting_nonlinear_dynamics(
        self, F: cs.Function, n_in: int, base: int
    ) -> None:
        # NOTE: see note in `_set_singleshooting_affine_dynamics`
        X0 = cs.vcat(self._initial_states.values())
        U = cs.vcat(self.single_actions_exp.values())
        if n_in < 3:
            args = (X0, U)
        else:
            D = cs.vcat(self.single_disturbances.values())
            args = (X0, U, D)

        Fmapaccum = F.mapaccum(
            self._prediction_horizon, {"base": base, "allow_free": True}
        )
        X_next = Fmapaccum(*args)
        X = cs.horzcat(X0, X_next)

        state_names = self.single_states.keys()
        cumsizes = npy.cumsum([0] + [s.shape[0] for s in self._initial_states.values()])
        for i in range(self._n_scenarios):
            X_i = self._chained_substitution_for_scenario_i(X, i, True)
            X_i_split = cs.vertsplit(X_i, cumsizes)
            for n, x in zip(state_names, X_i_split):
                self._states[_n(n, i)] = x

    def _chained_substitution_for_scenario_i(
        self, expr: SymType, i: int, skip_states: bool = False
    ) -> Union[SymType, cs.DM]:
        """Iternal utility to perform substitutions in chain for the i-th scenario."""
        states = ({}, {}) if skip_states else (self.single_states, self.states_i(i))
        return _chained_substitute(
            expr,
            (self.single_parameters, self.parameters_i(i)),
            states,
            (self.single_actions, self.actions_i(i)),
            (self.single_disturbances, self.disturbances_i(i)),
            (self.single_slacks, self.slacks_i(i)),
        )
