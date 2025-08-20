from collections.abc import Collection, Generator, Iterable, Sequence
from inspect import signature
from math import ceil
from typing import (
    Any,
    Callable,
    Literal,
    Optional,
    TypeVar,
    Union,
)

import casadi as cs
import numpy as np
import numpy.typing as npt

from ...core.solutions import Solution
from ...util.math import repeat
from ..wrapper import Nlp, NonRetroactiveWrapper

SymType = TypeVar("SymType", cs.SX, cs.MX)
MatType = TypeVar("MatType", SymType, cs.DM, np.ndarray)


def _n(statename: str) -> str:
    """Internal utility for naming initial states."""
    return f"{statename}_0"


def _callable2csfunc(
    F: Callable[[tuple[npt.ArrayLike, ...]], tuple[npt.ArrayLike, ...]],
    sym_type: type[SymType],
    sizes_in: tuple[int, ...],
) -> cs.Function:
    """Internal utility to convert a callable to a CasADi function."""
    sym_in = [sym_type.sym(f"s{i}", s, 1) for i, s in enumerate(sizes_in)]
    sym_out = F(*sym_in)
    if isinstance(F, Sequence):
        sym_out = sym_out[0]
    return cs.Function("F", sym_in, (sym_out,), {"allow_free": True, "cse": True})


def _create_ati_mats(
    N: int, A: MatType, B: MatType, D: Optional[MatType], c: Optional[MatType]
) -> tuple[MatType, MatType, Optional[MatType], Optional[MatType]]:
    """Internal utility to build the affine time-invariant (ATI) matrices."""
    ns = A.shape[0]

    row = eye = cs.DM.eye(ns)
    rows = [cs.horzcat(row, cs.DM(ns, ns * (N - 1)))]
    for k in range(1, N):
        row = cs.horzcat(A @ row, eye)
        rows.append(cs.horzcat(row, cs.DM(ns, ns * (N - k - 1))))
    base = cs.vcat(rows)

    F = base[:, :ns] @ A
    G = base @ cs.dcat([B] * N)
    H = None if D is None else base @ cs.dcat([D] * N)
    L = None if c is None else base @ cs.repmat(c, N, 1)
    return F, G, H, L


class Mpc(NonRetroactiveWrapper[SymType]):
    """A wrapper to easily turn an NLP scheme into an MPC controller. Most of the theory
    for MPC is taken from :cite:`rawlings_model_2017`.

    Parameters
    ----------
    nlp : Nlp
        NLP scheme to be wrapped
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
        Raises if the shooting method is invalid; or if any of the horizons are invalid.
    """

    def __init__(
        self,
        nlp: Nlp[SymType],
        prediction_horizon: int,
        control_horizon: Optional[int] = None,
        input_spacing: int = 1,
        shooting: Literal["single", "multi"] = "multi",
    ) -> None:
        super().__init__(nlp)

        inttypes = (int, np.integer, np.int_)
        if not isinstance(prediction_horizon, inttypes) or prediction_horizon <= 0:
            raise ValueError(
                "Prediction horizon must be positive and > 0; got "
                f"{prediction_horizon} instead."
            )
        if shooting == "single":
            self._is_multishooting = False
        elif shooting == "multi":
            self._is_multishooting = True
        else:
            raise ValueError("Invalid shooting method.")

        self._prediction_horizon = prediction_horizon
        if control_horizon is None:
            self._control_horizon = self._prediction_horizon
        elif not isinstance(control_horizon, inttypes) or control_horizon <= 0:
            raise ValueError("Control horizon must be positive and > 0.")
        else:
            self._control_horizon = control_horizon

        if not isinstance(input_spacing, inttypes) or input_spacing <= 0:
            raise ValueError("Input spacing factor must be positive and > 0.")
        else:
            self._input_spacing = input_spacing

        self._states: dict[str, SymType] = {}
        if self._is_multishooting:
            self._initial_states_idx: npt.NDArray[np.int64] = np.empty(0, dtype=int)
        else:
            self._initial_states: dict[str, SymType] = {}
        self._actions: dict[str, SymType] = {}
        self._actions_exp: dict[str, SymType] = {}
        self._slacks: dict[str, SymType] = {}
        self._disturbances: dict[str, SymType] = {}
        self._dynamics_already_set = False

    @property
    def prediction_horizon(self) -> int:
        """Gets the prediction horizon of the MPC controller."""
        return self._prediction_horizon

    @property
    def control_horizon(self) -> int:
        """Gets the control horizon of the MPC controller."""
        return self._control_horizon

    @property
    def states(self) -> dict[str, SymType]:
        """Gets the states of the MPC controller."""
        return self._states

    @property
    def first_states(self) -> dict[str, SymType]:
        """Gets the first (along the prediction horizon) states of the controller."""
        return {n: s[:, 0] for n, s in self._states.items()}

    @property
    def first_actions(self) -> dict[str, SymType]:
        """Gets the first (along the prediction horizon) actions of the controller."""
        return {n: a[:, 0] for n, a in self._actions.items()}

    @property
    def ns(self) -> int:
        """Gets the number of states of the MPC controller."""
        if self._is_multishooting:
            return self._initial_states_idx.size
        else:
            return sum(x0.shape[0] for x0 in self._initial_states.values())

    @property
    def actions(self) -> dict[str, SymType]:
        """Gets the control actions of the MPC controller."""
        return self._actions

    @property
    def actions_expanded(self) -> dict[str, SymType]:
        """Gets the expanded control actions of the MPC controller."""
        return self._actions_exp

    @property
    def na(self) -> int:
        """Gets the number of actions of the MPC controller."""
        return sum(a.shape[0] for a in self._actions.values())

    @property
    def slacks(self) -> dict[str, SymType]:
        """Gets the slack variables of the MPC controller."""
        return self._slacks

    @property
    def nslacks(self) -> int:
        """Gets the number of slacks of the MPC controller."""
        return sum(s.shape[0] for s in self._slacks.values())

    @property
    def disturbances(self) -> dict[str, SymType]:
        """Gets the disturbance parameters of the MPC controller."""
        return self._disturbances

    @property
    def nd(self) -> int:
        """Gets the number of disturbances in the MPC controller."""
        return sum(d.shape[0] for d in self._disturbances.values())

    def state(
        self,
        name: str,
        size: int = 1,
        discrete: bool = False,
        lb: Union[npt.ArrayLike, cs.DM] = -np.inf,
        ub: Union[npt.ArrayLike, cs.DM] = +np.inf,
        bound_terminal: bool = True,
    ) -> Optional[SymType]:
        """Adds a state variable to the MPC controller along the whole prediction
        horizon. Automatically creates the constraint on the initial conditions for this
        state.

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
            If ``False``, then the upper and lower bounds on the terminal state are not
            imposed, i.e., set to ``+/- np.inf``. By default ``True``.

        Returns
        -------
        state : casadi.SX or MX or None
            The state symbolic variable. If ``shooting=single``, then ``None`` is
            returned since the state will only be available once the dynamics are set.

        Raises
        ------
        ValueError
            Raises if there exists already a state with the same name.
        RuntimeError
            Raises in single shooting if lower or upper bounds have been specified,
            since these can only be set after the dynamics have been set via the
            :meth:`constraint` method.
        """
        if self._is_multishooting:
            shape = (size, self._prediction_horizon + 1)
            lb = np.broadcast_to(lb, shape).astype(float)
            ub = np.broadcast_to(ub, shape).astype(float)
            lb[:, 0] = ub[:, 0] = 0.0  # always force bounds for initial state
            if not bound_terminal:
                lb[:, -1] = -np.inf
                ub[:, -1] = +np.inf

            nx = self.nlp.nx
            self._initial_states_idx = np.concatenate(
                (self._initial_states_idx, np.arange(nx, nx + size))
            )
            x = self.nlp.variable(name, shape, discrete, lb, ub)[0]
        else:
            if np.any(lb != -np.inf) or np.any(ub != +np.inf):
                raise RuntimeError(
                    "in single shooting, lower and upper state bounds can only be "
                    "created, after the dynamics have been set, via the `constraint` "
                    "method"
                )
            x = None
            x0_name = _n(name)
            self._initial_states[x0_name] = self.nlp.parameter(x0_name, (size, 1))
        self._states[name] = x
        return x

    def action(
        self,
        name: str,
        size: int = 1,
        discrete: bool = False,
        lb: Union[npt.ArrayLike, cs.DM] = -np.inf,
        ub: Union[npt.ArrayLike, cs.DM] = +np.inf,
    ) -> tuple[SymType, SymType]:
        """Adds a control action variable to the MPC controller along the whole control
        horizon. Automatically expands this action to be of the same length of the
        prediction horizon by padding with the final action.

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
        action : casadi.SX or MX
            The control action symbolic variable.
        action_expanded : casadi.SX or MX
            The same control  action variable, but expanded to the same length of the
            prediction horizon.
        """
        Nc = self._control_horizon
        spacing = self._input_spacing
        nu_free = ceil(Nc / spacing)
        u = self.nlp.variable(name, (size, nu_free), discrete, lb, ub)[0]

        u_exp: SymType = u if spacing == 1 else repeat(u, (1, spacing))[:, :Nc]
        if gap := self._prediction_horizon - u_exp.shape[1]:
            u_last = u_exp[:, -1]
            u_exp = cs.horzcat(u_exp, *(u_last for _ in range(gap)))

        self._actions[name] = u
        self._actions_exp[name] = u_exp
        return u, u_exp

    def interleaved_states_and_actions(
        self,
        states_kwargs: Union[dict[str, Any], Collection[dict[str, Any]]],
        actions_kwargs: Union[dict[str, Any], Collection[dict[str, Any]]],
    ) -> Generator[
        tuple[
            dict[str, tuple[SymType, SymType, SymType]],
            dict[str, tuple[SymType, SymType, SymType]],
            dict[str, tuple[SymType, SymType, SymType]],
        ],
        None,
        None,
    ]:
        """Allows to create states and actions in an interleaved manner. In constrast to
        creating all states and then all actions, this method alternates, at each time
        step along the prediction horizon, creating a state and an action.

        Parameters
        ----------
        states_kwargs : kwargs or collection of kwargs
            The keyword arguments for the states to be created. See :meth:`state` for
            more information on the expected keys. Provide a collection of multiple
            dictionaries to create multiple states at each time step.
        actions_kwargs : kwargs or collection of kwargs
            Similar to `states_kwargs`, but for the actions. See :meth:`action`.

        Yields
        ------
        tuple of 3 dicts
            Yields, for each time step, three dictionaries with the current state, the
            current action, and lastly the next state. Each dictionary contains the
            symbolic variable and the multipliers for its lower and upper bounds
            (see :meth:`variable`).

        Raises
        ------
        RuntimeError
            Raises if the MPC is in single shooting mode.
        """
        if not self._is_multishooting:
            raise RuntimeError(
                "Interleaved states and actions are not supported in single-shooting."
            )
        if isinstance(states_kwargs, dict):
            states_kwargs = (states_kwargs,)
        if isinstance(actions_kwargs, dict):
            actions_kwargs = (actions_kwargs,)
        X = []
        U = []
        U_exp = []
        Np = self._prediction_horizon

        # define some helper functions
        def _create_states(k):
            states = {}
            for kw in states_kwargs:
                name = kw["name"]
                if k == 0:
                    lb = ub = 0.0  # always force bounds for initial state
                elif k == Np and not kw.get("bound_terminal", True):
                    lb = -np.inf
                    ub = np.inf
                else:
                    lb = kw.get("lb", -np.inf)
                    ub = kw.get("ub", np.inf)
                shape = (kw.get("size", 1), 1)
                discrete = kw.get("discrete", False)
                states[name] = self.nlp.variable(f"{name}{k}", shape, discrete, lb, ub)
            x = cs.vcat([s[0] for s in states.values()])
            return states, x

        def _create_actions(k):
            actions = {}
            for kw in actions_kwargs:
                name = kw["name"]
                actions[name] = self.nlp.variable(
                    f"{name}{k}",
                    (kw.get("size", 1), 1),
                    kw.get("discrete", False),
                    kw.get("lb", -np.inf),
                    kw.get("ub", np.inf),
                )
            u = cs.vcat([a[0] for a in actions.values()])
            return actions, u

        # create first states (k=0)
        nx = self.nlp.nx
        states, x = _create_states(0)
        self._initial_states_idx = np.concatenate(
            (self._initial_states_idx, np.arange(nx, nx + x.numel()))
        )
        X.append(x)

        for k in range(Np):
            # create actions at time step k
            if k % self._input_spacing == 0 and k < self._control_horizon:
                actions, u = _create_actions(k)
                U.append(u)
            U_exp.append(u)

            # create next states at time step k+1
            new_states, x_new = _create_states(k + 1)

            yield (states, actions, new_states)

            # shift the previous states
            states = new_states
            x = x_new
            X.append(x)

        # save all rolled-out lists of variables to internal dictionaries
        X = cs.hcat(X)
        state_names = (kw["name"] for kw in states_kwargs)
        state_cumsizes = np.cumsum([0] + [kw.get("size", 1) for kw in states_kwargs])
        self._states.update(zip(state_names, cs.vertsplit(X, state_cumsizes)))

        U = cs.hcat(U)
        U_exp = cs.hcat(U_exp)
        action_names = [kw["name"] for kw in actions_kwargs]
        action_cumsizes = np.cumsum([0] + [kw.get("size", 1) for kw in actions_kwargs])
        self._actions.update(zip(action_names, cs.vertsplit(U, action_cumsizes)))
        self._actions_exp.update(
            zip(action_names, cs.vertsplit(U_exp, action_cumsizes))
        )

    def disturbance(self, name: str, size: int = 1) -> SymType:
        """Adds a disturbance parameter to the MPC controller along the whole prediction
        horizon.

        Parameters
        ----------
        name : str
            Name of the disturbance.
        size : int, optional
            Size of the disturbance (assumed to be a vector). Defaults to ``1``.

        Returns
        -------
        casadi.SX or MX
            The symbol for the new disturbance in the MPC controller.
        """
        d = self.nlp.parameter(name, (size, self._prediction_horizon))
        self._disturbances[name] = d
        return d

    def constraint(
        self,
        name: str,
        lhs: Union[SymType, np.ndarray, cs.DM],
        op: Literal["==", ">=", "<="],
        rhs: Union[SymType, np.ndarray, cs.DM],
        soft: bool = False,
        simplify: bool = True,
    ) -> tuple[SymType, ...]:
        """See :meth:`csnlp.Nlp.constraint`."""
        out = self.nlp.constraint(name, lhs, op, rhs, soft, simplify)
        if soft:
            self._slacks[f"slack_{name}"] = out[2]
        return out

    def __call__(
        self,
        initial_conditions: Union[npt.ArrayLike, dict[str, npt.ArrayLike]],
        pars: Union[
            None, dict[str, npt.ArrayLike], Iterable[dict[str, npt.ArrayLike]]
        ] = None,
        vals0: Union[
            None, dict[str, npt.ArrayLike], Iterable[dict[str, npt.ArrayLike]]
        ] = None,
        **kwargs: Any,
    ) -> Union[Solution[SymType], list[Solution[SymType]]]:
        # Similar logic to `Wrapper.__call__` but with an additional argument
        if not self.nlp.is_multi or (
            (pars is None or isinstance(pars, dict))
            and (vals0 is None or isinstance(vals0, dict))
        ):
            return self.solve(initial_conditions, pars, vals0)
        return self.solve_multi(initial_conditions, pars, vals0, **kwargs)

    def solve(
        self,
        initial_conditions: Union[npt.ArrayLike, dict[str, npt.ArrayLike]],
        pars: Optional[dict[str, npt.ArrayLike]] = None,
        vals0: Optional[dict[str, npt.ArrayLike]] = None,
    ) -> Solution[SymType]:
        """Solves the MPC optimization problem.

        Parameters
        ----------
        initial_conditions : array_like or dict (str, array_like)
            Initial conditions for the states of the MPC controller. If a dictionary is
            provided, it must contain the initial conditions for each state variable
            (the keys must match the names of the states). If an array is provided, it
            must be a vector of the same length as the total number of states in the MPC
            controller (i.e., the sum of the sizes of all states).
        pars : dict (str, array_like), optional
            Dictionary or structure containing, for each parameter in the MPC scheme,
            the corresponding numerical value. Can be ``None`` if no parameters are
            present.
        vals0 : dict (str, array_like), optional
            Dictionary or structure containing, for each variable in the MPC scheme, the
            corresponding initial guess. By default, initial guesses are not passed to
            the solver.

        Returns
        -------
        sol : Solution
            A solution object containing all the information on the MPC solution.
        """
        pars = self._prepare_for_solve(initial_conditions, pars)
        return self.nlp.solve(pars, vals0)

    def solve_multi(
        self,
        initial_conditions: Union[npt.ArrayLike, dict[str, npt.ArrayLike]],
        pars: Union[
            None, dict[str, npt.ArrayLike], Iterable[dict[str, npt.ArrayLike]]
        ] = None,
        vals0: Union[
            None, dict[str, npt.ArrayLike], Iterable[dict[str, npt.ArrayLike]]
        ] = None,
        return_all_sols: bool = False,
        **kwargs: Any,
    ) -> Union[Solution[SymType], list[Solution[SymType]]]:
        """Solves the MPC with multiple parameters and guesses.

        Parameters
        ----------
        initial_conditions : array_like or dict (str, array_like)
            Initial conditions for the states of the MPC controller. If a dictionary is
            provided, it must contain the initial conditions for each state variable
            (the keys must match the names of the states). If an array is provided, it
            must be a vector of the same length as the total number of states in the MPC
            controller (i.e., the sum of the sizes of all states). Note that these
            initial conditions are shared among all multistarts.
        pars : dict of (str, array_like) or iterable of, optional
            An iterable that, for each multistart, contains a dictionary with, for each
            parameter in the MPC scheme, the corresponding numerical value. In case a
            single dict is passed, the same is used across all scenarions. Can be
            ``None`` if no parameters are present.
        vals0 : dict of (str, array_like) or iterable of, optional
            An iterable that, for each multistart, contains a dictionary with, for each
            variable in the MPC scheme, the corresponding initial guess. In case a
            single dict is passed, the same is used across all scenarions. By default
            ``None``, in which case  initial guesses are not passed to the solver.
        return_all_sols : bool, optional
            If ``True``, returns the solution of each multistart of the MPC; otherwise,
            only the best solution is returned. By default, ``False``.

        Returns
        -------
        Solution or list of Solutions
            Depending on the flags ``return_all_sols``, returns

            - the best solution out of all multiple starts
            - all the solutions (one per start).
        """
        pars = self._prepare_for_solve(initial_conditions, pars)
        return self.nlp.solve_multi(pars, vals0, return_all_sols, **kwargs)

    def _prepare_for_solve(
        self,
        initial_conditions: Union[npt.ArrayLike, dict[str, npt.ArrayLike]],
        pars: Union[None, dict[str, npt.ArrayLike], Iterable[dict[str, npt.ArrayLike]]],
    ) -> Union[None, dict[str, npt.ArrayLike], Iterable[dict[str, npt.ArrayLike]]]:
        """Internal method to prepare the input data before solving the MPC."""
        if self._is_multishooting:
            # convert initial conditions to an array-like
            if isinstance(initial_conditions, dict):
                x0_vec = np.concatenate([initial_conditions[k] for k in self.states])
            else:
                x0_vec = initial_conditions

            # enforce initial conditions with lbx and ubx
            idx = self._initial_states_idx
            self.nlp.lbx[idx] = self.nlp.ubx[idx] = x0_vec
        else:
            # convert initial conditions to a dictionary of initial conditions
            if isinstance(initial_conditions, dict):
                x0_dict = {k: initial_conditions[k] for k in self._initial_states}
            else:
                mpcstates = self.states
                if len(mpcstates) == 1:
                    states = (initial_conditions,)
                else:
                    cumsizes = np.cumsum([s.shape[0] for s in mpcstates.values()][:-1])
                    states = np.split(np.asarray(initial_conditions), cumsizes)
                x0_dict = dict(zip(mpcstates.keys(), states))

            # pass initial conditions as parameters
            if pars is None:
                pars = x0_dict
            elif isinstance(pars, dict):
                pars.update(x0_dict)
            else:
                pars = (p | x0_dict for p in pars)
        return pars

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
        r"""Sets affine dynamics as the controller's prediction model and creates the
        corresponding dynamics constraints. The dynamics are in the affine form

        .. math:: x_+ = A x + B u + D w + c,

        where :math:`x_+` is the next state, :math:`x` is the current state, :math:`u`
        is the control action, :math:`w` is the disturbance, and :math:`c` is a constant
        term.

        Parameters
        ----------
        A : symbolic or numerical array
            The state matrix :math:`A` in the dynamics equation. Can also be sparse.
        B : symbolic or numerical array
            The action matrix :math:`B` in the dynamics equation. Can also be sparse.
        D : symbolic or numerical array, optional
            The disturbance matrix :math:`D` in the dynamics equation. Must be ``None``
            if no disturbances were provided via the :meth:`disturbance` method. Can
            also be sparse.
        c : symbolic or numerical array, optional
            The constant term :math:`c` in the dynamics equation. By default, ``None``.
            If not provided, the dynamics become linear instead of affine.
        parallelization : "serial", "unroll", "inline", "thread", "openmp"
            The type of parallelization to use (see :func:`casadi.Function.map`) when
            applying the dynamics along the horizon in multiple shooting. By default,
            ``"thread"`` is selected.
        max_num_threads : int, optional
            Maximum number of threads to use in parallelization (if in multiple
            shooting). See :func:`casadi.Function.map` for more information. By default,
            set equal to the prediction horizon.

        Returns
        -------
        Optional 4-tuple of symbolic or numerical arrays
            In multiple shooting, returns a tuple of ``None``s. In single shooting,
            returns the matrices :math:`F, G, H, L` that parametrize the dynamics. See,
            e.g., :cite:`campi_scenario_2019`.

        Raises
        ------
        RuntimeError
            Raises if the dynamics were already set.
        ValueError
            Raises if any of the matrices have the wrong shape; or if D was not provided
            but disturbances were set; or if D was provided but there are no
            disturbances set.
        """
        if self._dynamics_already_set:
            raise RuntimeError("Dynamics were already set.")

        # check dimensions
        ns = self.ns
        if A.shape != (ns, ns):
            raise ValueError(f"A must have shape ({ns}, {ns}); got {A.shape}.")
        na = self.na
        if B.shape != (ns, na):
            raise ValueError(f"B must have shape ({ns}, {na}); got {B.shape}.")
        nd = self.nd
        if D is not None:
            if nd == 0:
                raise ValueError(
                    "Expected D to be `None` as no disturbance was provided via the "
                    "`disturbance` method."
                )
            if D.shape != (ns, nd):
                raise ValueError(f"D must have shape ({ns}, {nd}); got {D.shape}.")
        elif nd > 0:
            raise ValueError("D must be provided since there are disturbances.")
        if c is not None and c.shape != (ns,) and c.shape != (ns, 1):
            raise ValueError(
                f"c must have shape ({ns},) or ({ns}, {1}); got {c.shape}."
            )

        if max_num_threads is None:
            max_num_threads = self._prediction_horizon

        if self._is_multishooting:
            # not much optimization that we can do here
            if c is None:
                c = 0
            if D is None:
                sizes_in = (ns, na)
                F = lambda x, u: A @ x + B @ u + c
            else:
                sizes_in = (ns, na, nd)
                F = lambda x, u, d: A @ x + B @ u + D @ d + c
            dynamics = _callable2csfunc(F, self.nlp.sym_type, sizes_in)
            self._set_multishooting_nonlinear_dynamics(
                dynamics, len(sizes_in), parallelization, max_num_threads
            )
            F = G = H = L = None
        else:
            F, G, H, L = self._set_singleshooting_affine_dynamics(A, B, D, c)

        self._dynamics_already_set = True
        return F, G, H, L

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
        """Sets the nonlinear dynamics of the controller's prediction model and creates
        the corresponding dynamics constraints.

        Parameters
        ----------
        F : casadi.Function or callable
            A CasADi function of the form :math:`x_+ = F(x,u)` or :math:`x+ = F(x,u,d)`,
            where :math:`x,u,d` are the state, action, and disturbance respectively,
            :math:`F` is a generic nonlinear function and :math:`x_+` is the next state.
        parallelization : "serial", "unroll", "inline", "thread", "openmp"
            The type of parallelization to use (see :func:`casadi.Function.map`) when
            applying the dynamics along the horizon in multiple shooting. By default,
            ``"thread"`` is selected.
        max_num_threads_or_unrolling_base : int, optional
            Maximum number of threads to use in parallelization (if in multiple
            shooting), or the base for unrolling (if in single shooting). See
            :func:`casadi.Function.map` and :func:`casadi.Function.mapaccum` for more
            information, respectively. By default, set equal to the prediction horizon.

        Raises
        ------
        ValueError
            Raises if the dynamics do not accept 2 or 3 input arguments.
        RuntimeError
            Raises if the dynamics have been already set; or if the function ``F`` does
            not accept the expected input sizes.
        """
        if self._dynamics_already_set:
            raise RuntimeError("Dynamics were already set.")
        n_in = F.n_in() if isinstance(F, cs.Function) else len(signature(F).parameters)
        if n_in < 2 or n_in > 3:
            raise ValueError(
                "The dynamics function must accepted 2 or 3 arguments; got "
                f"{n_in} inputs."
            )

        if not isinstance(F, cs.Function):
            sizes_in = (self.ns, self.na) if n_in < 3 else (self.ns, self.na, self.nd)
            F = _callable2csfunc(F, self.nlp.sym_type, sizes_in)

        if max_num_threads_or_unrolling_base is None:
            max_num_threads_or_unrolling_base = self._prediction_horizon

        if self._is_multishooting:
            self._set_multishooting_nonlinear_dynamics(
                F, n_in, parallelization, max_num_threads_or_unrolling_base
            )
        else:
            self._set_singleshooting_nonlinear_dynamics(
                F, n_in, max_num_threads_or_unrolling_base
            )
        self._dynamics_already_set = True

    def _set_singleshooting_affine_dynamics(
        self, A: MatType, B: MatType, D: Optional[MatType], c: Optional[MatType]
    ) -> tuple[MatType, MatType, Optional[MatType], Optional[MatType]]:
        """Internal utility to create affine dynamics constraints and states in
        single shooting mode."""
        ns = A.shape[0]
        N = self._prediction_horizon
        F, G, H, L = _create_ati_mats(self._prediction_horizon, A, B, D, c)
        x_0 = cs.vcat(self._initial_states.values())
        U = cs.vec(cs.vcat(self._actions_exp.values()))  # NOTE: different from vvcat!
        X_next = F @ x_0 + G @ U
        if H is not None:
            X_next += H @ cs.vec(cs.vcat(self._disturbances.values()))
        if L is not None:
            X_next += L

        # append initial state, reshape and save to internal dict
        X = cs.vertcat(x_0, X_next).reshape((ns, N + 1))
        cumsizes = np.cumsum([0] + [s.shape[0] for s in self._initial_states.values()])
        self._states = dict(zip(self._states.keys(), cs.vertsplit(X, cumsizes)))
        return F, G, H, L

    def _set_multishooting_nonlinear_dynamics(
        self,
        F: cs.Function,
        n_in: int,
        parallelization: Literal["serial", "unroll", "inline", "thread", "openmp"],
        max_num_threads: int,
    ) -> None:
        """Internal utility to create nonlinear dynamics constraints in multiple
        shooting."""
        X = cs.vcat(self._states.values())
        U = cs.vcat(self._actions_exp.values())
        if n_in < 3:
            args = (X[:, :-1], U)
        else:
            D = cs.vcat(self._disturbances.values())
            args = (X[:, :-1], U, D)

        Fmap = F.map(self._prediction_horizon, parallelization, max_num_threads)
        X_next = Fmap(*args)
        self.constraint("dyn", X[:, 1:], "==", X_next)

    def _set_singleshooting_nonlinear_dynamics(
        self, F: cs.Function, n_in: int, base: int
    ) -> None:
        """Internal utility to create nonlinear dynamics constraints and states in
        single shooting."""
        X0 = cs.vcat(self._initial_states.values())
        U = cs.vcat(self._actions_exp.values())
        if n_in < 3:
            args = (X0, U)
        else:
            D = cs.vcat(self._disturbances.values())
            args = (X0, U, D)

        Fmapaccum = F.mapaccum(
            self._prediction_horizon, {"base": base, "allow_free": True}
        )
        X_next = Fmapaccum(*args)
        X = cs.horzcat(X0, X_next)
        cumsizes = np.cumsum([0] + [s.shape[0] for s in self._initial_states.values()])
        self._states = dict(zip(self._states.keys(), cs.vertsplit(X, cumsizes)))
