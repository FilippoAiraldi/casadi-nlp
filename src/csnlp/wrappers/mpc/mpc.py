from collections.abc import Collection, Generator, Sequence
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
    def initial_states(self) -> dict[str, SymType]:
        """Gets the initial states (parameters) of the MPC controller."""
        return self._initial_states

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
        bound_initial: bool = True,
        bound_terminal: bool = True,
    ) -> tuple[Optional[SymType], SymType]:
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
            Same as above, but for the terminal state. By default ``True``.

        Returns
        -------
        state : casadi.SX or MX or None
            The state symbolic variable. If ``shooting=single``, then ``None`` is
            returned since the state will only be available once the dynamics are set.
        initial state : casadi.SX or MX
            The initial state symbolic parameter.

        Raises
        ------
        ValueError
            Raises if there exists already a state with the same name.
        RuntimeError
            Raises in single shooting if lower or upper bounds have been specified,
            since these can only be set after the dynamics have been set via the
            :meth:`constraint` method.
        """
        x0_name = _n(name)
        if self._is_multishooting:
            shape = (size, self._prediction_horizon + 1)
            lb = np.broadcast_to(lb, shape).astype(float)
            ub = np.broadcast_to(ub, shape).astype(float)
            if not bound_initial:
                lb[:, 0] = -np.inf
                ub[:, 0] = +np.inf
            if not bound_terminal:
                lb[:, -1] = -np.inf
                ub[:, -1] = +np.inf

            # create state variable and initial state constraint
            x = self.nlp.variable(name, shape, discrete, lb, ub)[0]
            x0 = self.nlp.parameter(x0_name, (size, 1))
            self.nlp.constraint(x0_name, x[:, 0], "==", x0)
        else:
            if np.any(lb != -np.inf) or np.any(ub != +np.inf):
                raise RuntimeError(
                    "in single shooting, lower and upper state bounds can only be "
                    "created, after the dynamics have been set, via the `constraint` "
                    "method"
                )
            x = None
            x0 = self.nlp.parameter(x0_name, (size, 1))
        self._states[name] = x
        self._initial_states[x0_name] = x0
        return x, x0

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
                if (k == 0 and not kw.get("bound_initial", True)) or (
                    k == Np and not kw.get("bound_terminal", True)
                ):
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

        # create initial states' parameters
        state_names = [kw["name"] for kw in states_kwargs]
        state_sizes = [kw.get("size", 1) for kw in states_kwargs]
        initial_states = {}
        for n, s in zip(state_names, state_sizes):
            name = _n(n)
            initial_states[name] = self.nlp.parameter(name, (s, 1))
        x0 = cs.vcat(initial_states.values())

        # create first states (k=0) and impose initial state constraint
        states, x = _create_states(0)
        self.nlp.constraint("-".join(initial_states.keys()), x, "==", x0)
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
        state_cumsizes = np.cumsum([0] + state_sizes)
        self._states.update(zip(state_names, cs.vertsplit(X, state_cumsizes)))
        self._initial_states.update(initial_states)

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
