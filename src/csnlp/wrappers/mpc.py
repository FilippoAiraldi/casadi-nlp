from typing import Dict, Literal, Optional, Tuple, TypeVar, Union

import casadi as cs
import numpy as np
import numpy.typing as npt

from csnlp.wrappers.wrapper import Nlp, NonRetroactiveWrapper

T = TypeVar("T", cs.SX, cs.MX)


def _n(statename: str) -> str:
    """Internal utility for naming initial states."""
    return f"{statename}_0"


class Mpc(NonRetroactiveWrapper[T]):
    """A wrapper to easily turn the NLP scheme into an MPC controller. Most of the
    theory for MPC is taken from [1].

    References
    ----------
    [1] Rawlings, J.B., Mayne, D.Q. and Diehl, M., 2017. Model Predictive Control:
        theory, computation, and design (Vol. 2). Madison, WI: Nob Hill Publishing.
    """

    def __init__(
        self,
        nlp: Nlp[T],
        prediction_horizon: int,
        control_horizon: Optional[int] = None,
        shooting: Literal["single", "multi"] = "multi",
    ) -> None:
        """Initializes the MPC wrapper around the NLP instance.

        Parameters
        ----------
        nlp : Nlp
            NLP scheme to be wrapped
        prediction_horizon : int
            A positive integer for the prediction horizon of the MPC controller.
        control_horizon : int, optional
            A positive integer for the control horizon of the MPC controller. If not
            given, it is set equal to the control horizon.
        shooting : 'single' or 'multi', optional
            Type of approach in the direct shooting for parametrizing the control
            trajectory. See [1, Section 8.5]. By default, direct shooting is used.

        Raises
        ------
        ValueError
            Raises if the shooting method is invalid; or if any of the horizons are
            invalid.

        References
        ----------
        [1] Rawlings, J.B., Mayne, D.Q. and Diehl, M., 2017. Model Predictive Control:
            theory, computation, and design (Vol. 2). Madison, WI: Nob Hill Publishing.
        """
        super().__init__(nlp)

        if prediction_horizon <= 0:
            raise ValueError("Prediction horizon must be positive and > 0.")
        if shooting == "single":
            self._is_multishooting = False
        elif shooting == "multi":
            self._is_multishooting = True
        else:
            raise ValueError("Invalid shooting method.")

        self._prediction_horizon = prediction_horizon
        if control_horizon is None:
            self._control_horizon = self._prediction_horizon
        elif control_horizon <= 0:
            raise ValueError("Control horizon must be positive and > 0.")
        else:
            self._control_horizon = control_horizon

        self._states: Dict[str, T] = {}
        self._initial_states: Dict[str, T] = {}
        self._actions: Dict[str, T] = {}
        self._actions_exp: Dict[str, T] = {}
        self._slacks: Dict[str, T] = {}
        self._disturbances: Dict[str, T] = {}
        self._dynamics: cs.Function = None

    @property
    def prediction_horizon(self) -> int:
        """Gets the prediction horizon of the MPC controller."""
        return self._prediction_horizon

    @property
    def control_horizon(self) -> int:
        """Gets the control horizon of the MPC controller."""
        return self._control_horizon

    @property
    def states(self) -> Dict[str, T]:
        """Gets the states of the MPC controller."""
        return self._states

    @property
    def initial_states(self) -> Dict[str, T]:
        """Gets the initial states (parameters) of the MPC controller."""
        return self._initial_states

    @property
    def ns(self) -> int:
        """Gets the number of states of the MPC controller."""
        return sum(x0.shape[0] for x0 in self._initial_states.values())

    @property
    def actions(self) -> Dict[str, T]:
        """Gets the control actions of the MPC controller."""
        return self._actions

    @property
    def actions_expanded(self) -> Dict[str, T]:
        """Gets the expanded control actions of the MPC controller."""
        return self._actions_exp

    @property
    def na(self) -> int:
        """Gets the number of actions of the MPC controller."""
        return sum(a.shape[0] for a in self._actions.values())

    @property
    def slacks(self) -> Dict[str, T]:
        """Gets the slack variables of the MPC controller."""
        return self._slacks

    @property
    def nslacks(self) -> int:
        """Gets the number of slacks of the MPC controller."""
        return sum(s.shape[0] for s in self._slacks.values())

    @property
    def disturbances(self) -> Dict[str, T]:
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
        lb: Union[npt.ArrayLike, cs.DM] = -np.inf,
        ub: Union[npt.ArrayLike, cs.DM] = +np.inf,
    ) -> Tuple[Optional[T], T]:
        """Adds a state variable to the MPC controller along the whole prediction
        horizon. Automatically creates the constraint on the initial conditions for this
        state.

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

        Returns
        -------
        state : casadi.SX or MX or None
            The state symbolic variable. If `shooting=single`, then `None` is returned
            since the state will only be available once the dynamics are set.
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
        x0_name = _n(name)
        if self._is_multishooting:
            x = self.nlp.variable(name, (size, self._prediction_horizon + 1), lb, ub)[0]
            x0 = self.nlp.parameter(x0_name, (size, 1))
            self.nlp.constraint(x0_name, x[:, 0], "==", x0)
        else:
            if np.any(lb != -np.inf) or np.any(ub != +np.inf):
                raise RuntimeError(
                    "In single shooting, lower and upper state bounds can only"
                    " be created after the dynamics have been set"
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
        lb: Union[npt.ArrayLike, cs.DM] = -np.inf,
        ub: Union[npt.ArrayLike, cs.DM] = +np.inf,
    ) -> Tuple[T, T]:
        """Adds a control action variable to the MPC controller along the whole control
        horizon. Automatically expands this action to be of the same length of the
        prediction horizon by padding with the final action.

        Parameters
        ----------
        name : str
            Name of the control action.
        size : int, optional
            Size of the control action (assumed to be a vector). Defaults to 1.
        lb : array_like, casadi.DM, optional
            Hard lower bound of the control action, by default -np.inf.
        ub : array_like, casadi.DM, optional
            Hard upper bound of the control action, by default +np.inf.

        Returns
        -------
        action : casadi.SX or MX
            The control action symbolic variable.
        action_expanded : casadi.SX or MX
            The same control  action variable, but expanded to the same length of the
            prediction horizon.
        """
        u = self.nlp.variable(name, (size, self._control_horizon), lb, ub)[0]
        gap = self._prediction_horizon - self._control_horizon
        u_exp = cs.horzcat(u, *(u[:, -1] for _ in range(gap)))
        self._actions[name] = u
        self._actions_exp[name] = u_exp
        return u, u_exp

    def disturbance(self, name: str, size: int = 1) -> T:
        """Adds a disturbance parameter to the MPC controller along the whole prediction
        horizon.

        Parameters
        ----------
        name : str
            Name of the disturbance.
        size : int, optional
            Size of the disturbance (assumed to be a vector). Defaults to 1.

        Returns
        -------
        casadi.SX or MX
            The symbol for the new disturbance in the MPC controller.
        """
        d = self.nlp.parameter(name=name, shape=(size, self._prediction_horizon))
        self._disturbances[name] = d
        return d

    def constraint(
        self,
        name: str,
        lhs: Union[T, np.ndarray, cs.DM],
        op: Literal["==", ">=", "<="],
        rhs: Union[T, np.ndarray, cs.DM],
        soft: bool = False,
        simplify: bool = True,
    ) -> Tuple[T, ...]:
        """See `Nlp.constraint` method."""
        out = self.nlp.constraint(
            name=name, lhs=lhs, op=op, rhs=rhs, soft=soft, simplify=simplify
        )
        if soft:
            self._slacks[f"slack_{name}"] = out[2]
        return out

    @property
    def dynamics(self) -> Optional[cs.Function]:
        """Dynamics of the controller's prediction model, i.e., a CasADi function of the
        form `x+ = F(x,u)` or `x+ = F(x,u,d)`, where `x,u,d` are the state, action,
        disturbances respectively, and `x+` is the next state. The function can have
        multiple outputs, in which case `x+` is assumed to be the first one.

        Raises
        ------
        ValueError
            When setting, raises if the dynamics do not accept 2 or 3 input arguments.
        RuntimeError
            When setting, raises if the dynamics have been already set; or if the
            function `F` does not take accept the expected input sizes.
        """
        return self._dynamics

    @dynamics.setter
    def dynamics(self, F: cs.Function) -> None:
        if self._dynamics is not None:
            raise RuntimeError("Dynamics were already set.")
        n_in = F.n_in()
        n_out = F.n_out()
        if n_in < 2 or n_in > 3 or n_out < 1:
            raise ValueError(
                "The dynamics function must accepted 2 or 3 arguments and return at "
                f"at least 1 output; got {n_in} inputs and {n_out} outputs instead."
            )
        if self._is_multishooting:
            self._multishooting_dynamics(F, n_in, n_out)
        else:
            self._singleshooting_dynamics(F, n_in, n_out)
        self._dynamics = F

    def _multishooting_dynamics(self, F: cs.Function, n_in: int, n_out: int) -> None:
        """Internal utility to create dynamics constraints in multiple shooting."""
        X = cs.vertcat(*self._states.values())
        U = cs.vertcat(*self._actions_exp.values())
        if n_in < 3:
            args_at = lambda k: (X[:, k], U[:, k])  # type: ignore # noqa: E731
        else:
            D = cs.vertcat(*self._disturbances.values())
            args_at = lambda k: (X[:, k], U[:, k], D[:, k])  # type: ignore # noqa: E731
        for k in range(self._prediction_horizon):
            x_next = F(*args_at(k))
            if n_out != 1:
                x_next = x_next[0]
            self.constraint(f"dyn_{k}", X[:, k + 1], "==", x_next)

    def _singleshooting_dynamics(self, F: cs.Function, n_in: int, n_out: int) -> None:
        """Internal utility to create dynamics constraints and states in single
        shooting."""
        Xk = cs.vertcat(*self._initial_states.values())
        U = cs.vertcat(*self._actions_exp.values())
        if n_in < 3:
            args_at = lambda k: (U[:, k],)  # type: ignore  # noqa: E731
        else:
            D = cs.vertcat(*self._disturbances.values())
            args_at = lambda k: (U[:, k], D[:, k])  # type: ignore  # noqa: E731
        X = [Xk]
        for k in range(self._prediction_horizon):
            Xk = F(Xk, *args_at(k))
            if n_out != 1:
                Xk = Xk[0]
            X.append(Xk)
        X = cs.horzcat(*X)
        cumsizes = np.cumsum([0] + [s.shape[0] for s in self._initial_states.values()])
        self._states = dict(zip(self._states.keys(), cs.vertsplit(X, cumsizes)))
