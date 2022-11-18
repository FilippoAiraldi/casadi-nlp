from typing import List, Literal, Tuple, Union, Dict, Optional
import casadi as cs
import numpy as np
from csnlp.wrappers.wrapper import Wrapper, NlpType
from csnlp.util import (
    cached_property, cache_clearer,
    struct_symSX, dict2struct
)


class Mpc(Wrapper[NlpType]):
    '''
    A wrapper to easily turn the NLP scheme into an MPC controller. Most of the
    theory for MPC is taken from [1].

    References
    ----------
    [1] Rawlings, J.B., Mayne, D.Q. and Diehl, M., 2017. Model Predictive
        Control: theory, computation, and design (Vol. 2). Madison, WI: Nob
        Hill Publishing.
    '''

    def __init__(
        self,
        nlp: NlpType,
        prediction_horizon: int,
        control_horizon: Optional[int] = None,
        shooting: Literal['single', 'multi'] = 'multi'
    ) -> None:
        '''Initializes the MPC wrapper around the NLP instance.

        Parameters
        ----------
        nlp : NlpType
            NLP scheme to be wrapped
        prediction_horizon : int
            A positive integer for the prediction horizon of the MPC
            controller.
        control_horizon : int, optional
            A positive integer for the control horizon of the MPC controller.
            If not given, it is set equal to the control horizon.
        shooting : 'single' or 'multi', optional
            Type of approach in the direct shooting for parametrizing the
            control trajectory. See [1, Section 8.5]. By default, direct
            multiple shooting is used.

        Raises
        ------
        ValueError
            Raises if the shooting method is invalid; or if any of the horizons
            are invalid.

        References
        ----------
        [1] Rawlings, J.B., Mayne, D.Q. and Diehl, M., 2017. Model Predictive
            Control: theory, computation, and design (Vol. 2). Madison, WI: Nob
            Hill Publishing.
        '''
        super().__init__(nlp)
        if shooting not in {'single', 'multi'}:
            raise ValueError('Invalid shooting method.')
        if prediction_horizon <= 0:
            raise ValueError('Prediction horizon must be positive and > 0.')
        self._shooting = shooting
        self._prediction_horizon = prediction_horizon
        if control_horizon is None:
            self._control_horizon = self._prediction_horizon
        elif control_horizon <= 0:
            raise ValueError('Control horizon must be positive and > 0.')
        else:
            self._control_horizon = control_horizon
        self._state_names: List[str] = []
        self._action_names: List[str] = []
        self._slack_names: List[str] = []
        self._disturbance_names: List[str] = []
        self._actions_exp: Dict[str, Union[cs.SX, cs.MX]] = {}
        self._dynamics: cs.Function = None

    @property
    def prediction_horizon(self) -> int:
        '''Gets the prediction horizon of the MPC controller.'''
        return self._prediction_horizon

    @property
    def control_horizon(self) -> int:
        '''Gets the control horizon of the MPC controller.'''
        return self._control_horizon

    @cached_property
    def states(self) -> Union[struct_symSX, Dict[str, cs.MX]]:
        '''Gets the states of the MPC controller.'''
    @cached_property
    def initial_states(self) -> Union[struct_symSX, Dict[str, cs.MX]]:
        '''Gets the initial states (parameters) of the MPC controller.'''
        return dict2struct(
            {n: self.nlp._pars[f'{n}_0'] for n in self._state_names})

    @cached_property
    def actions(self) -> Union[struct_symSX, Dict[str, cs.MX]]:
        '''Gets the control actions of the MPC controller.'''
        return dict2struct({n: self.nlp._vars[n] for n in self._action_names})

    @cached_property
    def actions_expanded(self) -> Union[struct_symSX, Dict[str, cs.MX]]:
        '''Gets the expanded control actions of the MPC controller.'''
        return dict2struct(self._actions_exp)

    @cached_property
    def slacks(self) -> Union[struct_symSX, Dict[str, cs.MX]]:
        '''Gets the slack variables of the MPC controller.'''
        return dict2struct({n: self.nlp._vars[n] for n in self._slack_names})

    @cached_property
    def disturbances(self) -> Union[struct_symSX, Dict[str, cs.MX]]:
        '''Gets the disturbance parameters of the MPC controller.'''
        return dict2struct(
            {n: self.nlp._pars[n] for n in self._disturbance_names})

    @cache_clearer(states, initial_states)
    def state(
        self,
        name: str,
        dim: int = 1,
        lb: Union[np.ndarray, cs.DM] = -np.inf,
        ub: Union[np.ndarray, cs.DM] = +np.inf
    ) -> Union[Tuple[cs.SX, cs.SX], Tuple[cs.MX, cs.MX]]:
        '''Adds a state variable to the MPC controller along the whole
        prediction horizon. Automatically creates the constraint on the initial
        conditions for this state.

        Parameters
        ----------
        name : str
            Name of the state.
        dim : int
            Dimension of the state (assumed to be a vector).
        lb : array_like, casadi.DM, optional
            Hard lower bound of the state, by default -np.inf.
        ub : array_like, casadi.DM, optional
            Hard upper bound of the state, by default +np.inf.

        Returns
        -------
        state : casadi.SX, MX or None
            The state symbolic variable. If `shooting=single`, then `None` is
            returned since the state will only be available once the dynamics
            are set.
        initial state : casadi.SX, MX
            The initial state symbolic parameter.
        '''
        x = self.nlp.variable(
            name, (dim, self._prediction_horizon + 1), lb, ub)[0]
        x0 = self.nlp.parameter(f'{name}_0', (dim, 1))
        self.nlp.constraint(f'{name}_0', x[:, 0], '==', x0)
        self._state_names.append(name)
        return x, x0

    @cache_clearer(actions, actions_expanded)
    def action(
        self,
        name: str,
        dim: int = 1,
        lb: Union[np.ndarray, cs.DM] = -np.inf,
        ub: Union[np.ndarray, cs.DM] = +np.inf
    ) -> Union[cs.SX, cs.MX]:
        '''Adds a control action variable to the MPC controller along the whole
        control horizon. Automatically expands this action to be of the same
        length of the prediction horizon by padding with the final action.

        Parameters
        ----------
        name : str
            Name of the control action.
        dim : int, optional
            Dimension of the control action (assumed to be a vector). Defaults
            to 1.
        lb : Union[np.ndarray, cs.DM], optional
            Hard lower bound of the control action, by default -np.inf.
        ub : Union[np.ndarray, cs.DM], optional
            Hard upper bound of the control action, by default +np.inf.

        Returns
        -------
        action : SX or MX
            The control action symbolic variable.
        action_expanded : SX or MX
            The same control  action variable, but expanded to the same length
            of the prediction horizon.
        '''
        u = self.nlp.variable(name, (dim, self._control_horizon), lb, ub)[0]
        gap = self._prediction_horizon - self._control_horizon
        u_exp = cs.horzcat(u, *(u[:, -1] for _ in range(gap)))
        self._actions_exp[name] = u_exp
        self._action_names.append(name)
        return u, u_exp

    @cache_clearer(disturbances)
    def disturbance(self, name: str, dim: int = 1) -> Union[cs.SX, cs.MX]:
        '''Adds a disturbance parameter to the MPC controller along the whole
        prediction horizon.

        Parameters
        ----------
        name : str
            Name of the disturbance.
        dim : int, optional
            Dimension of the disturbance (assumed to be a vector). Defaults to
            1.

        Returns
        -------
        casadi.SX or MX
            The symbol for the new disturbance in the MPC controller.
        '''
        out = self.nlp.parameter(
            name=name, shape=(dim, self._prediction_horizon))
        self._disturbance_names.append(name)
        return out

    @cache_clearer(slacks)
    def constraint(
        self,
        name: str,
        lhs: Union[np.ndarray, cs.DM, cs.SX, cs.MX],
        op: Literal['==', '>=', '<='],
        rhs: Union[np.ndarray, cs.DM, cs.SX, cs.MX],
        soft: bool = False,
        simplify: bool = True
    ) -> Union[
        Tuple[cs.SX, cs.SX],
        Tuple[cs.MX, cs.MX],
        Tuple[cs.SX, cs.SX, cs.SX],
        Tuple[cs.MX, cs.MX, cs.MX],
    ]:
        '''See `Nlp.constraint` method.'''
        out = self.nlp.constraint(
            name=name, lhs=lhs, op=op, rhs=rhs, soft=soft, simplify=simplify)
        if soft:
            self._slack_names.append(f'slack_{name}')
        return out

    @property
    def dynamics(self) -> Optional[cs.Function]:
        '''Gets the prediction model's dynamics of the MPC controller. If not
        set, returns `None`.'''
        return self._dynamics

    @dynamics.setter
    def dynamics(self, F: cs.Function) -> None:
        if self._dynamics is not None:
            raise RuntimeError('Dynamics were already set.')
        n_in = F.n_in()
        n_out = F.n_out()
        if n_in < 2 or n_in > 3 or n_out < 1:
            raise ValueError(
                'The dynamics function must accepted 2 or 3 arguments and '
                f'return at least 1 output; got {n_in} inputs and {n_out} '
                'outputs instead.')

        X = cs.vertcat(*(self.nlp._vars[n] for n in self._state_names))
        U = cs.vertcat(*(self._actions_exp[n] for n in self._action_names))
        if n_in < 3:
            get_args = lambda k: (X[:, k], U[:, k])
        if n_in > 2:
            D = cs.vertcat(
                *(self.nlp._pars[n] for n in self._disturbance_names))
            get_args = lambda k: (X[:, k], U[:, k], D[:, k])
        for k in range(self._prediction_horizon):
            x_next = F(*get_args(k))
            if n_out != 1:
                x_next = x_next[0]
            self.constraint(f'dyn_{k}', X[:, k + 1], '==', x_next)
        self._dynamics = F
