from itertools import count
from typing import Any, Dict, Optional, Tuple, Union
import casadi as cs
import numpy as npy
from casadi_mpc.solutions import Solution, subsevalf
from casadi_mpc.debug import MpcDebug


class GenericMpc:
    '''
    The generic MPC class is a controller that solves an optimization problem
    to yield the (possibly, sub-) optimal action, according to the prediction
    model, in the current state.

    This is a generic class in the sense that it does not solve a particular
    problem, but only offers the generic methods to build one (e.g., variables,
    constraints, objective, solver).
    '''

    __ids = count(0)
    __slots__ = [
        'id', 'name', '_CSXX', '_vars', '_pars', '_cons',
        '_f', '_p', '_x', '_lbx', '_ubx', '_lam_lbx', '_lam_ubx',
        '_g', '_lam_g', '_h', '_lam_h',
        '_solver', '_solver_opts', '_failures', '_debug'
    ]

    def __init__(self, sym_type: str = 'SX', name: str = None) -> None:
        '''Creates an MPC controller instance with a given name.

        Parameters
        ----------
        sym_type : 'SX' or 'MX', optional
            The CasADi symbolic variable type to use in the MPC, by default 
            'SX'.
        name : str, optional
            _description_, by default None

        Raises
        ------
        ValueError
            Raises if `sym_type` is neither `'SX'` nor `'MX'`.
        '''

        self.id = next(self.__ids)
        self.name = f'MPC{self.id}' if name is None else name
        if sym_type == 'SX':
            self._CSXX = cs.SX
        elif sym_type == 'MX':
            self._CSXX = cs.MX
        else:
            raise ValueError('Expected symbolic type to be either SX or MX, '
                             f'got {sym_type} instead.')

        self._vars: Dict[str, Union[cs.SX, cs.MX]] = {}
        self._pars: Dict[str, Union[cs.SX, cs.MX]] = {}
        self._cons: Dict[str, Union[cs.SX, cs.MX]] = {}

        self._f: Union[cs.SX, cs.MX] = None  # objective
        self._p = self._CSXX()
        self._x = self._CSXX()
        self._lbx, self._ubx = npy.array([]), npy.array([])
        self._lam_lbx, self._lam_ubx = self._CSXX(), self._CSXX()
        self._g, self._lam_g = self._CSXX(), self._CSXX()
        self._h, self._lam_h = self._CSXX(), self._CSXX()

        self._solver: cs.Function = None
        self._solver_opts: Dict[str, Any] = {}
        self._failures = 0
        self._debug = MpcDebug()

    @property
    def f(self) -> Union[None, cs.SX, cs.MX]:
        '''Gets the objective of the MPC scheme, which is `None` if not set 
        previously set via the `minimize` method.'''
        return self._f

    @property
    def p(self) -> Union[cs.SX, cs.MX]:
        '''Gets the parameters of the MPC scheme.'''
        return self._p

    @property
    def x(self) -> Union[cs.SX, cs.MX]:
        '''Gets the primary variables of the MPC scheme in vector form.'''
        return self._x

    @property
    def lbx(self) -> npy.ndarray:
        '''Gets the lower bound constraints of primary variables of the MPC 
        scheme in vector form.'''
        return self._lbx

    @property
    def ubx(self) -> npy.ndarray:
        '''Gets the upper bound constraints of primary variables of the MPC 
        scheme in vector form.'''
        return self._ubx

    @property
    def lam_lbx(self) -> npy.ndarray:
        '''Gets the dual variables of the primary variables lower bound 
        constraints of the MPC scheme in vector form.'''
        return self._lam_lbx

    @property
    def lam_ubx(self) -> npy.ndarray:
        '''Gets the dual variables of the primary variables upper bound 
        constraints of the MPC scheme in vector form.'''
        return self._lam_ubx

    @property
    def g(self) -> Union[cs.SX, cs.MX]:
        '''Gets the equality constraint expressions of the MPC scheme in vector
        form.'''
        return self._g

    @property
    def h(self) -> Union[cs.SX, cs.MX]:
        '''Gets the inequality constraint expressions of the MPC scheme in 
        vector form.'''
        return self._h

    @property
    def lam_g(self) -> Union[cs.SX, cs.MX]:
        '''Gets the dual variables of the equality constraints of the MPC 
        scheme in vector form.'''
        return self._lam_g

    @property
    def lam_h(self) -> Union[cs.SX, cs.MX]:
        '''Gets the dual variables of the inequality constraints of the MPC 
        scheme in vector form.'''
        return self._lam_h

    @property
    def solver(self) -> Optional[cs.Function]:
        '''Gets the MPC optimization solver. Can be `None`, if the solver is 
        not set with method `set_solver`.'''
        return self._solver

    @property
    def solver_opts(self) -> Dict[str, Any]:
        '''Gets the MPC optimization solver options. The dict is empty, if the 
        solver options are not set with method `set_solver`.'''
        return self._solver_opts

    @property
    def failures(self) -> int:
        '''Gets the cumulative number of failures of the MPC solver.'''
        return self._failures

    @property
    def debug(self) -> MpcDebug:
        '''Gets debug information on the MPC scheme.'''
        return self._debug

    @property
    def nx(self) -> int:
        '''Number of variables in the MPC scheme.'''
        return self._x.shape[0]

    @property
    def np(self) -> int:
        '''Number of parameters in the MPC scheme.'''
        return self._p.shape[0]

    @property
    def ng(self) -> int:
        '''Number of equality constraints in the MPC scheme.'''
        return self._g.shape[0]

    @property
    def nh(self) -> int:
        '''Number of inequality constraints in the MPC scheme.'''
        return self._h.shape[0]

    def parameter(
        self, 
        name: str, 
        shape: Tuple[int, int] = (1, 1)
    ) -> Union[cs.SX, cs.MX]:
        '''Adds a parameter to the MPC scheme.

        Parameters
        ----------
        name : str
            Name of the new parameter. Must not be already in use.
        shape : tuple[int, int]
            Shape of the new parameter.

        Returns
        -------
        casadi.SX or MX
            The symbol for the new parameter.

        Raises
        ------
        ValueError
            Raises if there is already another parameter with the same name.
        '''
        if name in self._pars:
            raise ValueError(f'Parameter name \'{name}\' already exists.')
        par = self._CSXX.sym(name, *shape)
        self._pars[name] = par
        self._p = cs.vertcat(self._p, cs.vec(par))
        return par

    def minimize(self, objective: Union[cs.SX, cs.MX]) -> None:
        '''Sets the objective function to be minimized.'''
        self._f = objective


    def __str__(self) -> str:
        '''Returns the MPC name and a short description.'''
        msg = 'not initialized' if self.solver is None else 'initialized'
        C = len(self._cons)
        return f'{type(self).__name__} {{\n' \
               f'  name: {self.name}\n' \
               f'  #variables: {len(self._vars)} (nx={self.nx})\n' \
               f'  #parameters: {len(self._pars)} (np={self.np})\n' \
               f'  #constraints: {C} (ng={self.ng}, nh={self.nh})\n' \
               f'  CasADi solver {msg}.\n}}'

    def __repr__(self) -> str:
        '''Returns the string representation of the MPC instance.'''
        return f'{type(self).__name__}: {self.name}'
