from itertools import count
from typing import Any, Dict, Literal, Optional, Tuple, Type, Union
import casadi as cs
import numpy as npy
from casadi_mpc.solutions import Solution, subsevalf
from casadi_mpc.debug import MpcDebug
from casadi_mpc.util import \
    cached_property, cached_property_reset, struct_symSX, dict2struct


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

    def __init__(
        self,
        sym_type: Literal['SX', 'MX'] = 'SX',
        name: str = None
    ) -> None:
        '''Creates an MPC controller instance with a given name.

        Parameters
        ----------
        sym_type : 'SX' or 'MX', optional
            The CasADi symbolic variable type to use in the MPC, by default 
            'SX'.
        name : str, optional
            Name of the MPC scheme. If `None`, it is automatically assigned.
        '''

        self.id = next(self.__ids)
        self.name = f'MPC{self.id}' if name is None else name
        self._csXX: Union[Type[cs.SX], Type[cs.MX]] = getattr(cs, sym_type)

        self._vars: Dict[str, Union[cs.SX, cs.MX]] = {}
        self._pars: Dict[str, Union[cs.SX, cs.MX]] = {}
        self._cons: Dict[str, Union[cs.SX, cs.MX]] = {}

        self._f: Union[cs.SX, cs.MX] = None  # objective
        self._p = self._csXX()
        self._x = self._csXX()
        self._lbx, self._ubx = npy.array([]), npy.array([])
        self._lam_lbx, self._lam_ubx = self._csXX(), self._csXX()
        self._g, self._lam_g = self._csXX(), self._csXX()
        self._h, self._lam_h = self._csXX(), self._csXX()
        self._lbg, self._lbh = npy.array([]), npy.array([])

        self._solver: cs.Function = None
        self._solver_opts: Dict[str, Any] = {}
        self._failures = 0
        self._debug = MpcDebug()

    @property
    def sym_type(self) -> Union[Type[cs.SX], Type[cs.MX]]:
        '''Gets the CasADi symbolic type used in this MPC scheme.'''
        return self._csXX

    @property
    def f(self) -> Union[None, cs.SX, cs.MX]:
        '''Gets the objective of the MPC scheme, which is `None` if not set 
        previously set via the `minimize` method.'''
        return self._f

    @f.setter
    def f(self, objective: Union[cs.SX, cs.MX]) -> None:
        '''Sets the objective of the MPC scheme.'''
        return self.minimize(objective)

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
        not set with method `init_solver`.'''
        return self._solver

    @property
    def solver_opts(self) -> Dict[str, Any]:
        '''Gets the MPC optimization solver options. The dict is empty, if the 
        solver options are not set with method `init_solver`.'''
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

    @cached_property
    def parameters(self) -> Union[struct_symSX, Dict[str, cs.MX]]:
        '''Gets the parameters of the MPC scheme.'''
        return dict2struct(self._pars)

    @cached_property
    def variables(self) -> Union[struct_symSX, Dict[str, cs.MX]]:
        '''Gets the variables of the MPC scheme.'''
        return dict2struct(self._vars)

    @cached_property
    def constraints(self) -> Union[struct_symSX, Dict[str, cs.MX]]:
        '''Gets the constraints of the MPC scheme.'''
        return dict2struct(self._cons)

    @cached_property_reset(parameters)
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
        shape : tuple[int, int], optional
            Shape of the new parameter. By default, a scalar.

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
        par = self._csXX.sym(name, *shape)
        self._pars[name] = par
        self._p = cs.vertcat(self._p, cs.vec(par))
        self._debug.register('p', name, shape)
        return par

    @cached_property_reset(variables)
    def variable(
        self, name: str,
        shape: Tuple[int, int] = (1, 1),
        lb: Union[npy.ndarray, cs.DM] = -npy.inf,
        ub: Union[npy.ndarray, cs.DM] = +npy.inf
    ) -> Union[Tuple[cs.SX, cs.SX, cs.SX], Tuple[cs.MX, cs.MX, cs.MX]]:
        '''
        Adds a variable to the MPC problem.

        Parameters
        ----------
        name : str
            Name of the new variable. Must not be already in use.
        shape : tuple[int, int], optional
            Shape of the new variable. By default, a scalar.
        lb, ub: array_like, optional
            Lower and upper bounds of the new variable. By default, unbounded.
            If provided, their dimension must be broadcastable.

        Returns
        -------
        var : casadi.SX
            The symbol of the new variable.
        lam_lb : casadi.SX
            The symbol corresponding to the new variable lower bound
            constraint's multipliers.
        lam_ub : casadi.SX
            Same as above, for upper bound.

        Raises
        ------
        ValueError
            Raises if there is already another variable with the same name; or 
            if any element of the lower bound is larger than the corresponding 
            lower bound element.
        '''
        if name in self._vars:
            raise ValueError(f'Variable name \'{name}\' already exists.')
        lb, ub = npy.broadcast_to(lb, shape), npy.broadcast_to(ub, shape)
        if npy.all(lb > ub):
            raise ValueError('Improper variable bounds.')

        var = self._csXX.sym(name, *shape)
        self._vars[name] = var
        self._x = cs.vertcat(self._x, cs.vec(var))
        self._lbx = npy.concatenate((self._lbx, lb.flatten('F')))
        self._ubx = npy.concatenate((self._ubx, ub.flatten('F')))
        self._debug.register('x', name, shape)

        lam_lb = self._csXX.sym(f'lam_lb_{name}', *shape)
        self._lam_lbx = cs.vertcat(self._lam_lbx, cs.vec(lam_lb))
        lam_ub = self._csXX.sym(f'lam_ub_{name}', *shape)
        self._lam_ubx = cs.vertcat(self._lam_ubx, cs.vec(lam_ub))
        return var, lam_lb, lam_ub

    @cached_property_reset(constraints)
    def constraint(
        self,
        name: str,
        lhs: Union[cs.SX, cs.MX],
        op: Literal['==', '>=', '<='],
        rhs: Union[cs.SX, cs.MX],
        simplify: bool = True
    ) -> Union[Tuple[cs.SX, cs.SX], Tuple[cs.MX, cs.MX]]:
        '''Adds a constraint to the MPC problem, e.g., `lhs <= rhs`.

        Parameters
        ----------
        name : str
            Name of the new constraint. Must not be already in use.
        lhs : casadi.SX, MX
            Symbolic expression of the left-hand term of the constraint.
        op: str, {'==' '>=', '<='}
            Operator relating the two terms.
        rhs : casadi.SX, MX
            Symbolic expression of the right-hand term of the constraint.
        simplify : bool, optional
            Optionally simplies the constraint expression, but can be disabled.

        Returns
        -------
        expr : casadi.SX, MX
            The constraint expression in canonical form, i.e., `g(x,u) = 0` or
            `h(x,u) <= 0`.
        lam : casadi.SX, MX
            The symbol corresponding to the constraint's multipliers.

        Raises
        ------
        ValueError
            Raises if there is already another constraint with the same name;
            or if the operator is not recognized.
        TypeError
            Raises if the constraint is not symbolic.
        '''
        if name in self._cons:
            raise ValueError(f'Constraint name \'{name}\' already exists.')
        expr: Union[cs.SX, cs.MX] = lhs - rhs
        if not isinstance(expr, (cs.SX, cs.MX)):
            raise TypeError('Constraint must be symbolic.')
        if simplify:
            expr = cs.simplify(expr)

        shape = expr.shape
        if op == '==':
            is_eq = True
            lb = npy.zeros(npy.prod(shape))
        elif op == '<=':
            is_eq = False
            lb = npy.full(npy.prod(shape), -npy.inf)
        elif op == '>=':
            is_eq = False
            expr = -expr
            lb = npy.full(npy.prod(shape), -npy.inf)
        else:
            raise ValueError(f'Unrecognized operator {op}.')

        self._cons[name] = expr
        group, con, lam = \
            ('_g', '_lbg', '_lam_g') if is_eq else ('_h', '_lbh', '_lam_h')
        self._debug.register(group[1:], name, shape)
        lam_sym = self._csXX.sym(f'{lam[1:]}_{name}', *shape)
        setattr(self, group, cs.vertcat(getattr(self, group), cs.vec(expr)))
        setattr(self, con, npy.concatenate((getattr(self, con), lb)))
        setattr(self, lam, cs.vertcat(getattr(self, lam), cs.vec(lam_sym)))
        return expr, lam_sym

    def minimize(self, objective: Union[cs.SX, cs.MX]) -> None:
        '''Sets the objective function to be minimized.

        Parameters
        ----------
        objective : Union[cs.SX, cs.MX]
            A Symbolic variable dependent only on the MPC variables and 
            parameters that needs to be minimized. 
        '''
        self._f = objective

    def init_solver(self, opts: Dict[str, Any]) -> None:
        '''Initializes the IPOPT solver for this MPC with the given options.

        Parameters
        ----------
        opts : Dict[str, Any]
            Options to be passed to the CasADi interface to the solver.
        '''
        opts = opts.copy()
        con = cs.vertcat(self._g, self._h)
        nlp = {'x': self._x, 'p': self._p, 'g': con, 'f': self._f}
        self._solver = cs.nlpsol(f'nlpsol_{self.name}', 'ipopt', nlp, opts)
        self._solver_opts = opts

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
