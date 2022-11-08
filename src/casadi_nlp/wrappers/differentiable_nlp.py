from typing import Tuple, Union
import casadi as cs
import numpy as np
from casadi_nlp.wrappers.wrapper import Wrapper, NlpType
from casadi_nlp.solutions import Solution
from casadi_nlp.util import cached_property, cached_property_reset


'''Dict that dictates the order for operations related to primal-dual
variables. Each entry is a type of variable, and contains functions
    1) for grouping them in vector (see `primal_dual_variables`)
    2) for forming the kkt conditions (see `kkt`)'''
_PRIMAL_DUAL_ORDER = {
    'x': (
        lambda wpr: wpr.nlp._x,
        lambda wpr: cs.jacobian(wpr.lagrangian, wpr.nlp._x).T
    ),
    'g': (
        lambda wpr: wpr.nlp._lam_g,
        lambda wpr: wpr.nlp._g
    ),
    'h': (
        lambda wpr: wpr.nlp._lam_h,
        lambda wpr: wpr.nlp._lam_h * wpr.nlp._h
    ),
    'h_lbx': (
        lambda wpr: wpr.h_lbx[1],
        lambda wpr: wpr.h_lbx[0] * wpr.h_lbx[1],
    ),
    'h_ubx': (
        lambda wpr: wpr.h_ubx[1],
        lambda wpr: wpr.h_ubx[0] * wpr.h_ubx[1],
    )
}


class DifferentiableNlp(Wrapper[NlpType]):
    '''
    Wraps an NLP to allow to perform numerical sensitivity analysis and compute
    its derivates. See [1] for nonlinear programming sensitivity analysis.

    References
    ----------
    [1] Buskens, C. and Maurer, H. (2001). Sensitivity analysis and real-time
        optimization of parametric nonlinear programming problems. In M.
        Grotschel, S.O. Krumke, and J. Rambau (eds.), Online Optimization of
        Large Scale Systems, 3–16. Springer, Berlin, Heidelberg.
    '''

    def __init__(
        self,
        nlp: NlpType,
        simplify_x_bounds: bool = True,
        include_barrier_term: bool = True
    ) -> None:
        '''Instantiates the wrapper.

        Parameters
        ----------
        nlp : NlpType
            The NLP problem to be wrapped.
        simplify_x_bounds : bool, optional
            If `True`, then redundant entries in `lbx` and `ubx` are removed;
            see properties `h_lbx` and `h_ubx` for more details. By default,
            `True`.
        include_barrier_term : bool, optional
            If `True`, includes in the KKT matrix a new symbolic variable that
            represents the barrier function of the interior-point solver. 
            Otherwise, no additional variable is added. See property `kkt` for 
            more details. By default `True`.
        '''
        super().__init__(nlp)
        self.remove_reduntant_x_bounds = simplify_x_bounds
        self.include_barrier_term = include_barrier_term

    @cached_property
    def h_lbx(self) -> Union[Tuple[cs.SX, cs.SX], Tuple[cs.MX, cs.MX]]:
        '''Gets the inequalities due to `lbx` and their multipliers. If
        `simplify_x_bounds=True`, it removes redundant entries, i.e., where
        `lbx == -inf`; otherwise, returns all lower bound constraints.'''
        if self.remove_reduntant_x_bounds:
            idx = np.where(self.nlp._lbx != -np.inf)[0]
        else:
            idx = np.arange(self.nlp.nx)
        if idx.size == 0:
            return self.nlp._csXX(), self.nlp._csXX()
        h = self.nlp._lbx[idx, None] - self.nlp._x[idx]
        return h, self.nlp._lam_lbx[idx]

    @cached_property
    def h_ubx(self) -> Union[Tuple[cs.SX, cs.SX], Tuple[cs.MX, cs.MX]]:
        '''Gets the inequalities due to `ubx` and their multipliers. If
        `simplify_x_bounds=True`, it removes redundant entries, i.e., where
        `ubx == +inf`; otherwise, returns all upper bound constraints.'''
        if self.remove_reduntant_x_bounds:
            idx = np.where(self.nlp._ubx != np.inf)[0]
        else:
            idx = np.arange(self.nlp.nx)
        if idx.size == 0:
            return self.nlp._csXX(), self.nlp._csXX()
        h = self.nlp._x[idx] - self.nlp._ubx[idx, None]
        return h, self.nlp._lam_ubx[idx]

    @cached_property
    def lagrangian(self) -> Union[cs.SX, cs.MX]:
        '''Gets the Lagrangian of the NLP problem.'''
        h_lbx, lam_h_lbx = self.h_lbx
        h_ubx, lam_h_ubx = self.h_ubx
        return (self.nlp._f +
                cs.dot(self.nlp._lam_g, self.nlp._g) +
                cs.dot(self.nlp._lam_h, self.nlp._h) +
                cs.dot(lam_h_lbx, h_lbx) +
                cs.dot(lam_h_ubx, h_ubx))

    @cached_property
    def dual_variables(self) -> Union[cs.SX, cs.MX]:
        '''Gets the collection of dual variables.'''
        return cs.vertcat(*(f[0](self) for n, f in _PRIMAL_DUAL_ORDER.items() 
                            if n != 'x'))

    @cached_property
    def primal_dual_variables(self) -> Union[cs.SX, cs.MX]:
        '''Gets the collection of primal-dual variables.'''
        return cs.vertcat(*(f[0](self) for f in _PRIMAL_DUAL_ORDER.values()))

    @cached_property
    def kkt(self) -> Union[Tuple[cs.SX, cs.SX], Tuple[cs.MX, cs.MX]]:
        '''Gets the KKT conditions of the NLP problem in vector form, i.e.,
        ```
                        |      dLdx     |
                    K = |       G       |
                        | diag(lam_h)*H |
        ```
        where `dLdx` is the gradient of the lagrangian w.r.t. the primal
        variables `x`, `G` collects the equality constraints, `H` collects
        the inequality constraints and `lam_h` its corresponding dual
        variables.

        If `include_barrier_term=True`, the inequalities include an additional
        barrier term `tau`, so that
        ```
                        diag(lam_h)*H + tau
        ```
        which is also returned as the second element of the tuple.
        '''
        tau = self.nlp._csXX.sym('tau')

        def process(name, fcn):
            kkt_entry = fcn(self)
            if self.include_barrier_term and name[0] == 'h':
                kkt_entry += tau
            return kkt_entry

        kkt = cs.vertcat(
            *(process(n, f[1]) for n, f in _PRIMAL_DUAL_ORDER.items()))
        return kkt, tau

    def parametric_sensitivity(
        self, 
        solution: Solution = None
    ) -> Union[np.ndarray, cs.SX, cs.MX]:
        '''Performs the (symbolic or numerical) sensitivity of the NLP solution
        w.r.t. its parametrization, according to [1].

        Parameters
        ----------
        solution : Solution, optional
            If a solution is passed, then the sensitivity is numerically 
            computed for that solution; otherwise, the sensitivity is carried
            out symbolically (however, this is much more computationally 
            intensive).

        Returns
        -------
        Union[np.ndarray, cs.SX, cs.MX]
            The NLP parametric sensitivity in the form of an numerical array, 
            if a solution is passed, or a symbolic vector.

        Raises
        ------
        numpy.linalg.LinAlgError
            Raises if the KKT conditions lead to a singular matrix. 

        References
        ----------
        [1] Buskens, C. and Maurer, H. (2001). Sensitivity analysis and 
            real-time optimization of parametric nonlinear programming 
            problems. In M. Grotschel, S.O. Krumke, and J. Rambau (eds.), 
            Online Optimization of Large Scale Systems, 3–16. Springer, Berlin,
            Heidelberg.
        '''
        K = self.kkt[0]
        dKdy = cs.jacobian(K, self.primal_dual_variables)
        dKdp = cs.jacobian(K, self.nlp._p)
        return (
            (-cs.inv(dKdy) @ dKdp)
            if solution is None else
            (np.linalg.solve(solution.value(dKdy), -solution.value(dKdp)))
        )
    
    @cached_property_reset(
        h_lbx, h_ubx, lagrangian, dual_variables, primal_dual_variables, kkt)
    def variable(self, *args, **kwargs):
        return self.nlp.variable(*args, **kwargs)

    @cached_property_reset(
        lagrangian, dual_variables, primal_dual_variables, kkt)
    def constraint(self, *args, **kwargs):
        return self.nlp.constraint(*args, **kwargs)

    @cached_property_reset(lagrangian, kkt)
    def minimize(self, *args, **kwargs):
        return self.nlp.minimize(*args, **kwargs)
