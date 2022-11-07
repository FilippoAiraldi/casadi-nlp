from typing import Tuple, Union
import casadi as cs
import numpy as np
from casadi_nlp.wrappers.wrapper import Wrapper, NlpType
from casadi_nlp.util import cached_property, cached_property_reset


PRIMAL_DUAL_ORDER = {
    'x': lambda nlp: (nlp._x,),
    'g': lambda nlp: (nlp._lam_g,),
    'h': lambda nlp: (nlp._lam_h,),
    'h_lbx': lambda nlp: (nlp.h_lbx[1],),
    'h_ubx': lambda nlp: (nlp.h_ubx[1],),
}


class DifferentiableNlp(Wrapper[NlpType]):
    '''
    Wraps an NLP to allow to perform numerical sensitivity analysis and compute
    its derivates.

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
        simplify_x_bounds: bool = True
    ) -> None:
        super().__init__(nlp)
        self.remove_reduntant_x_bounds = simplify_x_bounds

    @cached_property
    def h_lbx(self) -> Union[Tuple[cs.SX, cs.SX], Tuple[cs.MX, cs.MX]]:
        '''Gets the inequalities due to `lbx` and their multipliers. Removes 
        redundant entries, i.e., `lbx == -inf`.
        '''
        if self.remove_reduntant_x_bounds:
            idx = np.where(self.nlp._lbx != -np.inf)[0]
        else:
            idx = np.arange(self.nlp.nx)
        h = self.nlp._lbx[idx, None] - self.nlp._x[idx]
        return h, self.nlp._lam_lbx[idx]

    @cached_property
    def h_ubx(self) -> Union[Tuple[cs.SX, cs.SX], Tuple[cs.MX, cs.MX]]:
        '''Gets the inequalities due to `ubx` and their multipliers. Removes 
        redundant entries, i.e., `lubx == +inf`.
        '''
        if self.remove_reduntant_x_bounds:
            idx = np.where(self.nlp._ubx != np.inf)[0]
        else:
            idx = np.arange(self.nlp.nx)
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
    def primal_dual_variables(self) -> Union[cs.SX, cs.MX]:
        '''Gets the collection of primal-dual variables.'''
        return cs.vertcat(*(f(self)[0] for f in PRIMAL_DUAL_ORDER.values()))

    @cached_property_reset(h_lbx, h_ubx, lagrangian, primal_dual_variables)
    def variable(self, *args, **kwargs):
        return self.nlp.variable(*args, **kwargs)

    @cached_property_reset(lagrangian, primal_dual_variables)
    def constraint(self, *args, **kwargs):
        return self.nlp.constraint(*args, **kwargs)

    @cached_property_reset(lagrangian)
    def minimize(self, *args, **kwargs):
        return self.nlp.minimize(*args, **kwargs)
