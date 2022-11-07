from typing import Tuple, Union
import casadi as cs
import numpy as np
from casadi_nlp.wrappers.wrapper import Wrapper, NlpType
from casadi_nlp.util import cached_property, cached_property_reset


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

    @cached_property
    def essential_x_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        '''Gets the indices of `lbx` and `ubx` that are not redundant, i.e., 
        `i` where `lbx[i] != -inf` and `ubx[i] != +inf`.
        '''
        return (np.where(self.nlp._lbx != -np.inf)[0], 
                np.where(self.nlp._ubx != np.inf)[0])

    @cached_property
    def lagrangian(self) -> Union[cs.SX, cs.MX]:
        '''Gets the Lagrangian of the NLP problem.'''
        idx_lbx, idx_ubx = self.essential_x_bounds
        h_lbx = self.nlp._lbx[idx_lbx, None] - self.nlp._x[idx_lbx]
        h_ubx = self.nlp._x[idx_ubx] - self.nlp._ubx[idx_ubx, None]
        return (self.nlp._f +
                cs.dot(self.nlp._lam_g, self.nlp._g) +
                cs.dot(self.nlp._lam_h, self.nlp._h) +
                cs.dot(self.nlp._lam_lbx[idx_lbx], h_lbx) +
                cs.dot(self.nlp._lam_ubx[idx_ubx], h_ubx))

    @cached_property_reset(essential_x_bounds, lagrangian)
    def variable(self, *args, **kwargs):
        return self.nlp.variable(*args, **kwargs)
    
    @cached_property_reset(lagrangian)
    def constraint(self, *args, **kwargs):
        return self.nlp.constraint(*args, **kwargs)
    
    @cached_property_reset(lagrangian)
    def minimize(self, *args, **kwargs):
        return self.nlp.minimize(*args, **kwargs)
