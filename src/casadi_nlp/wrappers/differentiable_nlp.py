from typing import Dict, Tuple, Union, Optional
import casadi as cs
import numpy as np
from casadi_nlp.nlp import _DUAL_VARIABLES_ORDER
from casadi_nlp.wrappers.wrapper import Wrapper, NlpType
from casadi_nlp.solutions import Solution
from casadi_nlp.util import cached_property, cache_clearer


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
        include_barrier_term: bool = True
    ) -> None:
        '''Instantiates the wrapper.

        Parameters
        ----------
        nlp : NlpType
            The NLP problem to be wrapped.
        include_barrier_term : bool, optional
            If `True`, includes in the KKT matrix a new symbolic variable that
            represents the barrier function of the interior-point solver.
            Otherwise, no additional variable is added. See property `kkt` for
            more details. By default `True`.
        '''
        super().__init__(nlp)
        self.include_barrier_term = include_barrier_term
        self._tau = self.nlp._csXX.sym('tau') if include_barrier_term else 0

    @cached_property
    def lagrangian(self) -> Union[cs.SX, cs.MX]:
        '''Gets the Lagrangian of the NLP problem (usually, `L`).'''
        h_lbx, lam_h_lbx = self.nlp.h_lbx
        h_ubx, lam_h_ubx = self.nlp.h_ubx
        return (self.nlp._f +
                cs.dot(self.nlp._lam_g, self.nlp._g) +
                cs.dot(self.nlp._lam_h, self.nlp._h) +
                cs.dot(lam_h_lbx, h_lbx) +
                cs.dot(lam_h_ubx, h_ubx))

    @cached_property
    def kkt(
        self
    ) -> Union[Tuple[cs.SX, Optional[cs.SX]], Tuple[cs.MX, Optional[cs.MX]]]:
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
        which is also returned as the second element of the tuple. Otherwise,
        `tau` is `None`.

        Note: The order of the KKT conditions can be adjusted via the
        `_PRIMAL_DUAL_ORDER` dict.
        '''
        items = {
            'g': self.nlp._g,
            'h': (self.nlp._lam_h * self.nlp._h) + self._tau,
            'h_lbx': (self.nlp.h_lbx[0] * self.nlp.h_lbx[1]) + self._tau,
            'h_ubx': (self.nlp.h_ubx[0] * self.nlp.h_ubx[1]) + self._tau
        }
        kkt = cs.vertcat(
            cs.jacobian(self.lagrangian, self.nlp._x).T,
            *(items.pop(v) for v in _DUAL_VARIABLES_ORDER)
        )
        assert not items, 'Internal error. _DUAL_VARIABLES_ORDER modified.'
        return kkt, (self._tau if self.include_barrier_term else None)

    @cached_property
    def jacobians(self) -> Dict[str, Union[cs.SX, cs.MX]]:
        '''Computes various partial derivatives, which are then grouped in a
        dict with the following entries
            - `L-p`: lagrangian w.r.t. parameters
            - `K-p`: kkt conditions w.r.t. parameters
            - `K-y`: kkt conditions w.r.t. primal-dual variables
            - `g-x`: equality constraints w.r.t. primal variables
            - `h-x`: inequality constraints w.r.t. primal variables
        '''
        # in case of MX, jacobians throw if the MX are indexed (no more
        # symbolical according to the exception)
        K = self.kkt[0]
        return {
            'L-p': cs.jacobian(self.lagrangian, self.nlp._p),
            'K-p': cs.jacobian(K, self.nlp._p),
            'K-y': cs.jacobian(K, self.nlp.primal_dual_vars),
            'g-x': cs.jacobian(self.nlp._g, self.nlp._x),
            'h-x': cs.jacobian(self.nlp._h, self.nlp._x),
        }

    @cached_property
    def hessians(self) -> Dict[str, Union[cs.SX, cs.MX]]:
        '''Computes various partial hessians, which are then grouped in a
        dict with the following entries
            - `L-pp`: lagrangian w.r.t. parameters (twice)
            - `L-xx`: lagrangian w.r.t. primal variables (twice)
            - `L-px`: lagrangian w.r.t. parameters and then primal variables
        '''
        L = self.lagrangian
        Lpp, Lp = cs.hessian(L, self.nlp._p)
        Lpx = cs.jacobian(Lp, self.nlp._x)
        return {
            'L-pp': Lpp,
            'L-xx': cs.hessian(L, self.nlp._x)[0],
            'L-px': Lpx,
        }

    @property
    def licq(self) -> Union[np.ndarray, cs.SX, cs.MX]:
        '''Gets the symbolic matrix for LICQ, defined as
        ```
                    LICQ = [ dgdx^T, dhdx^T ]^T
        ```
        If the matrix is linear independent, then the NLP satisfies the Linear
        Independence Constraint Qualification.

        Note:
            1) the LICQ are computed for only the active inenquality
               constraints. Since this is a symbolic representation, all are
               included, and it's up to the user to eliminate
               the inactive.

            2) lower and upper bound inequality constraints are not included in
               `h` since they are by nature linear independent.
        '''
        return cs.vertcat(self.jacobians['g-x'], self.jacobians['h-x'])

    def parametric_sensitivity(
        self,
        solution: Optional[Solution] = None
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
        dKdy = self.jacobians['K-y']
        dKdp = self.jacobians['K-p']
        return (
            (-cs.inv(dKdy) @ dKdp)
            if solution is None else
            (np.linalg.solve(solution.value(dKdy), -solution.value(dKdp)))
        )

    @cache_clearer(jacobians, hessians)
    def parameter(self, *args, **kwargs):
        return self.nlp.parameter(*args, **kwargs)

    @cache_clearer(lagrangian, kkt, jacobians, hessians)
    def variable(self, *args, **kwargs):
        return self.nlp.variable(*args, **kwargs)

    @cache_clearer(lagrangian, kkt, jacobians, hessians)
    def constraint(self, *args, **kwargs):
        return self.nlp.constraint(*args, **kwargs)

    @cache_clearer(lagrangian, kkt, jacobians, hessians)
    def minimize(self, *args, **kwargs):
        return self.nlp.minimize(*args, **kwargs)
