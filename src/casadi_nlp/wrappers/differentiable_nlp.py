from typing import Dict, Literal, Tuple, Union, Optional
import casadi as cs
import numpy as np
from casadi_nlp.nlp import _DUAL_VARIABLES_ORDER
from casadi_nlp.wrappers.wrapper import Wrapper, NlpType
from casadi_nlp.solutions import Solution
from casadi_nlp.util import \
    cached_property, cache_clearer, hojacobian, cs2array, array2cs


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
            - `g-x`: equality constraints w.r.t. primal variables
            - `h-x`: inequality constraints w.r.t. primal variables
            - `K-p`: kkt conditions w.r.t. parameters
            - `K-y`: kkt conditions w.r.t. primal-dual variables
        '''
        K = self.kkt[0]
        # in case of MX, jacobians throw if the MX are indexed (no more
        # symbolical according to the exception). So we take the jacobian
        # with all primal-dual vars, and then index the relevant ones.
        if self.nlp._csXX is cs.SX:
            y = self.nlp.primal_dual_vars()
            idx = slice(None)
        else:
            y = self.nlp.primal_dual_vars(all=True)
            h_lbx_idx = np.where(self.nlp._lbx != -np.inf)[0]
            h_ubx_idx = np.where(self.nlp._ubx != +np.inf)[0]
            n = self.nlp.nx + self.nlp.ng + self.nlp.nh
            idx = np.concatenate((
                np.arange(n),
                h_lbx_idx + n,
                h_ubx_idx + n + h_lbx_idx.size
            ))
        return {
            'L-p': cs.jacobian(self.lagrangian, self.nlp._p),
            'g-x': cs.jacobian(self.nlp._g, self.nlp._x),
            'h-x': cs.jacobian(self.nlp._h, self.nlp._x),
            'K-p': cs.jacobian(K, self.nlp._p),
            'K-y': cs.jacobian(K, y)[:, idx],
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

    @cached_property
    def hoderivatives(self) -> Dict[str, np.ndarray]:
        '''Computes various 3rd-order derivatives, which are then grouped in a
        dict with the following entries
            - `K-pp`: kkt conditions w.r.t. parameters (twice)
            - `K-yp`: kkt conditions w.r.t. parameters and primal variables
            - `K-yy`: kkt conditions w.r.t. primal variables (twice)
            - `K-py`: kkt conditions w.r.t. primal variables and parameters
        '''
        jacobians = self.jacobians
        Kp = jacobians['K-p']
        Ky = jacobians['K-y']
        # in case of MX, jacobians throw if the MX are indexed (no more
        # symbolical according to the exception). So we take the jacobian
        # with all primal-dual vars, and then index the relevant ones.
        if self.nlp._csXX is cs.SX:
            y = self.nlp.primal_dual_vars()
            idx = slice(None)
        else:
            y = self.nlp.primal_dual_vars(all=True)
            h_lbx_idx = np.where(self.nlp._lbx != -np.inf)[0]
            h_ubx_idx = np.where(self.nlp._ubx != +np.inf)[0]
            n = self.nlp.nx + self.nlp.ng + self.nlp.nh
            idx = np.concatenate((
                np.arange(n),
                h_lbx_idx + n,
                h_ubx_idx + n + h_lbx_idx.size
            ))
        return {
            'K-pp': hojacobian(Kp, self.nlp._p).squeeze(),
            'K-yp': hojacobian(Ky, self.nlp._p).squeeze(),
            'K-yy': hojacobian(Ky, y).squeeze()[..., idx],
            'K-py': hojacobian(Kp, y).squeeze()[..., idx],
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
        solution: Optional[Solution] = None,
        order: Literal[1, 2] = 1,
        p_index: int = None
    ) -> Union[
        Tuple[np.ndarray, np.ndarray], Tuple[cs.SX, cs.SX], Tuple[cs.MX, cs.MX]
    ]:
        '''Performs the (symbolic or numerical) sensitivity of the NLP solution
        w.r.t. its parametrization, according to [1].

        Parameters
        ----------
        solution : Solution, optional
            If a solution is passed, then the sensitivity is numerically
            computed for that solution; otherwise, the sensitivity is carried
            out symbolically (however, this is much more computationally
            intensive).
        order : 1 or 2, optional
            Order of the sensitivity analysis. By default, first order.
        p_index : int, optional
            If `order==2`, then the 2nd order sensitivity analysis can be
            performed only on one parameter, a.k.a., the one specified by this
            index.

        Returns
        -------
        Union[np.ndarray, cs.SX, cs.MX]
            The NLP parametric sensitivity in the form of an numerical array,
            if a solution is passed, or a symbolic vector.

        Raises
        ------
        ValueError
            Raises if `order==2` but `p_index` is None or outside bounds.
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

        # first order sensitivity, a.k.a., dydp
        Ky = self.jacobians['K-y']
        Kp = self.jacobians['K-p']
        if solution is None:
            dydp = -cs.inv(Ky) @ Kp
            d2ydp2 = self.nlp._csXX()  # create empty symbol
        else:
            A = solution.value(Ky)
            b = -solution.value(Kp)
            dydp = np.linalg.solve(A, b)
            d2ydp2 = np.array([])  # create empty array
        if order == 1:
            return dydp, d2ydp2

        # second order sensitivity, a.k.a., d2ydp2
        if p_index is None or (p_index < 0 or p_index >= self.nlp.np):
            raise ValueError('Invalid parameter index for 2nd order '
                             'sensitivity analysis')
        Kpp = self.hoderivatives['K-pp']
        Kpy = self.hoderivatives['K-py']
        Kyp = self.hoderivatives['K-yp']
        Kyy = self.hoderivatives['K-yy']

        dydp = dydp[:, p_index]
        Kpp = Kpp[:, p_index, p_index]
        Kpy = Kpy[:, p_index, :]
        Kyp = Kyp[..., p_index]

        dydp_ = cs2array(dydp)
        M = ((Kpy + Kyp + (Kyy @ dydp_).squeeze()) @ dydp_).squeeze()
        if solution is None:
            d2ydp2 = -cs.inv(Ky) @ array2cs(Kpp + M)
        else:
            b = -solution.value(Kpp + M)
            d2ydp2 = np.linalg.solve(A, b)
        return dydp, d2ydp2

    @cache_clearer(jacobians, hessians, hoderivatives)
    def parameter(self, *args, **kwargs):
        return self.nlp.parameter(*args, **kwargs)

    @cache_clearer(lagrangian, kkt, jacobians, hessians, hoderivatives)
    def variable(self, *args, **kwargs):
        return self.nlp.variable(*args, **kwargs)

    @cache_clearer(lagrangian, kkt, jacobians, hessians, hoderivatives)
    def constraint(self, *args, **kwargs):
        return self.nlp.constraint(*args, **kwargs)

    @cache_clearer(lagrangian, kkt, jacobians, hessians, hoderivatives)
    def minimize(self, *args, **kwargs):
        return self.nlp.minimize(*args, **kwargs)
