from typing import Dict, Tuple, Union, Optional
import casadi as cs
import numpy as np
from csnlp.nlp import _DUAL_VARIABLES_ORDER
from csnlp.wrappers.wrapper import Wrapper, NlpType
from csnlp.solutions import Solution
from csnlp.util.funcs import cached_property, cache_clearer
from csnlp.util.data import cs2array, array2cs
from csnlp.util.array import hojacobian, hohessian


class NlpSensitivity(Wrapper[NlpType]):
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
        '''Instantiates the wrapper for performing NLP sensitivities.

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
        y, idx = self._y_idx
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
    def hojacobians(self) -> Dict[str, np.ndarray]:
        '''Computes various 3D jacobians, which are then grouped in a dict with
        the following entries
            - `K-pp`: kkt conditions w.r.t. parameters (twice)
            - `K-yp`: kkt conditions w.r.t. parameters and primal variables
            - `K-yy`: kkt conditions w.r.t. primal variables (twice)
            - `K-py`: kkt conditions w.r.t. primal variables and parameters
        '''
        jacobians = self.jacobians
        Kp = jacobians['K-p']
        Ky = jacobians['K-y']
        y, idx = self._y_idx
        return {
            'K-pp': hojacobian(Kp, self.nlp._p)[..., 0],
            'K-yp': hojacobian(Ky, self.nlp._p)[..., 0],
            'K-yy': hojacobian(Ky, y)[..., idx, 0],
            'K-py': hojacobian(Kp, y)[..., idx, 0],
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
        expr: Union[cs.SX, cs.MX] = None,
        solution: Optional[Solution] = None,
    ) -> Union[
        Tuple[np.ndarray, np.ndarray], Tuple[cs.SX, cs.SX], Tuple[cs.MX, cs.MX]
    ]:
        '''Performs the (symbolic or numerical) sensitivity of the NLP w.r.t.
        its parametrization, according to [1].

        Parameters
        ----------
        expression : cs.SX or MX, optional
            If provided, computes the sensitivity of this expression (which
            must be dependent on the primal-dual variables and/or parameters of
            the NLP) w.r.t. the NLP parameters. If `None`, then the sensitivity
            of the primal-dual variables is returned.
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
        # first order sensitivity, a.k.a., dydp
        Ky = self.jacobians['K-y']
        Kp = self.jacobians['K-p']
        if solution is None:
            dydp = -cs.inv(Ky) @ Kp
        else:
            A = solution.value(Ky)
            b = -solution.value(Kp)
            dydp = np.linalg.solve(A, b)

        # second order sensitivity, a.k.a., d2ydp2
        dydp = cs2array(dydp)
        Kpp = self.hojacobians['K-pp']
        Kpy = self.hojacobians['K-py']
        Kyp = self.hojacobians['K-yp']
        Kyy = self.hojacobians['K-yy']
        M = (
            Kpp + (
                Kpy.transpose((0, 2, 1)) +
                Kyp +
                (Kyy @ dydp)
            ).transpose((0, 2, 1)) @ dydp
        )
        if solution is None:
            A = cs2array(cs.inv(Ky))
            b = -M.transpose((2, 0, 1))
            d2ydp2 = (A @ b).transpose((1, 2, 0))
        else:
            A = solution.value(Ky)
            b = -solution.value(M).transpose((2, 0, 1))
            d2ydp2 = np.linalg.solve(A, b).transpose((1, 2, 0))

        # sensitivity of custom expression
        Z = expr  # Z := z(x(p),lam(p),p)
        if Z is None:
            dydp = dydp.squeeze()
            d2ydp2 = d2ydp2.squeeze()
            if solution is not None:
                return dydp, d2ydp2
            return (
                array2cs(dydp),
                array2cs(d2ydp2) if d2ydp2.ndim <= 2 else d2ydp2
            )
        Zshape = Z.shape
        Z = cs.vec(Z)

        p = self.nlp._p
        y, idx = self._y_idx
        Zpp, Zp = (o.squeeze() for o in hohessian(Z, p))
        Zyy, Zy = (o.squeeze() for o in hohessian(Z, y))
        Zyp = hohessian(Z, y, p)[0].squeeze()
        Zpy = hohessian(Z, p, y)[0].squeeze()
        Zy = Zy[..., idx]
        Zyy = Zyy[..., idx, :][..., idx]
        Zyp = Zyp[..., idx, :]
        Zpy = Zpy[..., idx]

        dzdp = Zy @ dydp + Zp
        T1 = (d2ydp2.transpose((1, 2, 0)) @ Zy.T).transpose((2, 0, 1))
        T2 = Zyy.transpose((0, 2, 1)) @ dydp + Zpy.transpose((0, 2, 1)) + Zyp
        d2zdp2 = Zpp + T1 + dydp.T @ T2

        np_ = self.nlp.np
        dzdp = dzdp.reshape(Zshape + (np_,), order='F')
        d2zdp2 = d2zdp2.reshape(Zshape + (np_, np_), order='F')
        if solution is not None:
            return solution.value(dzdp), solution.value(d2zdp2)
        return dzdp, d2zdp2

    @cache_clearer(jacobians, hessians, hojacobians)
    def parameter(self, *args, **kwargs):
        return self.nlp.parameter(*args, **kwargs)

    @cache_clearer(lagrangian, kkt, jacobians, hessians, hojacobians)
    def variable(self, *args, **kwargs):
        return self.nlp.variable(*args, **kwargs)

    @cache_clearer(lagrangian, kkt, jacobians, hessians, hojacobians)
    def constraint(self, *args, **kwargs):
        return self.nlp.constraint(*args, **kwargs)

    @cache_clearer(lagrangian, kkt, jacobians, hessians, hojacobians)
    def minimize(self, *args, **kwargs):
        return self.nlp.minimize(*args, **kwargs)

    @property
    def _y_idx(self) -> Tuple[Union[cs.SX, cs.MX], np.ndarray]:
        '''Internal utility to return all the primal-dual variables and indices
        that are associated to non-redundant entries in the kkt conditions.'''
        if self.nlp._csXX is cs.SX:
            y = self.nlp.primal_dual_vars()
            idx = slice(None)
        else:
            # in case of MX, jacobians throw if the MX are indexed (no more
            # symbolical according to the exception). So we take the jacobian
            # with all primal-dual vars, and then index the relevant rows/cols.
            y = self.nlp.primal_dual_vars(all=True)
            h_lbx_idx = np.where(self.nlp._lbx != -np.inf)[0]
            h_ubx_idx = np.where(self.nlp._ubx != +np.inf)[0]
            n = self.nlp.nx + self.nlp.ng + self.nlp.nh
            idx = np.concatenate((
                np.arange(n),
                h_lbx_idx + n,
                h_ubx_idx + n + h_lbx_idx.size
            ))
        return y, idx
