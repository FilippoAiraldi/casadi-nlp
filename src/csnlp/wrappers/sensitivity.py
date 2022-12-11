from typing import Callable, Dict, Literal, Optional, Tuple, TypeVar, Union

import casadi as cs
import numpy as np
import numpy.typing as npt

from csnlp.core.cache import cached_property, invalidate_cache
from csnlp.core.data import array2cs, cs2array
from csnlp.core.derivatives import hohessian, hojacobian
from csnlp.core.solutions import Solution
from csnlp.wrappers.wrapper import Nlp, Wrapper

T = TypeVar("T", cs.SX, cs.MX)


class NlpSensitivity(Wrapper[T]):
    """
    Wraps an NLP to allow to perform numerical sensitivity analysis and compute its
    derivates. See [1] for nonlinear programming sensitivity analysis.

    References
    ----------
    [1] Buskens, C. and Maurer, H. (2001). Sensitivity analysis and real-time
        optimization of parametric nonlinear programming problems. In M. Grotschel, S.O.
        Krumke, and J. Rambau (eds.), Online Optimization of Large Scale Systems, 3–16.
        Springer, Berlin, Heidelberg.
    """

    def __init__(self, nlp: Nlp[T], include_barrier_term: bool = True) -> None:
        """Instantiates the wrapper for performing NLP sensitivities.

        Parameters
        ----------
        nlp : Nlp
            The NLP problem to be wrapped.
        include_barrier_term : bool, optional
            If `True`, includes in the KKT matrix a new symbolic variable that
            represents the barrier function of the interior-point solver. Otherwise, no
            additional variable is added. See property `kkt` for more details. By
            default `True`.
        """
        super().__init__(nlp)
        self.include_barrier_term = include_barrier_term
        self._tau = nlp.sym_type.sym("tau") if include_barrier_term else 0

    @cached_property
    def lagrangian(self) -> T:
        """Gets the Lagrangian of the NLP problem (usually, `L`)."""
        h_lbx, lam_h_lbx = self.nlp.h_lbx
        h_ubx, lam_h_ubx = self.nlp.h_ubx
        return (
            self.nlp.f
            + cs.dot(self.nlp.lam_g, self.nlp.g)
            + cs.dot(self.nlp.lam_h, self.nlp.h)
            + cs.dot(lam_h_lbx, h_lbx)
            + cs.dot(lam_h_ubx, h_ubx)
        )

    @cached_property
    def kkt(self) -> Tuple[T, Optional[T]]:
        """Gets the KKT conditions of the NLP problem in vector form, i.e.,
        ```
                            |      dLdx     |
                        K = |       G       |
                            | diag(lam_h)*H |
        ```
        where `dLdx` is the gradient of the lagrangian w.r.t. the primal variables `x`,
        `G` collects the equality constraints, `H` collects the inequality constraints
        and `lam_h` its corresponding dual variables.

        If `include_barrier_term=True`, the inequalities include an additional barrier
        term `tau`, so that
        ```
                            diag(lam_h)*H + tau
        ```
        which is also returned as the second element of the tuple. Otherwise, `tau` is
        `None`.

        Note: The order of the KKT conditions can be adjusted via the class attribute
        `_PRIMAL_DUAL_ORDER`.
        """
        items = {
            "g": self.nlp.g,
            "h": (self.nlp.lam_h * self.nlp.h) + self._tau,
            "h_lbx": (self.nlp.h_lbx[0] * self.nlp.h_lbx[1]) + self._tau,
            "h_ubx": (self.nlp.h_ubx[0] * self.nlp.h_ubx[1]) + self._tau,
        }
        kkt = cs.vertcat(
            cs.jacobian(self.lagrangian, self.nlp.x).T,
            *(items.pop(v) for v in self.nlp.dual_variables_order)
        )
        assert not items, "Internal error. `dual_variables_order` modified."
        return kkt, (self._tau if self.include_barrier_term else None)

    @cached_property
    def jacobians(self) -> Dict[str, T]:
        """Computes various partial derivatives, which are then grouped in a dict with
        the following entries
            - `L-p`: lagrangian w.r.t. parameters
            - `g-x`: equality constraints w.r.t. primal variables
            - `h-x`: inequality constraints w.r.t. primal variables
            - `K-p`: kkt conditions w.r.t. parameters
            - `K-y`: kkt conditions w.r.t. primal-dual variables
        """
        K = self.kkt[0]
        y, idx = self._y_idx
        return {
            "L-p": cs.jacobian(self.lagrangian, self.nlp.p),
            "g-x": cs.jacobian(self.nlp.g, self.nlp.x),
            "h-x": cs.jacobian(self.nlp.h, self.nlp.x),
            "K-p": cs.jacobian(K, self.nlp.p),
            "K-y": cs.jacobian(K, y)[:, idx],
        }

    @cached_property
    def hessians(self) -> Dict[str, T]:
        """Computes various partial hessians, which are then grouped in a dict with the
        following entries
            - `L-pp`: lagrangian w.r.t. parameters (twice)
            - `L-xx`: lagrangian w.r.t. primal variables (twice)
            - `L-px`: lagrangian w.r.t. parameters and then primal variables
        """
        L = self.lagrangian
        Lpp, Lp = cs.hessian(L, self.nlp.p)
        Lpx = cs.jacobian(Lp, self.nlp.x)
        return {
            "L-pp": Lpp,
            "L-xx": cs.hessian(L, self.nlp.x)[0],
            "L-px": Lpx,
        }

    @cached_property
    def hojacobians(self) -> Dict[str, np.ndarray]:
        """Computes various 3D jacobians, which are then grouped in a dict with the
        following entries
            - `K-pp`: kkt conditions w.r.t. parameters (twice)
            - `K-yp`: kkt conditions w.r.t. parameters and primal variables
            - `K-yy`: kkt conditions w.r.t. primal variables (twice)
            - `K-py`: kkt conditions w.r.t. primal variables and parameters
        """
        jacobians = self.jacobians
        Kp = jacobians["K-p"]
        Ky = jacobians["K-y"]
        y, idx = self._y_idx
        return {
            "K-pp": hojacobian(Kp, self.nlp.p)[..., 0],
            "K-yp": hojacobian(Ky, self.nlp.p)[..., 0],
            "K-yy": hojacobian(Ky, y)[..., idx, 0],
            "K-py": hojacobian(Kp, y)[..., idx, 0],
        }

    @property
    def licq(self) -> T:
        """Gets the symbolic matrix for LICQ, defined as
        ```
                    LICQ = [ dgdx^T, dhdx^T ]^T
        ```
        If the matrix is linear independent, then the NLP satisfies the Linear
        Independence Constraint Qualification.

        Note:
            1) the LICQ are computed for only the active inenquality constraints. Since
               this is a symbolic representation, all are included, and it's up to the
               user to eliminate the inactive.

            2) lower and upper bound inequality constraints are not included in `h`
               since they are by nature linear independent.
        """
        return cs.vertcat(self.jacobians["g-x"], self.jacobians["h-x"])

    def parametric_sensitivity(
        self,
        expr: T = None,
        solution: Optional[Solution[T]] = None,
        second_order: bool = False,
    ) -> Union[
        Tuple[T, Optional[T]],
        Tuple[cs.DM, Optional[cs.DM]],
        Tuple[npt.NDArray[np.double], Optional[npt.NDArray[np.double]]],
    ]:
        """Performs the (symbolic or numerical) sensitivity of the NLP w.r.t. its
        parametrization, according to [1].

        Parameters
        ----------
        expression : casadi.SX or MX, optional
            If provided, computes the sensitivity of this expression (which must be
            dependent on the primal-dual variables and/or parameters of the NLP) w.r.t.
            the NLP parameters. If `None`, then the sensitivity of the primal-dual
            variables is returned.
        solution : Solution, optional
            If a solution is passed, then the sensitivity is numerically computed for
            that solution; otherwise, the sensitivity is carried out symbolically
            (however, this is much more computationally intensive).
        second_order : bool, optional
            If `second_order=False`, the analysis is stopped at the first order;
            otherwise, also second-order information is computed.

        Returns
        -------
        2-element tuple of casadi.SX, or MX, or DM, or arrays
            The 1st and 2nd-order NLP parametric sensitivity in the form of an array/DM,
            if a solution is passed, or a symbolic vector SX/MX. When
            `second_order=False` the second element in the tuple is `None`.

        Raises
        ------
        numpy.linalg.LinAlgError
            Raises if the KKT conditions lead to a singular matrix.

        References
        ----------
        [1] Buskens, C. and Maurer, H. (2001). Sensitivity analysis and real-time
            optimization of parametric nonlinear programming problems. In M. Grotschel,
            S.O. Krumke, and J. Rambau (eds.), Online Optimization of Large Scale
            Systems, 3–16. Springer, Berlin, Heidelberg.
        """
        d: Callable[[T], Union[T, cs.DM, np.ndarray]] = (
            (lambda o: o)
            if solution is None
            else (lambda o: solution.value(o))  # type: ignore
        )

        # first order sensitivity, a.k.a., dydp
        Ky = d(self.jacobians["K-y"])
        Kp = d(self.jacobians["K-p"])
        if solution is None:
            Ky_inv = cs.inv(Ky)
            dydp = -Ky_inv @ Kp
        else:
            dydp = -np.linalg.solve(Ky, Kp)

        # first order sensitivity of Z, a.k.a., dZdp
        Z = expr  # Z := z(x(p),lam(p),p)
        if Z is not None:
            Zshape = Z.shape
            Z = cs.vec(Z)
            p = self.nlp.p
            np_ = self.nlp.np
            y, idx = self._y_idx

            if second_order:
                Zpp, Zp = hohessian(Z, p)
                Zyy, Zy = hohessian(Z, y)
            else:
                Zp = hojacobian(Z, p)
                Zy = hojacobian(Z, y)
            Zp = d(array2cs(Zp.squeeze((1, 3))))
            Zy = d(array2cs(Zy[:, 0, idx, 0]))

            dZdp = Zy @ dydp + Zp
            dZdp = cs2array(dZdp).reshape(Zshape + (np_,), order="F").squeeze()
            if dZdp.ndim <= 2:
                dZdp = array2cs(dZdp)

        # if no second order is required, just return
        if not second_order:
            return dydp if Z is None else dZdp, None

        # second order sensitivity, a.k.a., d2ydp2
        dydp = cs2array(dydp)
        Kpp = d(self.hojacobians["K-pp"])
        Kpy = d(self.hojacobians["K-py"])
        Kyp = d(self.hojacobians["K-yp"])
        Kyy = d(self.hojacobians["K-yy"])
        M = (
            Kpp
            + (Kpy.transpose((0, 2, 1)) + Kyp + (Kyy @ dydp)).transpose((0, 2, 1))
            @ dydp
        ).transpose((2, 0, 1))
        if solution is None:
            d2ydp2 = -(cs2array(Ky_inv) @ M).transpose((1, 2, 0))
        else:
            d2ydp2 = -np.linalg.solve(Ky, M).transpose((1, 2, 0))

        # if no expression is given, just return
        if Z is None:
            dydp = array2cs(dydp.squeeze())
            d2ydp2 = d2ydp2.squeeze()
            if d2ydp2.ndim <= 2:
                d2ydp2 = array2cs(d2ydp2)
            return dydp, d2ydp2

        # second order sensitivity of Z, a.k.a., d2Zdp2
        Zyp = hohessian(Z, y, p)[0]
        Zpy = hohessian(Z, p, y)[0]
        Zpp = d(Zpp.squeeze((1, 3, 5)))
        Zyy = d(Zyy.squeeze((1, 3, 5))[:, idx, :][..., idx])
        Zyp = d(Zyp.squeeze((1, 3, 5))[:, idx, :])
        Zpy = d(Zpy.squeeze((1, 3, 5))[..., idx])
        T1 = (d2ydp2.transpose((1, 2, 0)) @ cs2array(Zy).T).transpose((2, 0, 1))
        T2 = Zyy.transpose((0, 2, 1)) @ dydp + Zpy.transpose((0, 2, 1)) + Zyp
        d2Zdp2 = Zpp + T1 + dydp.T @ T2
        d2Zdp2 = d2Zdp2.reshape(Zshape + (np_, np_), order="F").squeeze()
        if d2Zdp2.ndim <= 2:
            d2Zdp2 = array2cs(d2Zdp2)
        return dZdp, d2Zdp2

    @invalidate_cache(jacobians, hessians, hojacobians)
    def parameter(self, name: str, shape: Tuple[int, int] = (1, 1)) -> T:
        """See `Nlp.parameter` method."""
        return self.nlp.parameter(name, shape)

    @invalidate_cache(lagrangian, kkt, jacobians, hessians, hojacobians)
    def variable(
        self,
        name: str,
        shape: Tuple[int, int] = (1, 1),
        lb: Union[npt.ArrayLike, cs.DM] = -np.inf,
        ub: Union[npt.ArrayLike, cs.DM] = +np.inf,
    ) -> Tuple[T, T, T]:
        """See `Nlp.variable` method."""
        return self.nlp.variable(name, shape, lb, ub)

    @invalidate_cache(lagrangian, kkt, jacobians, hessians, hojacobians)
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
        return self.nlp.constraint(name, lhs, op, rhs, soft, simplify)

    @invalidate_cache(lagrangian, kkt, jacobians, hessians, hojacobians)
    def minimize(self, objective: T) -> None:
        """See `Nlp.minimize` method."""
        return self.nlp.minimize(objective)

    @property
    def _y_idx(self) -> Tuple[T, Union[slice, npt.NDArray[np.int64]]]:
        """Internal utility to return all the primal-dual variables and indices that are
        associated to non-redundant entries in the kkt conditions."""
        if self.nlp.sym_type is cs.SX:
            y = self.nlp.primal_dual_vars()
            idx = slice(None)
        else:
            # in case of MX, jacobians throw if the MX are indexed (no more
            # symbolical according to the exception). So we take the jacobian
            # with all primal-dual vars, and then index the relevant rows/cols.
            y = self.nlp.primal_dual_vars(all=True)
            h_lbx_idx = np.where(self.nlp.lbx != -np.inf)[0]
            h_ubx_idx = np.where(self.nlp.ubx != +np.inf)[0]
            n = self.nlp.nx + self.nlp.ng + self.nlp.nh
            idx = np.concatenate(
                (np.arange(n), h_lbx_idx + n, h_ubx_idx + n + h_lbx_idx.size)
            )
        return y, idx
