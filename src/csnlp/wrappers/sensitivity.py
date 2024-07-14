from functools import cached_property
from typing import Callable, Literal, Optional, TypeVar, Union

import casadi as cs
import numpy as np
import numpy.typing as npt

from ..core.cache import invalidate_cache
from ..core.data import array2cs, cs2array, find_index_in_vector
from ..core.derivatives import hohessian, hojacobian
from ..core.solutions import Solution
from .wrapper import Nlp, Wrapper

SymType = TypeVar("SymType", cs.SX, cs.MX)


class NlpSensitivity(Wrapper[SymType]):
    """Wraps an instance of :class:`csnlp.Nlp` to allow to perform symbolical and
    numerical sensitivity analysis and compute its derivates w.r.t. to different
    quantities, including primal variables and parameters. See [1]_ for nonlinear
    programming sensitivity analysis. The computations are tailored to the IPOPT solver,
    so no guarantees are provided for other solvers. This is due to the fact that IPOPT,
    as an interior-point solver, guarantees that the KKT conditions at the solution are
    satisfied. This may not the case for others.

    Parameters
    ----------
    nlp : Nlp
        The NLP problem to be wrapped.
    target_parameters : casadi.SX or MX, optional
        If provided, computes the sensitibity only w.r.t. these parameters. Of course,
        these parameters must be a subset of all of the NLP's parameters. If ``None``,
        then the derivatives and sensitivity are computed w.r.t. all parameters. If new
        parameters are added after wrapping the nlp in this, be sure to call
        :meth:`set_target_parameters`` again.
    include_barrier_term : bool, optional
        If ``True``, includes in the KKT matrix a new symbolic variable that represents
        the barrier function of the interior-point solver. Otherwise, no additional
        variable is added. See property :meth:`kkt` for more details. By default,
        ``True``.

    References
    ----------
    .. [1] Buskens, C. and Maurer, H. (2001). Sensitivity analysis and real-time
        optimization of parametric nonlinear programming problems. In M. Grotschel, S.O.
        Krumke, and J. Rambau (eds.), Online Optimization of Large Scale Systems, 3–16.
        Springer, Berlin, Heidelberg.
    """

    def __init__(
        self,
        nlp: Nlp[SymType],
        target_parameters: Optional[SymType] = None,
        include_barrier_term: bool = True,
    ) -> None:
        super().__init__(nlp)
        self.include_barrier_term: bool = include_barrier_term
        self.set_target_parameters(target_parameters)
        self._tau = nlp.sym_type.sym("tau") if include_barrier_term else None

    @property
    def target_parameters(self) -> SymType:
        """Gets the parameters of the NLP that are the target of the sensitivity
        wrapper, i.e., derivatives and sensitivities are limited to these parameters."""
        return self._p_idx[0]

    @cached_property
    def lagrangian(self) -> SymType:
        """Gets the Lagrangian of the NLP problem (usually denoted as ``L``)."""
        return (
            self.nlp.f
            + cs.dot(self.nlp.lam_g, self.nlp.g)
            + cs.dot(self.nlp.lam_h, self.nlp.h)
            + cs.dot(self.nlp.lam_lbx, self.nlp.h_lbx)
            + cs.dot(self.nlp.lam_ubx, self.nlp.h_ubx)
        )

    @cached_property
    def kkt(self) -> tuple[SymType, Optional[SymType]]:
        r"""Gets the KKT conditions of the NLP problem in vector form, i.e.,

        .. math::
            K = \begin{bmatrix}
                \frac{\partial L}{\partial x} \\
                G \\
                \text{diag}(\lambda_h) \cdot H \\
            \end{bmatrix}

        where :math:`\frac{\partial L}{\partial x}` is the gradient of the lagrangian
        w.r.t. the primal variables :math:`x`, :math:`G` collects the equality
        constraints, :math:`H` collects the inequality constraints and :math:`\lambda_h`
        its corresponding dual variables.

        If ``include_barrier_term=True``, the inequalities include an additional barrier
        term ``tau``, so that

        .. math::
                            \text{diag}(\lambda_h) \cdot H + \tau

        which is also returned as the second element of the tuple. Otherwise, ``tau`` is
        ``None``.
        """
        tau = self._tau if self._tau is not None else 0
        kkt = cs.vertcat(
            cs.jacobian(self.lagrangian, self.nlp.x).T,
            self.nlp.g,
            (self.nlp.lam_h * self.nlp.h) + tau,
            (self.nlp.h_lbx * self.nlp.lam_lbx) + tau,
            (self.nlp.h_ubx * self.nlp.lam_ubx) + tau,
        )
        return kkt, self._tau

    @cached_property
    def jacobians(self) -> dict[str, SymType]:
        """Computes various partial derivatives, which are then grouped in a dict with
        the following entries

        - ``L-x``: lagrangian w.r.t. primal variables
        - ``L-p``: lagrangian w.r.t. parameters
        - ``g-x``: equality constraints w.r.t. primal variables
        - ``h-x``: inequality constraints w.r.t. primal variables
        - ``K-p``: kkt conditions w.r.t. parameters
        - ``K-y``: kkt conditions w.r.t. primal-dual variables.
        """
        L = self.lagrangian
        K = self.kkt[0]
        x = self.nlp.x
        y = self.nlp.primal_dual
        p, p_idx = self._p_idx
        return {
            "L-x": cs.jacobian(L, x).T,
            "L-p": cs.jacobian(L, p)[:, p_idx].T,
            "g-x": cs.jacobian(self.nlp.g, x),
            "h-x": cs.jacobian(self.nlp.h, x),
            "K-p": cs.jacobian(K, p)[:, p_idx],
            "K-y": cs.jacobian(K, y),
        }

    @cached_property
    def hessians(self) -> dict[str, SymType]:
        """Computes various partial hessians, which are then grouped in a dict with the
        following entries

        - ``L-pp``: lagrangian w.r.t. parameters (twice)
        - ``L-xx``: lagrangian w.r.t. primal variables (twice)
        - ``L-px``: lagrangian w.r.t. parameters and then primal variables
        """
        x = self.nlp.x
        p, p_idx = self._p_idx
        Lx = self.jacobians["L-x"]
        Lp = self.jacobians["L-p"]
        return {
            "L-xx": cs.jacobian(Lx, x),
            "L-pp": cs.jacobian(Lp, p)[:, p_idx],
            "L-px": cs.jacobian(Lp, x),
        }

    @cached_property
    def hojacobians(self) -> dict[str, np.ndarray]:
        """Computes various 3D jacobians, which are then grouped in a dict with the
        following entries

        - ``K-pp``: kkt conditions w.r.t. parameters (twice)
        - ``K-yp``: kkt conditions w.r.t. parameters and primal variables
        - ``K-yy``: kkt conditions w.r.t. primal variables (twice)
        - ``K-py``: kkt conditions w.r.t. primal variables and parameters
        """
        jacobians = self.jacobians
        Kp = jacobians["K-p"]
        Ky = jacobians["K-y"]
        y = self.nlp.primal_dual
        p, p_idx = self._p_idx
        return {
            "K-pp": hojacobian(Kp, p)[..., p_idx, 0],
            "K-yp": hojacobian(Ky, p)[..., p_idx, 0],
            "K-yy": hojacobian(Ky, y)[..., 0],
            "K-py": hojacobian(Kp, y)[..., 0],
        }

    @property
    def licq(self) -> SymType:
        r"""Gets the symbolic matrix for the LICQ, defined as

        .. math::
            LICQ = \begin{bmatrix}
                \frac{\partial g}{\partial x} \\
                \frac{\partial h}{\partial x} \\
                \frac{\partial h_{lbx}}{\partial x} \\
                \frac{\partial h_{ubx}}{\partial x}
            \end{bmatrix}

        If the matrix is linear independent, then the NLP satisfies the Linear
        Independence Constraint Qualification.

        Notes
        -----
        Theoretically, the LICQ is computed for only the active inenquality constraints.
        Since this is merely a symbolic representation, all constraints are included,
        and it's up to the user to eliminate the inactive ones.
        """
        return cs.vertcat(
            self.jacobians["g-x"],
            self.jacobians["h-x"],
            cs.jacobian(self.nlp.h_lbx, self.nlp.x),
            cs.jacobian(self.nlp.h_ubx, self.nlp.x),
        )

    def parametric_sensitivity(
        self,
        expr: SymType = None,
        solution: Optional[Solution[SymType]] = None,
        second_order: bool = False,
    ) -> Union[tuple[SymType, Optional[SymType]], tuple[cs.DM, Optional[cs.DM]]]:
        r"""Performs the (symbolic or numerical) sensitivity of the NLP w.r.t. its
        parametrization, according to [1]_.

        Parameters
        ----------
        expr : casadi.SX or MX, optional
            If provided, computes the sensitivity of this expression (which must be
            dependent on the primal-dual variables and/or parameters of the NLP) w.r.t.
            the NLP parameters. If ``None``, then the sensitivity of the primal-dual
            variables is returned.
        solution : Solution, optional
            If a solution is passed, then the sensitivity is numerically computed for
            that solution; otherwise, the sensitivity is carried out symbolically
            (however, this is much more computationally intensive).
        second_order : bool, optional
            If ``second_order=False``, the analysis is stopped at the first order;
            otherwise, also second-order information is computed.

        Returns
        -------
        2-element tuple of casadi.SX, or MX, or DM
            The 1st and 2nd-order NLP parametric sensitivity in the form of an array/DM,
            if a solution is passed, or a symbolic vector SX/MX. When
            ``second_order=False`` the second element in the tuple is ``None``.

        Raises
        ------
        numpy.linalg.LinAlgError
            Raises if the KKT conditions lead to a singular matrix.
        """
        # first and second order sensitivities, a.k.a., dydp and d2dydp2
        d: Callable[[SymType], Union[SymType, cs.DM]] = (
            (lambda o: o)
            if solution is None
            else (lambda o: solution.value(o))  # type: ignore[union-attr]
        )
        dydp, dydp_np, d2ydp2 = self._y_parametric_sensitivity(
            solution, second_order, d
        )

        # first order sensitivity of Z, a.k.a., dZdp
        Z = expr  # Z := z(x(p),lam(p),p)
        if Z is not None:
            Zshape = Z.shape
            Z = cs.vec(Z)
            y = self.nlp.primal_dual
            p, p_idx = self._p_idx
            np_ = p.shape[0] if isinstance(p_idx, slice) else p_idx.size

            if second_order:
                Zpp, Zp = hohessian(Z, p)
                Zyy, Zy = hohessian(Z, y)
            else:
                Zp = hojacobian(Z, p)
                Zy = hojacobian(Z, y)
            Zp = d(array2cs(Zp[:, 0, p_idx, 0]))
            Zy = d(array2cs(Zy[:, 0, :, 0]))

            dZdp = Zy @ dydp + Zp
            dZdp = cs2array(dZdp).reshape(Zshape + (np_,), order="F").squeeze()
            if dZdp.ndim <= 2:
                dZdp = array2cs(dZdp)

        # if no second order is required, just return
        if not second_order:
            return dydp if Z is None else dZdp, None

        # if no expression is given, just return
        if Z is None:
            d2ydp2 = d2ydp2.squeeze()
            if d2ydp2.ndim <= 2:
                d2ydp2 = array2cs(d2ydp2)
            return dydp, d2ydp2

        # second order sensitivity of Z, a.k.a., d2Zdp2
        Zyp = d(hohessian(Z, y, p)[0][:, 0, :, 0][:, :, p_idx, 0])
        Zpy = d(hohessian(Z, p, y)[0][:, 0, p_idx, 0][:, :, :, 0])
        Zpp = d(Zpp[:, 0, p_idx, 0][:, :, p_idx, 0])
        Zyy = d(Zyy[:, 0, :, 0][:, :, :, 0])
        T1 = (d2ydp2.transpose((1, 2, 0)) @ cs2array(Zy).T).transpose((2, 0, 1))
        T2 = Zyy.transpose((0, 2, 1)) @ dydp_np + Zpy.transpose((0, 2, 1)) + Zyp
        d2Zdp2 = Zpp + T1 + dydp_np.T @ T2
        d2Zdp2 = d2Zdp2.reshape(Zshape + (np_, np_), order="F").squeeze()
        if d2Zdp2.ndim <= 2:
            d2Zdp2 = array2cs(d2Zdp2)
        return dZdp, d2Zdp2

    def _y_parametric_sensitivity(
        self,
        solution: Optional[Solution[SymType]],
        second_order: bool,
        d: Callable[[SymType], Union[SymType, cs.DM]],
    ) -> Union[tuple[SymType, np.ndarray, SymType], tuple[cs.DM, np.ndarray, cs.DM]]:
        """Internal utility to compute the sensitivity of ``y`` w.r.t. ``p``."""
        # first order sensitivity, a.k.a., dydp
        Ky = d(self.jacobians["K-y"])
        Kp = d(self.jacobians["K-p"])
        dydp = (cs.solve if solution is None else np.linalg.solve)(-Ky, Kp)
        if not second_order:
            return dydp, None, None  # type: ignore[return-value]

        # second order sensitivity, a.k.a., d2ydp2
        dydp_ = cs2array(dydp)
        Kpp = d(self.hojacobians["K-pp"])
        Kpy = d(self.hojacobians["K-py"])
        Kyp = d(self.hojacobians["K-yp"])
        Kyy = d(self.hojacobians["K-yy"])
        M = (
            Kpp
            + (Kpy.transpose((0, 2, 1)) + Kyp + (Kyy @ dydp_)).transpose((0, 2, 1))
            @ dydp_
        ).transpose((2, 0, 1))
        if solution is None:
            d2ydp2 = -(cs2array(cs.inv(Ky)) @ M).transpose((1, 2, 0))
        else:
            d2ydp2 = -np.linalg.solve(Ky, M).transpose((1, 2, 0))
        return dydp, dydp_, d2ydp2

    @invalidate_cache(jacobians, hessians, hojacobians)
    def parameter(self, name: str, shape: tuple[int, int] = (1, 1)) -> SymType:
        """See :meth:`csnlp.Nlp.parameter`."""
        return self.nlp.parameter(name, shape)

    @invalidate_cache(lagrangian, kkt, jacobians, hessians, hojacobians)
    def variable(
        self,
        name: str,
        shape: tuple[int, int] = (1, 1),
        lb: Union[npt.ArrayLike, cs.DM] = -np.inf,
        ub: Union[npt.ArrayLike, cs.DM] = +np.inf,
    ) -> tuple[SymType, SymType, SymType]:
        """See :meth:`csnlp.Nlp.variable`."""
        return self.nlp.variable(name, shape, lb, ub)

    @invalidate_cache(lagrangian, kkt, jacobians, hessians, hojacobians)
    def constraint(
        self,
        name: str,
        lhs: Union[SymType, np.ndarray, cs.DM],
        op: Literal["==", ">=", "<="],
        rhs: Union[SymType, np.ndarray, cs.DM],
        soft: bool = False,
        simplify: bool = True,
    ) -> tuple[SymType, ...]:
        """See :meth:`csnlp.Nlp.constraint`."""
        return self.nlp.constraint(name, lhs, op, rhs, soft, simplify)

    @invalidate_cache(lagrangian, kkt, jacobians, hessians, hojacobians)
    def minimize(self, objective: SymType) -> None:
        """See :meth:`csnlp.Nlp.minimize`."""
        return self.nlp.minimize(objective)

    @invalidate_cache(jacobians, hessians, hojacobians)
    def set_target_parameters(self, parameters: Optional[SymType]) -> None:
        """Sets the target parameters of the sensitivity wrapper.

        Parameters
        ----------
        parameters : casadi.MX or SX or None
            New parameters to target during the sensitivity analyses. If ``None``, all
            NLP parameters are included.
        """
        if parameters is None:
            self._p_idx_internal = None, slice(None)
            return
        p: SymType = cs.vec(parameters)
        if self.nlp.sym_type is cs.SX:
            self._p_idx_internal = p, slice(None)
            return
        self._p_idx_internal = self.nlp.p, find_index_in_vector(self.nlp.p, p)

    @property
    def _p_idx(self) -> tuple[SymType, Union[slice, npt.NDArray[np.int64]]]:
        """Internal utility to return the indices of ``p`` from all the NLP pars. While
        SX is fine with computing jacobians with indexed variables, MX requires purely
        symbolic variables. So, for MX, jacobians need to be computed for all elements
        and then indexed."""
        p = self._p_idx_internal[0]
        return (self.nlp.p if p is None else p), self._p_idx_internal[1]
