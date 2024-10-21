from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, Optional, TypeVar, Union

import casadi as cs
import numpy as np
import numpy.typing as npt

from ...core.data import find_index_in_vector
from .mpc import Mpc

SymType = TypeVar("SymType", cs.SX, cs.MX)


@dataclass
class PwaRegion:
    """Stores the matrices defining the i-th region in a piecewise affine system."""

    A: npt.NDArray[np.floating]
    r"""The state matrix :math:`A_i` of the linear dynamics
    :math:`x_+ = A_i x + B_i u + c_i`."""

    B: npt.NDArray[np.floating]
    r"""The input matrix :math:`B_i` of the linear dynamics
    :math:`x_+ = A_i x + B_i u + c_i`."""

    c: npt.NDArray[np.floating]
    r"""The affine constant vector :math:`c_i` of the linear dynamics
    :math:`x_+ = A_i x + B_i u + c_i`."""

    S: npt.NDArray[np.floating]
    r"""The state matrix :math:`S_i` of the region inequality
    :math:`S_i x + R_i u \leq T_i`"""

    R: npt.NDArray[np.floating]
    r"""The input matrix :math:`R_i` of the region inequality
    :math:`S_i x + R_i u \leq T_i`"""

    T: npt.NDArray[np.floating]
    r"""The constant vector :math:`T_i` of the region inequality
    :math:`S_i x + R_i u \leq T_i`"""


class PwaMpc(Mpc[SymType]):
    r"""MPC controller for piecewise affine (PWA) systems. A PWA system is characterized
    by linear dynamics that switch between different regions of the state-action space.
    In mathematical terms, given a PWA system with :math:`s` regions, the dynamics are

    .. math::
        x_+ = \begin{cases}
            A_1 x + B_1 u + c_1 & \text{if } S_1 x + R_1 u \leq T_1 \\
            & \vdots \\
            A_i x + B_i u + c_i & \text{if } S_i x + R_i u \leq T_i \\
            & \vdots \\
            A_s x + B_s u + c_s & \text{if } S_s x + R_s u \leq T_s
        \end{cases}

    Following :cite:`bemporad_control_1999`, the PWA dynamics can be converted to
    mixed-logical dynamical form, and the ensuing MPC optimization becomes a
    mixed-integer optimization problem. This is done under the hood via the
    :meth:`set_pwa_dynamics` method. See also :cite:`borrelli_predictive_2017` for
    further details.

    Parameters
    ----------
    nlp : Nlp
        NLP scheme to be wrapped
    prediction_horizon : int
        A positive integer for the prediction horizon of the MPC controller.
    control_horizon : int, optional
        A positive integer for the control horizon of the MPC controller. If not given,
        it is set equal to the control horizon.
    input_spacing : int, optional
        Spacing between independent input actions. This argument allows to reduce the
        number of free actions along the control horizon by allowing only the first
        action every ``n`` to be free, and the following ``n-1`` to be fixed equal to
        that action (where ``n`` is given by ``input_spacing``). By default, no spacing
        is allowed, i.e., ``1``.
    shooting : 'single' or 'multi', optional
        Type of approach in the direct shooting for parametrizing the control
        trajectory. See Section 8.5 in :cite:`rawlings_model_2017`. By default, direct
        shooting is used.

    Raises
    ------
    ValueError
        Raises if the shooting method is invalid; or if any of the horizons are invalid;
        or if the number of scenarios is not a positive integer."""

    def set_pwa_dynamics(
        self,
        pwa_system: Sequence[PwaRegion],
        D: Union[npt.NDArray[np.floating], cs.DM],
        E: npt.NDArray[np.floating],
        clp_opts: Optional[dict[str, Any]] = None,
        parallelization: Literal[
            "serial", "unroll", "inline", "thread", "openmp"
        ] = "thread",
        max_num_threads: Optional[int] = None,
    ) -> None:
        r"""Sets the piecewise affine dynamics of the system for the MPC controller,
        creating auxiliary variables and constraints to handle the PWA switching. In
        order to perform the conversion of the PWA dynamics to mixed-logical dynamical
        form, the method solves a series of linear programmes via the ``CLP`` solver.
        Parallelization can also be enabled to speed up the process.

        Parameters
        ----------
        pwa_system : collection of PwaRegion
            A sequence of :class:`PwaRegion` objects, where the i-th object contains
            the matrices defining the i-th region of the PWA system.
        D : array or casadi.DM of shape (n_ineq, ns + na)
            The (possibly sparse) matrix ``D`` defining the polytopic constraints on the
            state-action space :math:`D [x^\top, u^\top]^\top \leq E`, where ``ns`` and
            ``na`` are the numbers of states and actions in the MPC problem,
            respectively.
        E : array of shape (n_ineq,)
            The matrix ``E`` defining the polytopic constraints on the state-action
            space :math:`D [x, u]^\top \leq E`.
        clp_opts : dict, optional
            Options for the CLP solver. Defaults to ``None``.
        parallelization : "serial", "unroll", "inline", "thread", "openmp"
            The type of parallelization to use (see :func:`casadi.Function.map`) when
            solving the linear programmes. By default, ``"thread"`` is selected.
        max_num_threads : int, optional
            Maximum number of threads to use in parallelization; if ``None``, the number
            of threads is equal to the number of regions in the system.

        Raises
        ------
        RuntimeError
            Raises if the dynamics were already set, or if lower and upper bounds on the
            states and/or actions are set.
        ValueError
            Raises if the dimensions of any matrix in any region do not match the
            expected shape.

        Notes
        -----
        This function will raise an error if lower and upper bounds on the states and
        actions are set. This is because these bounds should be instead specified via
        the matrices ``D`` and ``E``. Moreover, this function only uses these matrices
        for internal computations. The user should take care to impose the constraints
        :math::math:`D [x^\top, u^\top]^\top \leq E` in the optimization problem, as
        well as any other, via the :meth:`constraint` method.
        """
        if self._dynamics is not None:
            raise RuntimeError("Dynamics were already set.")

        # retrieve lower and upper bounds on the states and actions and check they are
        # all unset, i.e., -/+ inf
        U = cs.vvcat(self._actions_exp.values())
        idx = find_index_in_vector(self.x, U)
        lbu = self.lbx[idx].data
        ubu = self.ubx[idx].data
        if (~np.isneginf(lbu)).any() or (~np.isposinf(ubu)).any():
            raise RuntimeError(
                "cannot set lower and upper bounds on the actions in PWA systems; use "
                "arguments `D` and `E` of `set_pwa_dyanmics` instead"
            )
        if self._is_multishooting:
            X = cs.vvcat(self._states.values())
            idx = find_index_in_vector(self.x, X)
            lbx = self.lbx[idx].data
            ubx = self.ubx[idx].data
            if (~np.isneginf(lbx)).any() or (~np.isposinf(ubx)).any():
                raise RuntimeError(
                    "cannot set lower and upper bounds on the states in PWA systems; "
                    "use arguments `D` and `E` of `set_pwa_dyanmics` instead"
                )

        # validate dimensions
        ns = self.ns
        na = self.na
        n_ineq = pwa_system[0].T.shape[0]  # must be the same for all regions
        for i, region in enumerate(pwa_system):
            if region.A.shape != (ns, ns):
                raise ValueError(f"A in region {i} must have shape ({ns}, {ns}).")
            if region.B.shape != (ns, na):
                raise ValueError(f"B in region {i} must have shape ({ns}, {na}).")
            if region.c.shape != (ns,):
                raise ValueError(f"c in region {i} must have shape ({ns},).")
            if region.S.shape != (n_ineq, ns):
                raise ValueError(f"S in region {i} must have shape ({n_ineq}, {ns}).")
            if region.R.shape != (n_ineq, na):
                raise ValueError(f"R in region {i} must have shape ({n_ineq}, {na}).")
            if region.T.shape != (n_ineq,):
                raise ValueError(f"T in region {i} must have shape ({n_ineq},).")
        n_ineq = E.shape[0]
        if D.shape != (n_ineq, ns + na):
            raise ValueError(f"D must have shape ({n_ineq}, {ns + na}).")
        if E.shape != (n_ineq,):
            raise ValueError(f"E must have shape ({n_ineq},).")

        # set the PWA dynamics
        if max_num_threads is None:
            max_num_threads = len(pwa_system)
        if clp_opts is None:
            clp_opts = {}
        self._set_pwa_dynamics(
            pwa_system, D, E, clp_opts, parallelization, max_num_threads
        )
        self._dynamics = object()  # TODO New dynamics will just be a flag

    def _set_pwa_dynamics(
        self,
        regions: Sequence[PwaRegion],
        D: Union[npt.NDArray[np.floating], cs.DM],
        E: npt.NDArray[np.floating],
        clp_opts: dict[str, Any],
        parallelization: Literal["serial", "unroll", "inline", "thread", "openmp"],
        max_num_threads: int,
    ) -> None:
        """Internal utility to create PWA dynamics constraints."""
        nr = len(regions)
        n_ineq = regions[0].T.size
        ns, na = regions[0].B.shape
        N = self._prediction_horizon

        # solve linear programs to determine bounds for big-M relaxations. These LPs are
        # solved parallelly for each region and for each inequality defining the region.
        D = cs.sparsify(D)  # can be sparse
        SR_, T_, AB_, C_ = [], [], [], []
        for r in regions:
            SR_.append(np.hstack((r.S, r.R)))
            T_.append(r.T)
            AB_.append(np.hstack((r.A, r.B)))
            C_.append(r.c)
        SR = np.vstack(SR_).T
        T = np.asarray(T_)
        AB = np.vstack(AB_).T
        C = np.asarray(C_)
        lp = {"h": cs.Sparsity(ns + na, ns + na), "a": D.sparsity()}
        lpsolver = cs.conic("lpsolver", "clp", lp, clp_opts)

        mapped_lpsolver = lpsolver.map(nr * n_ineq, parallelization, max_num_threads)
        sol = mapped_lpsolver(g=SR, a=D, uba=E)
        big_M = -sol["cost"].toarray().reshape(nr, n_ineq) - T

        mapped_lpsolver = lpsolver.map(nr * ns, parallelization, max_num_threads)
        sol = mapped_lpsolver(g=AB, a=D, uba=E)
        tmp_lb = sol["cost"].toarray().reshape(nr, ns) + C
        M_lb = tmp_lb.min(0)
        sol = mapped_lpsolver(g=-AB, a=D, uba=E)
        tmp_ub = -sol["cost"].toarray().reshape(nr, ns) + C
        M_ub = tmp_ub.max(0)

        # dynamics constraints - we now have to add constraints for all regions at each
        # time-step, with the binary variable delta selecting the active region
        z = [self.variable(f"z_{i}", (ns, N))[0] for i in range(nr)]
        if self._is_multishooting:
            X = cs.vcat(self._states.values())
            self.constraint("dynamics", X[:, 1:], "==", sum(z))
        else:
            Xk = cs.vcat(self._initial_states.values())
            X = cs.horzcat(Xk, sum(z))
            cumsizes = np.cumsum(
                [0] + [s.shape[0] for s in self._initial_states.values()]
            )
            self._states = dict(zip(self._states.keys(), cs.vertsplit(X, cumsizes)))

        U = cs.vcat(self._actions_exp.values())
        delta, _, _ = self.variable("delta", (nr, N), lb=0, ub=1, discrete=True)
        self.constraint("delta_sum", cs.sum1(delta), "==", 1)
        z_ub = []
        z_lb = []
        region = []
        z_x_ub = []
        z_x_lb = []
        X_ = X[:, :-1]
        for i, r in enumerate(regions):
            z_i = z[i]
            delta_i = delta[i, :]
            z_ub.append(z_i - M_ub @ delta_i)
            z_lb.append(z_i - M_lb @ delta_i)
            region.append(r.S @ X_ + r.R @ U - r.T - big_M[i, :] @ (1 - delta_i))
            z_x_ub.append(z_i - (r.A @ X_ + r.B @ U + r.c - M_lb @ (1 - delta_i)))
            z_x_lb.append(z_i - (r.A @ X_ + r.B @ U + r.c - M_ub @ (1 - delta_i)))
        self.constraint("z_ub", cs.vcat(z_ub), "<=", 0)
        self.constraint("z_lb", cs.vcat(z_lb), ">=", 0)
        self.constraint("region", cs.vcat(region), "<=", 0)
        self.constraint("z_x_ub", cs.vcat(z_x_ub), "<=", 0)
        self.constraint("z_x_lb", cs.vcat(z_x_lb), ">=", 0)
