from collections.abc import Collection, Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Iterable, Literal, Optional, TypeVar, Union

import casadi as cs
import numpy as np
import numpy.typing as npt

from csnlp.core.solutions import Solution

from ...core.data import find_index_in_vector
from .mpc import Mpc

SymType = TypeVar("SymType", cs.SX, cs.MX)


def _n(parname: str, index: int) -> str:
    """Internal utility for the naming convention of the ``i``-th region parameters."""
    return f"{parname}__{index}"


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
    :math:`S_i [x^\top, u^\top]^\top \leq T_i`"""

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

    This MPC controller class can then handle two different types of specifications for
    the dynamics.

    * **time-varying affine dynamics**: the MPC controller can be considered as a
      linear MPC controller with time-varying dynamics, in which case the dynamics
      must be defined via the :meth:`set_time_varying_affine_dynamics` method. Then,
      prior to solving the optimization problem, the sequence of regions to be active
      at each time-step needs to be set via :meth:`set_switching_sequence` by the
      user/externally.
    * **optimzing also the sequence**: alternatively, the sequence of PWA regions can
      be optimized over, in which case the dynamics must be defined via the
      :meth:`set_pwa_dynamics` method. Following :cite:`bemporad_control_1999`, the PWA
      dynamics are converted to mixed-logical dynamical form, and the ensuing MPC
      optimization becomes a mixed-integer optimization problem. This is done under the
      hood via the :meth:`set_pwa_dynamics` method. See also
      :cite:`borrelli_predictive_2017` for further details.

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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._fixed_sequence_dynamics = False
        self._pwa_system: Optional[Sequence[PwaRegion]] = None
        self._sequence: Optional[Collection[int]] = None

    def validate_pwa_dimensions(self, pwa_system: Iterable[PwaRegion]) -> None:
        """Validates that the dimensions are correct for all matrices in
        the passed PWA system.

        Parameters
        ----------
        pwa_system : iterable of PwaRegion
            An iterable of :class:`PwaRegion` objects, where the i-th object contains
            the matrices defining the i-th region of the PWA system.

        Raises
        ------
        ValueError
            Raises if the dimensions of any matrix in any region do not match the
            expected shape.
        """
        ns = self.ns
        na = self.na
        nsa = ns + na
        n_ineq = pwa_system[0].T.shape[0]  # must be the same for all regions
        for i, region in enumerate(pwa_system):
            if region.A.shape != (ns, ns):
                raise ValueError(f"A in region {i} must have shape ({ns}, {ns}).")
            if region.B.shape != (ns, na):
                raise ValueError(f"B in region {i} must have shape ({ns}, {na}).")
            if region.c.shape != (ns,):
                raise ValueError(f"c in region {i} must have shape ({ns},).")
            if region.S.shape != (n_ineq, nsa):
                raise ValueError(f"S in region {i} must have shape ({n_ineq}, {nsa}).")
            if region.T.shape != (n_ineq,):
                raise ValueError(f"T in region {i} must have shape ({n_ineq},).")

    def set_pwa_dynamics(
        self,
        pwa_system: Collection[PwaRegion],
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
            A collection of :class:`PwaRegion` objects, where the i-th object contains
            the matrices defining the i-th region of the PWA system.
        D : array or casadi.DM of shape (n_ineq, ns + na)
            The (possibly sparse) matrix ``D`` defining the polytopic constraints on the
            state-action space :math:`D [x^\top, u^\top]^\top \leq E`, where ``ns`` and
            ``na`` are the numbers of states and actions in the MPC problem,
            respectively.
        E : array of shape (n_ineq,)
            The matrix ``E`` defining the polytopic constraints on the state-action
            space :math:`D [x^\top, u^\top]^\top \leq E`.
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
        When multiple states and/or control inputs are defined, these are concatenated
        into the single vectors :math:`x` and :math:`u`, respectively. Also, this
        function will raise an error if lower and upper bounds on any state or action
        are set. This is because these bounds should be instead specified via the
        matrices ``D`` and ``E``. Moreover, this function only uses these matrices for
        internal computations, so the user should take care to impose the constraints
        :math:`D [x^\top, u^\top]^\top \leq E` in the optimization problem, as well as
        any other, via the :meth:`constraint` method.
        """
        if self._dynamics_already_set:
            raise RuntimeError("Dynamics were already set.")

        # retrieve lower and upper bounds on the states and actions and check they are
        # all unset, i.e., -/+ inf
        U = cs.vvcat(self._actions_exp.values())
        idx = find_index_in_vector(self.x, U)
        lbu = self.lbx[idx].data
        ubu = self.ubx[idx].data
        if (~np.isneginf(lbu)).any() or (~np.isposinf(ubu)).any():
            raise RuntimeError(
                "Cannot set lower and upper bounds on the actions in PWA systems; use "
                "arguments `D` and `E` of `set_pwa_dyanmics` instead."
            )
        if self._is_multishooting:
            X = cs.vvcat(self._states.values())
            idx = find_index_in_vector(self.x, X)
            lbx = self.lbx[idx].data
            ubx = self.ubx[idx].data
            if (~np.isneginf(lbx)).any() or (~np.isposinf(ubx)).any():
                raise RuntimeError(
                    "Cannot set lower and upper bounds on the states in PWA systems; "
                    "use arguments `D` and `E` of `set_pwa_dyanmics` instead."
                )

        # validate dimensions
        self.validate_pwa_dimensions(pwa_system)
        nsa = self.ns + self.na
        n_ineq = E.shape[0]
        if D.shape != (n_ineq, nsa):
            raise ValueError(f"D must have shape ({n_ineq}, {nsa}).")
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
        self._dynamics_already_set = True

    def _set_pwa_dynamics(
        self,
        regions: Collection[PwaRegion],
        D: Union[npt.NDArray[np.floating], cs.DM],
        E: npt.NDArray[np.floating],
        clp_opts: dict[str, Any],
        parallelization: Literal["serial", "unroll", "inline", "thread", "openmp"],
        max_num_threads: int,
    ) -> None:
        """Internal utility to create PWA dynamics constraints."""
        # solve linear programs to determine bounds for big-M relaxations. These LPs are
        # solved parallelly for each region and for each inequality defining the region.
        D = cs.sparsify(D)  # can be sparse
        S_, T_, AB_, C_ = [], [], [], []
        for r in regions:
            S_.append(r.S)
            T_.append(r.T)
            AB_.append(np.hstack((r.A, r.B)))
            C_.append(r.c)
        S = np.vstack(S_).T
        T = np.asarray(T_)
        AB = np.vstack(AB_).T
        C = np.asarray(C_)

        nr = len(regions)
        n_ineq = r.T.size
        ns, na = r.B.shape
        nsa = ns + na
        N = self._prediction_horizon

        lp = {"h": cs.Sparsity(nsa, nsa), "a": D.sparsity()}
        lpsolver = cs.conic("lpsolver", "clp", lp, clp_opts)

        mapped_lpsolver = lpsolver.map(nr * n_ineq, parallelization, max_num_threads)
        sol = mapped_lpsolver(g=S, a=D, uba=E)
        big_M = -sol["cost"].toarray().reshape(nr, n_ineq) + T

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
        XU = cs.vertcat(X[:, :-1], U)
        for i, r in enumerate(regions):
            z_i = z[i]
            delta_i = delta[i, :]
            AB_i = AB_[i]
            z_ub.append(z_i - M_ub @ delta_i)
            z_lb.append(z_i - M_lb @ delta_i)
            region.append(r.S @ XU - r.T - big_M[i, :] @ (1 - delta_i))
            z_x_ub.append(z_i - (AB_i @ XU + r.c - M_lb @ (1 - delta_i)))
            z_x_lb.append(z_i - (AB_i @ XU + r.c - M_ub @ (1 - delta_i)))
        self.constraint("z_ub", cs.vcat(z_ub), "<=", 0)
        self.constraint("z_lb", cs.vcat(z_lb), ">=", 0)
        self.constraint("region", cs.vcat(region), "<=", 0)
        self.constraint("z_x_ub", cs.vcat(z_x_ub), "<=", 0)
        self.constraint("z_x_lb", cs.vcat(z_x_lb), ">=", 0)

    def set_time_varying_affine_dynamics(self, pwa_system: Sequence[PwaRegion]) -> None:
        r"""Sets the time-varying affine dynamics of the system for the MPC controller.
        The possible values taken by the affine dynamics are defined by the sequence of
        :class:`PwaRegion`.

        Parameters
        ----------
        pwa_system : sequence of PwaRegion
            A sequence of :class:`PwaRegion` objects, where the i-th object contains
            the matrices defining the i-th region of the PWA system.

        Raises
        ------
        RuntimeError
            Raises if the dynamics were already set.
        ValueError
            Raises if the dimensions of any matrix in any region do not match the
            expected shape.

        """
        if self._dynamics_already_set:
            raise RuntimeError("Dynamics were already set.")

        self.validate_pwa_dimensions(pwa_system)
        N = self._prediction_horizon
        ns = self.ns
        na = self.na
        nsa = ns + na
        n_ineq = pwa_system[0].T.size

        # parameters defining time-varying dynamics
        A = [self.parameter(f"A[{k}]", (ns, ns)) for k in range(N)]
        B = [self.parameter(f"B[{k}]", (ns, na)) for k in range(N)]
        c = [self.parameter(f"c[{k}]", (ns, 1)) for k in range(N)]
        S = [self.parameter(f"S[{k}]", (n_ineq, nsa)) for k in range(N)]
        T = [self.parameter(f"T[{k}]", (n_ineq, 1)) for k in range(N)]

        X = cs.vcat(self._states.values())
        U = cs.vcat(self._actions_exp.values())

        # set dynamics constraints and region constraints
        if not self._is_multishooting:
            raise NotImplementedError("Single shooting not implemented yet.")
        xs_next = []
        for k in range(N):
            x_next = A[k] @ X[:, k] + B[k] @ U[:, k] + c[k]
            xs_next.append(x_next)
        self.constraint(
            "region",
            cs.diagcat(*S)
            @ cs.vertcat(*[cs.vertcat(X[:, k], U[:, k]) for k in range(N)])
            - cs.vertcat(*T),
            "<=",
            0,
        )
        self.constraint("dyn", cs.hcat(xs_next), "==", X[:, 1:])

        self._pwa_system = pwa_system
        self._dynamics_already_set = True
        self._fixed_sequence_dynamics = True

    def set_switching_sequence(self, sequence: Collection[int]) -> None:
        """Sets the sequence of regions to be active at each time step along the MPC
        prediction horizon. Then, when solved, the MPC optimization will only optimze
        over the action and state trajectories, while enforcing the sequence of regions
        visited by the states to be the one provided here.

        Parameters
        ----------
        sequence : collection of int
            A collection of integers representing the indices of the regions to be
            active at each time step along the prediction horizon, i.e., the k-th entry
            of this collection represents the index of the region the state must be at
            time step ``k``.

        Raises
        ------
        ValueError
            Raises if dynamics have not been set via
            :meth:`set_time_varying_affine_dynamics`; if the sequence is not the same
            length as the prediction horizon; if the sequence does not contain integers;
            if the sequence contains integers that exceed the number of PWA regions
            specified via :meth:`set_time_varying_affine_dynamics`.

        Notes
        -----
        For internal validation purposes, please call first
        :meth:`set_time_varying_affine_dynamics` and only then call this method.
        """
        if not self._fixed_sequence_dynamics or self._pwa_system is None:
            raise ValueError(
                "Sequence can only be set if time-varying dynamics are used."
            )
        if len(sequence) != self._prediction_horizon:
            raise ValueError("Length of sequence must match the prediction horizon")

        N = len(self._pwa_system)
        for i, idx in enumerate(sequence):
            if not isinstance(idx, Integral):
                raise ValueError(
                    f"{i}-th element of the sequence is not an integer; got {idx}."
                )
            if not 0 <= idx < N:
                raise ValueError(
                    f"{i}-th element of the sequence must be in [0, N); got {idx}."
                )
        self._sequence = sequence

    def solve(
        self,
        pars: Optional[dict[str, npt.ArrayLike]] = None,
        vals0: Optional[dict[str, npt.ArrayLike]] = None,
    ) -> Solution[SymType]:
        if self._fixed_sequence_dynamics:
            regions = self._pwa_system
            assert regions is not None, "PWA system should have been set!"
            if self._sequence is None:
                raise ValueError(
                    "A sequence must be set via `set_switching_sequence` prior to "
                    "solving the MPC because the dyanmics were set via "
                    "`set_time_varying_affine_dynamics`. Use `set_pwa_dynamics` instead"
                    " to optimize over the sequence as well."
                )

            if pars is None:
                pars = {}
            for k, idx in enumerate(self._sequence):
                pars[_n("A", k)] = regions[idx].A
                pars[_n("B", k)] = regions[idx].B
                pars[_n("c", k)] = regions[idx].c
                pars[_n("S", k)] = regions[idx].S
                pars[_n("T", k)] = regions[idx].T
        return self.nlp.solve(pars, vals0)
