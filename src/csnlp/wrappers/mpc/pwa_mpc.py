from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, Optional, TypeVar

import casadi as cs
import numpy as np
import numpy.typing as npt

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

    def state(
        self,
        name: str,
        size: int = 1,
        discrete: bool = False,
        bound_initial: bool = True,
        bound_terminal: bool = True,
    ) -> tuple[Optional[SymType], SymType]:
        """Adds a state variable to the MPC controller along the whole prediction
        horizon. Automatically creates the constraint on the initial conditions for this
        state. Note that lower and upper bounds cannot be specified here; specify them
        instead as the polytopic state constraint :math:`D x \leq E` in
        :meth:`set_pwa_dynamics`.

        Parameters
        ----------
        name : str
            Name of the state.
        size : int
            Size of the state (assumed to be a vector).
        discrete : bool, optional
            Flag indicating if the state is discrete. Defaults to ``False``.
        bound_initial : bool, optional
            If ``False``, then the upper and lower bounds on the initial state are not
            imposed, i.e., set to ``+/- np.inf`` (since the initial state is constrained
            to be equal to the current state of the system, it is sometimes advantageous
            to remove its bounds). By default ``True``.
        bound_terminal : bool, optional
            Same as above, but for the terminal state. By default ``True``.

        Returns
        -------
        state : casadi.SX or MX or None
            The state symbolic variable. If ``shooting=single``, then ``None`` is
            returned since the state will only be available once the dynamics are set.
        initial state : casadi.SX or MX
            The initial state symbolic parameter.

        Raises
        ------
        ValueError
            Raises if there exists already a state with the same name.
        RuntimeError
            Raises in single shooting if lower or upper bounds have been specified,
            since these can only be set after the dynamics have been set via the
            :meth:`constraint` method.
        """
        return super().state(
            name,
            size,
            discrete,
            bound_initial=bound_initial,
            bound_terminal=bound_terminal,
        )

    def action(
        self, name: str, size: int = 1, discrete: bool = False
    ) -> tuple[SymType, SymType]:
        """Adds a control action variable to the MPC controller along the whole control
        horizon. Automatically expands this action to be of the same length of the
        prediction horizon by padding with the final action. Note that lower and upper
        bounds cannot be specified here; specify them instead as the polytopic action
        constraint :math:`F u \leq G` in :meth:`set_pwa_dynamics`.

        Parameters
        ----------
        name : str
            Name of the control action.
        size : int, optional
            Size of the control action (assumed to be a vector). Defaults to ``1``.
        discrete : bool, optional
            Flag indicating if the action is discrete. Defaults to ``False``.

        Returns
        -------
        action : casadi.SX or MX
            The control action symbolic variable.
        action_expanded : casadi.SX or MX
            The same control  action variable, but expanded to the same length of the
            prediction horizon.
        """
        return super().action(name, size, discrete)

    def set_pwa_dynamics(
        self,
        pwa_system: Sequence[PwaRegion],
        D: npt.NDArray[np.floating],
        E: npt.NDArray[np.floating],
        F: npt.NDArray[np.floating],
        G: npt.NDArray[np.floating],
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
        D : array of shape (n_ineq_x, ns)
            The matrix defining the polytopic constraints on the state space
            :math:`D x \leq E`.
        E : array of shape (n_ineq_x,)
            The vector defining the polytopic constraints on the state space
            :math:`D x \leq E`.
        F : array of shape (n_ineq_u, na)
            The matrix defining the polytopic constraints on the input space
            :math:`F u \leq G`.
        G : array of shape (n_ineq_u,)
            The vector defining the polytopic constraints on the input space
            :math:`F u \leq G`.
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
            Raises if the dynamics were already set.
        ValueError
            Raises if the dimensions of any matrix in any region do not match the
            expected shape.
        """
        if self._dynamics is not None:
            raise RuntimeError("Dynamics were already set.")

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
        n_ineq_x = E.shape[0]
        if D.shape != (n_ineq_x, ns):
            raise ValueError(f"D must have shape ({n_ineq_x}, {ns}).")
        if E.shape != (n_ineq_x,):
            raise ValueError(f"E must have shape ({n_ineq_x},).")
        n_ineq_u = G.shape[0]
        if F.shape != (n_ineq_u, na):
            raise ValueError(f"F must have shape ({n_ineq_u}, {na}).")
        if G.shape != (n_ineq_u,):
            raise ValueError(f"G must have shape ({n_ineq_u},).")

        # set the PWA dynamics
        if max_num_threads is None:
            max_num_threads = len(pwa_system)
        if clp_opts is None:
            clp_opts = {}
        self._set_pwa_dynamics(
            pwa_system, D, E, F, G, clp_opts, parallelization, max_num_threads
        )
        self._dynamics = object()  # TODO New dynamics will just be a flag

    def _set_pwa_dynamics(
        self,
        regions: Sequence[PwaRegion],
        D: npt.NDArray[np.floating],
        E: npt.NDArray[np.floating],
        F: npt.NDArray[np.floating],
        G: npt.NDArray[np.floating],
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
        DF = cs.diagcat(cs.sparsify(D), cs.sparsify(F))
        EG = np.concatenate((E, G))
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
        lp = {"h": cs.Sparsity(ns + na, ns + na), "a": DF.sparsity()}
        lpsolver = cs.conic("lpsolver", "clp", lp, clp_opts)

        mapped_lpsolver = lpsolver.map(nr * n_ineq, parallelization, max_num_threads)
        sol = mapped_lpsolver(g=SR, a=DF, uba=EG)
        big_M = -sol["cost"].toarray().reshape(nr, n_ineq) - T

        mapped_lpsolver = lpsolver.map(nr * ns, parallelization, max_num_threads)
        sol = mapped_lpsolver(g=AB, a=DF, uba=EG)
        tmp_lb = sol["cost"].toarray().reshape(nr, ns) + C
        M_lb = tmp_lb.min(0)
        sol = mapped_lpsolver(g=-AB, a=DF, uba=EG)
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

        # set polytopic domain constraints
        self.constraint("state_constraints", D @ X - E, "<=", 0)
        self.constraint("input_constraints", F @ U - G, "<=", 0)
