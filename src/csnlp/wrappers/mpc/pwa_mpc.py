from typing import TypeVar

import casadi as cs
import numpy as np

from .mpc import Mpc

SymType = TypeVar("SymType", cs.SX, cs.MX)


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
        self, pwa_system: dict
    ) -> None:  # TODO add arguments passed explicitly
        """Sets the piecewise affine dynamics of the system for the MPC controller,
        creating auxiliary variables and constraints to handle the PWA switching.

        Parameters
        ----------
        pwa_system : dict
            A dictionary containing the piecewise affine system dynamics matrics. The
            dynamics are defined as x^+ = A[i]x + B[i]u + c[i] if S[i]x + R[i]u <= T[i].
            Additionally, matrices for constraining both the state and input are
            required, as both must be bounded for the model conversion. The dictionary
            has the following keys:

            - 'A' : numpy.ndarray
                The state matrix of the dynamics.
            - 'B' : numpy.ndarray
                The input matrix of the dynamics.
            - 'c' : numpy.ndarray
                The constant term of the dynamics.
            - 'S' : numpy.ndarray
                The state matrix of the region inequality.
            - 'R' : numpy.ndarray
                The input matrix of the region inequality.
            - 'T' : numpy.ndarray
                The constant matrix of the region inequality.
            - 'D' : numpy.ndarray
                The state matrix of the state constraints Dx <= E.
            - 'E' : numpy.ndarray
                The constant matrix of the state constraints Dx <= E.
            - 'F' : numpy.ndarray
                The input matrix of the input constraints Fu <= G.
            - 'G' : numpy.ndarray
                The constant matrix of the input constraints Fu <= G.
        """
        if self._dynamics is not None:
            raise RuntimeError("Dynamics were already set.")
        s = len(pwa_system["A"])  # the number of switching regions
        if not all(len(pwa_system[key]) == s for key in ("B", "c", "S", "R", "T")):
            raise ValueError("The number of matrices in the PWA system must be equal.")
        l = pwa_system["T"][0].shape[0]  # the number of inequalities defining regions
        if not all(pwa_system["T"][i].shape[0] == l for i in range(s)):
            raise ValueError(
                "The number of inequalities defining regions must be equal for all "
                " regions."
            )
        if self._is_multishooting:
            self._multishooting_pwa_dynamics(pwa_system)
        else:
            self._single_shooting_pwa_dynamics(pwa_system)
        self._dynamics = object()  # TODO New dynamics will just be a flag - change

    def _multishooting_pwa_dynamics(self, pwa_system: dict) -> None:
        """Internal utility to create pwa dynamics constraints in multiple shooting."""
        # extract values from system
        S = pwa_system["S"]
        R = pwa_system["R"]
        T = pwa_system["T"]
        A = pwa_system["A"]
        B = pwa_system["B"]
        c = pwa_system["c"]
        D = pwa_system["D"]
        E = pwa_system["E"]
        F = pwa_system["F"]
        G = pwa_system["G"]
        s = len(S)  # number of PWA regions
        n = A[0].shape[0]  # state dimension
        m = B[0].shape[1]  # input dimension
        l = T[0].shape[0]  # number of inequalities defining regions
        N = self._prediction_horizon

        # solve linear programs to determine bounds for big-M relaxations
        x = cs.SX.sym("x", n)
        u = cs.SX.sym("u", m)

        # big-M relaxation for region constraints (number of region, number of
        # inequalities)
        M_region = np.zeros((s, l))
        for i in range(s):
            region = S[i] @ x + R[i] @ u - T[i]
            for j in range(l):
                # TODO add solve options including verbose
                lp = {
                    "x": cs.vertcat(x, u),
                    "f": -region[j],
                    "g": cs.vertcat(D @ x - E, F @ u - G),
                }  # negative sign for maximization
                solver = cs.qpsol("S", "clp", lp)
                sol = solver(ubg=0)
                M_region[i, j] = -sol["f"]

        M_ub = np.zeros((n, 1))  # big-M relaxation for dynamics
        M_lb = np.zeros((n, 1))
        temp_upper = np.zeros((s, 1))
        temp_lower = np.zeros((s, 1))
        for j in range(n):
            for i in range(s):
                lp = {
                    "x": cs.vertcat(x, u),
                    "f": -(A[i][[j], :] @ x + B[i][[j], :] @ u + c[i][[j], :]),
                    "g": cs.vertcat(D @ x - E, F @ u - G),
                }
                solver = cs.qpsol("S", "clp", lp)
                sol = solver(ubg=0)
                temp_upper[i] = -sol["f"]

                lp = {
                    "x": cs.vertcat(x, u),
                    "f": A[i][[j], :] @ x + B[i][[j], :] @ u + c[i][[j], :],
                    "g": cs.vertcat(D @ x - E, F @ u - G),
                }
                solver = cs.qpsol("S", "clp", lp)
                sol = solver(ubg=0)
                temp_lower[i] = sol["f"]
            M_ub[j] = np.max(temp_upper)
            M_lb[j] = np.min(temp_lower)

        # auxiliary variables
        z = [self.variable(f"z_{i}", (n, N))[0] for i in range(s)]
        delta, _, _ = self.variable("delta", (s, N), lb=0, ub=1, discrete=True)

        # dynamics constraints - we now have to add constraints for all regions at each
        # time-step, with the binary variable delta selecting the active region
        X = cs.vcat(self._states.values())
        U = cs.vcat(self._actions_exp.values())
        self.constraint("delta_sum", cs.sum1(delta), "==", 1)
        self.constraint("state_constraints", D @ X - E, "<=", 0)
        self.constraint("input_constraints", F @ U - G, "<=", 0)
        self.constraint("dynamics", X[:, 1:], "==", sum(z))
        X_ = X[:, :-1]
        z_ub = []
        z_lb = []
        region = []
        z_x_ub = []
        z_x_lb = []
        for i in range(s):
            z_ub.append(z[i] - M_ub @ delta[i, :])
            z_lb.append(z[i] - M_lb @ delta[i, :])
            region.append(
                S[i] @ X_ + R[i] @ U - T[i] - M_region[i, :] @ (1 - delta[i, :])
            )
            z_x_ub.append(
                z[i] - (A[i] @ X_ + B[i] @ U + c[i] - M_lb @ (1 - delta[i, :]))
            )
            z_x_lb.append(
                z[i] - (A[i] @ X_ + B[i] @ U + c[i] - M_ub @ (1 - delta[i, :]))
            )
        self.constraint("z_ub", cs.vcat(z_ub), "<=", 0)
        self.constraint("z_lb", cs.vcat(z_lb), ">=", 0)
        self.constraint("region", cs.vcat(region), "<=", 0)
        self.constraint("z_x_ub", cs.vcat(z_x_ub), "<=", 0)
        self.constraint("z_x_lb", cs.vcat(z_x_lb), ">=", 0)

    def _single_shooting_pwa_dynamics(self, *_, **__) -> None:
        raise NotImplementedError(
            "Single shooting for PWA dynamics is not yet implemented."
        )
