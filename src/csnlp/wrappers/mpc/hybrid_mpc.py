from inspect import signature
from itertools import chain
from typing import Callable, Literal, Optional, TypeVar, Union

import casadi as cs
import numpy as np

from csnlp.multistart.multistart_nlp import _chained_subevalf, _n

from ..wrapper import Nlp
from .mpc import Mpc
from .mpc import _n as _name_init_state
from ...core.data import cs2array

SymType = TypeVar("SymType", cs.SX, cs.MX)


class HybridMpc(Mpc[SymType]):
    """A hybrid MPC controller, where there exists discrete variables in either
    the state, control, or constraints, e.g., piecewise affine dynamics converted
    to mixed logical dynamical form via mixed-integer constraints:

    Bemporad, Alberto, and Manfred Morari.
    "Control of systems integrating logic, dynamics, and constraints."
    Automatica 35.3 (1999): 407-427.

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
    # TODO NOT ON LBU AND UBU NOT BEING SET WITH PWA_DYNAMICS

    def set_pwa_dynamics(
        self, pwa_system: dict
    ) -> None:  # TODO add arguments passed explicitly
        """Sets the piecewise affine dynamics of the system for the MPC controller,
        creating auxiliary variables and constraints to handle the PWA switching.

        Parameters
        ----------
        pwa_system : dict
            A dictionary containing the piecewise affine system dynamics matrics. The dynamics are
            defined as x^+ = A[i]x + B[i]u + c[i] if S[i]x + R[i]u <= T[i]. Additionally, matrices
            for constraining both the state and input are required, as both must be bounded for
            the model conversion. The dictionary has the following keys:

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
                "The number of inequalities defining regions must be equal for all regions."
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

        M_region = np.zeros(
            (s, l)
        )  # big-M relaxation for region constraints (number of region, number of inequalities)
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

        # dynamics constraints - we now have to add constraints for all regions at each time-step, with the binary variable delta
        # selecting the active region
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

    def _single_shooting_pwa_dynamics(self, pwa_system: dict) -> None:
        # TODO Implement single shooting for PWA dynamics
        raise NotImplementedError(
            "Single shooting for PWA dynamics is not yet implemented."
        )
