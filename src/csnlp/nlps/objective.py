from functools import partial
from typing import Any, Dict, Literal, Optional, TypeVar

import casadi as cs
import numpy as np
import numpy.typing as npt

from csnlp.core.solutions import Solution, subsevalf
from csnlp.nlps.constraints import HasConstraints
from csnlp.nlps.parameters import HasParameters

T = TypeVar("T", cs.SX, cs.MX)


class HasObjective(HasParameters[T], HasConstraints[T]):
    """Class for creating an NLP problem with parameters, variables, constraints and an
    objective."""

    def __init__(
        self,
        sym_type: Literal["SX", "MX"] = "SX",
        remove_redundant_x_bounds: bool = True,
        name: Optional[str] = None,
    ) -> None:
        """Instantiate the class.

        Parameters
        ----------
        sym_type : "SX" or "MX", optional
            The CasADi symbolic variable type to use in the NLP, by default "SX".
        remove_redundant_x_bounds : bool, optional
            If `True`, then redundant entries in `lbx` and `ubx` are removed when
            properties `h_lbx` and `h_ubx` are called. See these two properties for more
            details. By default, `True`.
        """
        HasParameters.__init__(self, sym_type)
        HasConstraints.__init__(self, sym_type, remove_redundant_x_bounds)
        self.name = name
        self._f: Optional[T] = None
        self._solver: Optional[cs.Function] = None
        self._solver_opts: Dict[str, Any] = {}
        self._failures = 0

    @property
    def f(self) -> Optional[T]:
        """Gets the objective of the NLP scheme, which is `None` if not set previously
        set via the `minimize` method."""
        return self._f

    @property
    def solver(self) -> Optional[cs.Function]:
        """Gets the NLP optimization solver. Can be `None`, if the solver is not set
        with method `init_solver`."""
        return self._solver

    @property
    def solver_opts(self) -> Dict[str, Any]:
        """Gets the NLP optimization solver options. The dict is empty, if the solver
        options are not set with method `init_solver`."""
        return self._solver_opts

    @property
    def failures(self) -> int:
        """Gets the cumulative number of failures of the NLP solver."""
        return self._failures

    def init_solver(
        self,
        opts: Optional[Dict[str, Any]] = None,
        solver: Literal["opti", "qp"] = "opti",
    ) -> None:
        """Initializes the solver for this NLP with the given options.

        Parameters
        ----------
        opts : Dict[str, Any], optional
            Options to be passed to the CasADi interface to the solver.
        solver : 'ipopt' or 'qp', optional
            Type of solver to instantiate. By default, IPOPT is chosen to deal with
            NLPs. However, if the optimization problem is linear, `qp` can be specified.

        Raises
        ------
        RuntimeError
            Raises if the objective has not yet been specified with `minimize`.
        """
        if self._f is None:
            raise RuntimeError("NLP objective not set.")
        opts = {} if opts is None else opts.copy()
        con = cs.vertcat(self._g, self._h)
        problem = {"x": self._x, "p": self._p, "g": con, "f": self._f}
        func, stype = (cs.qpsol, "qrqp") if solver == "qp" else (cs.nlpsol, "ipopt")
        self._solver = func(f"solver_{stype}_{self.name}", stype, problem, opts)
        self._solver_type = solver
        self._solver_opts = opts

    def refresh_solver(self) -> None:
        """Refresh and resets the internal solver function (with the same options, if
        previously set)."""
        if self._solver is not None:
            self.init_solver(self._solver_opts, self._solver_type)

    def minimize(self, objective: T) -> None:
        """Sets the objective function to be minimized.

        Parameters
        ----------
        objective : casadi SX or MX
            A Symbolic variable dependent only on the NLP variables and parameters that
            needs to be minimized.

        Raises
        ------
        ValueError
            Raises if the objective is not scalar.
        """
        if not objective.is_scalar():
            raise ValueError("Objective must be scalar.")
        self._f = objective
        self.refresh_solver()

    def parameter(self, *args, **kwargs):
        out = super().parameter(*args, **kwargs)
        self.refresh_solver()
        return out

    def variable(self, *args, **kwargs):
        out = super().variable(*args, **kwargs)
        self.refresh_solver()
        return out

    def constraint(self, *args, **kwargs):
        out = super().constraint(*args, **kwargs)
        self.refresh_solver()
        return out

    def solve(
        self,
        pars: Optional[Dict[str, npt.ArrayLike]] = None,
        vals0: Optional[Dict[str, npt.ArrayLike]] = None,
    ) -> Solution:
        """Solves the NLP optimization problem.

        Parameters
        ----------
        pars : dict[str, array_like], optional
            Dictionary or structure containing, for each parameter in the NLP scheme,
            the corresponding numerical value. Can be `None` if no parameters are
            present.
        vals0 : dict[str, array_like], optional
            Dictionary or structure containing, for each variable in the NLP scheme, the
            corresponding initial guess. By default, initial guesses are not passed to
            the solver.

        Returns
        -------
        sol : Solution
            A solution object containing all the information.

        Raises
        ------
        RuntimeError
            Raises if the solver is un-initialized (see `init_solver`); or if not all
            the parameters are not provided with a numerical value.
        """
        if pars is None:
            pars = {}
        if self._solver is None:
            raise RuntimeError("Solver uninitialized.")
        parsdiff = self._pars.keys() - pars.keys()
        if len(parsdiff) != 0:
            raise RuntimeError(
                "Trying to solve the NLP with unspecified parameters: "
                + ", ".join(parsdiff)
                + "."
            )

        p = subsevalf(self._p, self._pars, pars)
        kwargs = {
            "p": p,
            "lbx": self._lbx,
            "ubx": self._ubx,
            "lbg": np.concatenate((np.zeros(self.ng), np.full(self.nh, -np.inf))),
            "ubg": 0,
        }
        if vals0 is not None:
            kwargs["x0"] = subsevalf(self._x, self._vars, vals0)
        sol: Dict[str, cs.DM] = self._solver(**kwargs)

        # extract lam_x, lam_g and lam_h
        lam_lbx = -cs.fmin(sol["lam_x"], 0)
        lam_ubx = cs.fmax(sol["lam_x"], 0)
        lam_g = sol["lam_g"][: self.ng, :]
        lam_h = sol["lam_g"][self.ng :, :]

        vars = self.variables
        vals = {name: subsevalf(var, self._x, sol["x"]) for name, var in vars.items()}
        old = cs.vertcat(
            self._p, self._x, self._lam_g, self._lam_h, self._lam_lbx, self._lam_ubx
        )
        new = cs.vertcat(p, sol["x"], lam_g, lam_h, lam_lbx, lam_ubx)
        get_value = partial(subsevalf, old=old, new=new)
        solution = Solution[T](
            f=float(sol["f"]),
            vars=vars,
            vals=vals,
            stats=self._solver.stats().copy(),
            _get_value=get_value,
        )
        self._failures += int(not solution.success)
        return solution
