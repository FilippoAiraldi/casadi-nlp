from functools import partial
from typing import Any, Callable, Dict, Literal, Optional, TypeVar

import casadi as cs
import numpy as np
import numpy.typing as npt
from joblib import Memory
from joblib.memory import MemorizedFunc

from csnlp.core.solutions import Solution, subsevalf
from csnlp.nlps.constraints import HasConstraints

SymType = TypeVar("SymType", cs.SX, cs.MX)

_XXSOL: Dict[str, Callable] = {
    "ipopt": cs.nlpsol,
    "sqpmethod": cs.nlpsol,
    "qrqp": cs.qpsol,
    "osqp": cs.qpsol,
}


def _solve_and_get_stats(
    solver: MemorizedFunc, kwargs: Dict[str, npt.ArrayLike]
) -> Dict[str, Any]:
    """Internal utility to simultaneously run the solver and get its stats."""
    sol = solver(**kwargs)
    sol["p"] = kwargs["p"]  # add to solution the parameters for which it was computed
    sol["stats"] = solver.func.stats()  # add to solution the solver stats
    # NOTE: in case of failure in retrieving the stats
    # try:
    #     stats = solver.func.stats()
    # except RuntimeError:
    #     stats = {"success": False}
    # sol["stats"] = stats  # add to solution the solver stats
    return sol


class HasObjective(HasConstraints[SymType]):
    """Class for creating an NLP problem with parameters, variables, constraints and an
    objective."""

    __slots__ = (
        "name",
        "_f",
        "_solver",
        "_solver_opts",
        "_solver_type",
        "_cache",
        "_failures",
    )

    def __init__(
        self,
        sym_type: Literal["SX", "MX"] = "SX",
        remove_redundant_x_bounds: bool = True,
        cache: Memory = None,
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
        cache : joblib.Memory, optional
            Optional cache to avoid computing the same exact NLP more than once. By
            default, no caching occurs.
        name : str, optional
            Name of the NLP scheme. If `None`, it is automatically assigned.
        """
        super().__init__(sym_type, remove_redundant_x_bounds)
        self.name = name
        self._f: Optional[SymType] = None
        self._solver: Optional[MemorizedFunc] = None
        self._solver_opts: Dict[str, Any] = {}
        self._cache = cache if cache is not None else Memory(None)
        self._failures = 0

    @property
    def f(self) -> Optional[SymType]:
        """Gets the objective of the NLP scheme, which is `None` if not set previously
        set via the `minimize` method."""
        return self._f

    @property
    def solver(self) -> Optional[cs.Function]:
        """Gets the NLP optimization solver. Can be `None`, if the solver is not set
        with method `init_solver`."""
        return self._solver.func if self._solver is not None else None

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
        solver: Literal["ipopt", "sqpmethod", "qrqp", "osqp"] = "ipopt",
    ) -> None:
        """Initializes the solver for this NLP with the given options.

        Parameters
        ----------
        opts : Dict[str, Any], optional
            Options to be passed to the CasADi interface to the solver.
        solver : {'ipopt', 'sqpmethod', 'qrqp', 'osqp'}, optional
            Type of solver to instantiate. "ipopt" and "sqpmethod" trigger the
            instantiation of an NLP problem, while "qrqp" and "osqp" a conic one. By
            default, 'ipopt' is selected.

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

        func = _XXSOL.get(solver, cs.nlpsol)
        solver_func = func(f"solver_{solver}_{self.name}", solver, problem, opts)

        self._solver = self._cache.cache(solver_func)
        self._solver_type = solver
        self._solver_opts = opts

    def refresh_solver(self) -> None:
        """Refresh and resets the internal solver function (with the same options, if
        previously set)."""
        if self._solver is not None:
            self.init_solver(self._solver_opts, self._solver_type)

    def minimize(self, objective: SymType) -> None:
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
    ) -> Solution[SymType]:
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
        if self._solver is None:
            raise RuntimeError("Solver uninitialized.")
        kwargs = self._process_pars_and_vals0(
            {
                "lbx": self._lbx.data,
                "ubx": self._ubx.data,
                "lbg": np.concatenate((np.zeros(self.ng), np.full(self.nh, -np.inf))),
                "ubg": 0,
            },
            pars,
            vals0,
        )
        sol_with_stats = _solve_and_get_stats(self._solver, kwargs)
        solution = self._process_solver_sol(sol_with_stats)
        self._failures += not solution.success
        return solution

    def _process_pars_and_vals0(
        self,
        kwargs: Dict[str, npt.ArrayLike],
        pars: Optional[Dict[str, npt.ArrayLike]],
        vals0: Optional[Dict[str, npt.ArrayLike]],
    ) -> Dict[str, npt.ArrayLike]:
        """Internal utility to convert pars and initial-val dicts to solver kwargs."""
        if pars is None:
            pars = {}
        if parsdiff := self._pars.keys() - pars.keys():
            raise RuntimeError(
                "Trying to solve the NLP with unspecified parameters: "
                + ", ".join(parsdiff)
                + "."
            )
        kwargs["p"] = subsevalf(self._p, self._pars, pars)
        if vals0 is not None:
            if vals0diff := self._vars.keys() - vals0.keys():
                vals0.update({v: 0 for v in vals0diff})  # type: ignore[has-type]
            kwargs["x0"] = subsevalf(self._x, self._vars, vals0)
        return kwargs

    def _process_solver_sol(self, sol: Dict[str, Any]) -> Solution:
        """Internal utility to convert the solver sol dict to a Solution instance."""
        # objective
        f = float(sol["f"])

        # primal variables and values
        vars = self.variables
        vals = {name: subsevalf(var, self._x, sol["x"]) for name, var in vars.items()}

        # dual variables and values
        lam_lbx = -cs.fmin(sol["lam_x"], 0)[self.nonmasked_lbx_idx, :]
        lam_ubx = cs.fmax(sol["lam_x"], 0)[self.nonmasked_ubx_idx, :]
        lam_g = sol["lam_g"][: self.ng, :]
        lam_h = sol["lam_g"][self.ng :, :]
        dual_vars = self.dual_variables
        dual_vals = {}
        for n, var in dual_vars.items():
            if n.startswith("lam_lb"):
                dual_vals[n] = subsevalf(var, self._lam_lbx, lam_lbx)
            elif n.startswith("lam_ub"):
                dual_vals[n] = subsevalf(var, self._lam_ubx, lam_ubx)
            elif n.startswith("lam_g"):
                dual_vals[n] = subsevalf(var, self._lam_g, lam_g)
            elif n.startswith("lam_h"):
                dual_vals[n] = subsevalf(var, self._lam_h, lam_h)
            else:
                raise RuntimeError(f"unknown dual variable type {n}")

        # get_value function
        old = cs.vertcat(
            self._x, self._lam_g, self._lam_h, self._lam_lbx, self._lam_ubx, self._p
        )
        new = cs.vertcat(sol["x"], lam_g, lam_h, lam_lbx, lam_ubx, sol["p"])
        get_value = partial(subsevalf, old=old, new=new)
        return Solution(f, vars, vals, dual_vars, dual_vals, sol["stats"], get_value)
