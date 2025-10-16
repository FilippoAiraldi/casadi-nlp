from typing import Any, Literal, Optional, TypeVar

import casadi as cs
import numpy as np
import numpy.typing as npt
from joblib import Memory
from joblib.memory import MemorizedFunc

from ..core.solutions import LazySolution, Solution, subsevalf
from .constraints import HasConstraints

SymType = TypeVar("SymType", cs.SX, cs.MX)


def _solve_and_get_stats(
    solver: MemorizedFunc, kwargs: dict[str, npt.ArrayLike]
) -> dict[str, Any]:
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
    r"""Class for creating an NLP problem with parameters, variables, constraints and an
    objective. It builds on top of :class:`HasConstraints`, which handles parameters,
    variables and constraints.

    Parameters
    ----------
    sym_type : {"SX", "MX"}, optional
        The CasADi symbolic variable type to use in the NLP, by default ``"SX"``.
    remove_redundant_x_bounds : bool, optional
        If ``True``, then redundant entries in :meth:`lbx` and :meth:`ubx` are removed
        when properties :meth:`h_lbx` and :meth:`h_ubx` are called. See these two
        properties for more details. By default, ``True``.
    cache : joblib.Memory, optional
        Optional cache to avoid computing the same exact NLP more than once. By default,
        no caching occurs.
    name : str, optional
        Name of the NLP scheme. If ``None``, it is automatically assigned.

    Notes
    -----
    Constraints are internally handled in their canonical form, i.e., :math:`g(x,p) = 0`
    and :math:`h(x,p) \leq 0`. The objective :math:`f(x,p)` is always a scalar function
    to be minimized.
    """

    def __init__(
        self,
        sym_type: Literal["SX", "MX"] = "SX",
        remove_redundant_x_bounds: bool = True,
        cache: Memory = None,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(sym_type, remove_redundant_x_bounds)
        self.name = name
        self._f: Optional[SymType] = None
        self._solver: Optional[MemorizedFunc] = None
        self._solver_opts: dict[str, Any] = {}
        self._cache = cache if cache is not None else Memory(None)

    @property
    def f(self) -> Optional[SymType]:
        """Gets the objective of the NLP scheme, which is ``None`` if not set previously
        set via the :meth:`minimize` method."""
        return self._f

    @property
    def solver(self) -> Optional[cs.Function]:
        """Gets the NLP optimization solver. Can be ``None``, if the solver is not set
        with method :meth:`init_solver`."""
        return self._solver.func if self._solver is not None else None

    @property
    def solver_opts(self) -> dict[str, Any]:
        """Gets the NLP optimization solver options. The dict is empty, if the solver
        options are not set with method :meth:`init_solver`."""
        return self._solver_opts

    def init_solver(
        self,
        opts: Optional[dict[str, Any]] = None,
        solver: str = "ipopt",
        type: Optional[Literal["nlp", "conic"]] = None,
    ) -> None:
        """Initializes the solver for this NLP with the given options.

        Parameters
        ----------
        opts : dict[str, Any], optional
            Options to be passed to the CasADi interface to the solver. Must not contain
            the ``"discrete"`` key, which is automatically set based on the variables'
            domains. By default, ``None``.
        solver : str, optional
            Type of solver to instantiate. For example, ``"ipopt"`` and ``"sqpmethod"``
            trigger the instantiation of an NLP problem, while, e.g., ``"qrqp"``,
            ``"osqp"``, and ``"qpoases"`` of a conic one. However, the solver type can
            be overruled with the ``type`` argument. By default, ``"ipopt"`` is
            selected.
        type : "nlp", "conic", optional
            Type of problem to instantiate. If ``"nlp"``, then the problem is forced to
            be instantiated with :func:`casadi.nlpsol`. If ``"conic"``, then
            :func:`casadi.qpsol` is forced instead. If ``None``, then the problem type
            is selected automatically.

        Raises
        ------
        ValueError
            Raises if the given problem type is not recognized, or if the ``opts`` dict
            contains the
        RuntimeError
            Raises if the type of the problem cannot be inferred automatically (when the
            solver supports both conic and NLPs), if the specified solver plugin cannot
            be found, and if the objective has not yet been specified with
            :meth:`minimize`.
        """
        has_conic = cs.has_conic(solver)
        has_nlpsol = cs.has_nlpsol(solver)
        auto_type = type is None
        if has_conic and has_nlpsol and auto_type:
            raise RuntimeError(
                f"`{solver}` supports both conic- and nlp-type problems, so the problem"
                " type cannot be inferred automatically. Please, provide also argument"
                " 'type' to the method."
            )

        if type == "conic" or (auto_type and has_conic):
            func = cs.qpsol
        elif type == "nlp" or (auto_type and has_nlpsol):
            func = cs.nlpsol
        elif type not in (None, "nlp", "conic"):
            raise ValueError(f"unknown problem type: '{type}'")
        else:
            raise RuntimeError(f"'{solver}' plugin not found in either conic or nlp.")

        if self._f is None:
            raise RuntimeError("NLP objective not set.")

        opts = {} if opts is None else opts.copy()
        if "discrete" in opts or "equality" in opts:
            raise ValueError("'discrete' and 'equality' options are reserved.")
        disc = self.discrete
        opts["discrete"] = disc.tolist() if disc.size == 1 else disc  # bugfix
        eq = np.concatenate(
            (np.full(self.ng, True, dtype=bool), np.full(self.nh, False, dtype=bool))
        )
        opts["equality"] = eq.tolist() if eq.size == 1 else eq  # bugfix

        con = cs.vertcat(self._g, self._h)
        problem = {"x": self._x, "p": self._p, "g": con, "f": self._f}
        solver_func = func(f"solver_{solver}_{self.name}", solver, problem, opts)
        self._solver = self._cache.cache(solver_func)

        opts.pop("discrete", None)
        opts.pop("equality", None)
        self._solver_opts = opts
        self._solver_plugin = solver
        self._solver_type = type

    def refresh_solver(self) -> None:
        """Refresh and resets the internal solver function (with the same options, if
        previously set)."""
        if self._solver is not None:
            self.init_solver(self._solver_opts, self._solver_plugin, self._solver_type)

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

    def parameter(self, *args: Any, **kwargs: Any) -> SymType:
        out = super().parameter(*args, **kwargs)
        self.refresh_solver()
        return out

    def variable(self, *args: Any, **kwargs: Any) -> tuple[SymType, SymType, SymType]:
        out = super().variable(*args, **kwargs)
        self.refresh_solver()
        return out

    def constraint(self, *args: Any, **kwargs: Any) -> tuple[SymType, ...]:
        out = super().constraint(*args, **kwargs)
        self.refresh_solver()
        return out

    def solve(
        self,
        pars: Optional[dict[str, npt.ArrayLike]] = None,
        vals0: Optional[dict[str, npt.ArrayLike]] = None,
    ) -> Solution[SymType]:
        """Solves the NLP optimization problem.

        Parameters
        ----------
        pars : dict[str, array_like], optional
            Dictionary or structure containing, for each parameter in the NLP scheme,
            the corresponding numerical value. Can be ``None`` if no parameters are
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
            Raises if the solver is un-initialized (see :meth:`init_solver`); or if not
            all the parameters are not provided with a numerical value.
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
        return LazySolution.from_casadi_solution(sol_with_stats, self)

    def _process_pars_and_vals0(
        self,
        kwargs: dict[str, npt.ArrayLike],
        pars: Optional[dict[str, npt.ArrayLike]],
        vals0: Optional[dict[str, npt.ArrayLike]],
    ) -> dict[str, npt.ArrayLike]:
        """Internal utility to convert pars and initial-val dicts to solver kwargs."""
        if self._pars:
            if pars is None:
                pars = {}
            if parsdiff := self._pars.keys() - pars.keys():
                raise RuntimeError(
                    "Trying to solve the NLP with unspecified parameters: "
                    + ", ".join(parsdiff)
                    + "."
                )
            kwargs["p"] = subsevalf(self._p, self._pars, pars)
        else:
            kwargs["p"] = cs.DM()
        if vals0 is not None:
            if vals0diff := self._vars.keys() - vals0.keys():
                vals0.update(dict.fromkeys(vals0diff, 0))
            kwargs["x0"] = subsevalf(self._x, self._vars, vals0)
        return kwargs
