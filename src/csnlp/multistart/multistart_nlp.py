from collections.abc import Iterable, Iterator
from functools import lru_cache
from itertools import repeat
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Generic,
    Literal,
    Optional,
    TypeVar,
    Union,
)

import casadi as cs
import numpy as np
import numpy.typing as npt
from joblib import Parallel, delayed

if TYPE_CHECKING:
    from joblib.memory import MemorizedFunc

from ..core.cache import invalidate_cache
from ..core.solutions import (
    EagerSolution,
    LazySolution,
    Solution,
    _is_infeas,
)
from ..nlps.nlp import Nlp
from ..nlps.objective import _solve_and_get_stats

SymType = TypeVar("SymType", cs.SX, cs.MX)


def _n(sym_name: str, scenario: int) -> str:
    """Internal utility for the naming convention of the ``i``-th scenario's symbols."""
    return f"{sym_name}__{scenario}"


def _chained_substitute(
    expr: SymType, *old_and_new: tuple[dict[str, SymType], dict[str, SymType]]
) -> Union[SymType, cs.DM]:
    """Internal utility to perform substitutions in chain."""
    OLD = []
    NEW = []
    for old, new in old_and_new:
        if old and new:
            for n, old_ in old.items():
                OLD.append(old_)
                NEW.append(new[n])
    return cs.substitute(expr, cs.vvcat(OLD), cs.vvcat(NEW))


def _cmp_key(
    sol: dict[str, Any], plugin_solver: Optional[str]
) -> tuple[bool, bool, float]:
    """Internal utility, similar to :func:`Solution.cmp_key`, but for native CasADi's
    solution dictionaries."""
    stats = sol["stats"]
    status = str(stats["return_status"])
    is_infeas = _is_infeas(status, plugin_solver) if plugin_solver is not None else None
    if is_infeas is None:
        is_infeas = "infeas" in status.lower()
    return is_infeas, not stats["success"], float(sol["f"])


def _get_kkt_stats(
    sol: dict[str, Any],
    ng: int,
    nonmasked_lbx_idx: Union[slice, npt.NDArray[np.int64]],
    nonmasked_ubx_idx: Union[slice, npt.NDArray[np.int64]],
    lbx: npt.NDArray[np.float64],
    ubx: npt.NDArray[np.float64],
    dlagrangian: cs.Function,
    tol_eq: float,
    tol_ineq: float,
    tol_comp: float,
    tol_stat: float,
) -> dict[str, Any]:
    """Internal utility to check the KKT conditions of a solution."""
    g_and_h = sol["g"]
    g = np.asarray(g_and_h[:ng, :].elements())
    h = np.asarray(g_and_h[ng:, :].elements())
    if (h > tol_ineq).any() or (np.abs(g) > tol_eq).any():
        return {"success": False, "return_status": "infeasible (primal)"}

    lam_g_and_h = sol["lam_g"]
    lam_h = np.asarray(lam_g_and_h[ng:, :].elements())
    lam_x = np.asarray(sol["lam_x"].elements())
    lam_lbx = -np.minimum(lam_x[nonmasked_lbx_idx], 0)
    lam_ubx = np.maximum(lam_x[nonmasked_ubx_idx], 0)
    x = np.asarray(sol["x"].elements())
    if (
        (np.abs(lam_lbx * (lbx - x[nonmasked_lbx_idx])) > tol_comp).any()
        or (np.abs(lam_ubx * (x[nonmasked_ubx_idx] - ubx)) > tol_comp).any()
        or (np.abs(lam_h * h) > tol_comp).any()
    ):
        return {"success": False, "return_status": "infeasible (complementary)"}

    lam_g = lam_g_and_h[:ng, :]
    dL = dlagrangian(x, sol["p"], lam_g, lam_h, lam_lbx, lam_ubx)
    dL = np.asarray(dL.elements()).flatten()
    if (np.abs(dL) > tol_stat).any():
        return {"success": False, "return_status": "non-stationarity point"}

    return {"success": True, "return_status": "suspected success"}


class MultistartNlp(Nlp[SymType], Generic[SymType]):
    """Base class for NLP with multistarting. This class lays the foundation for solving
    an NLP problem (described as an instance of :class:`csnlp.Nlp`) multiple times with
    different initial conditions, and requires subclasses to implement the actual
    multistarting logic in :meth:`solve_multi`.

    Parameters
    ----------
    args, kwargs
        See inherited :meth:`csnlp.Nlp.__init__`.
    starts : int
        A positive integer for the number of multiple starting guesses to optimize.

    Raises
    ------
    ValueError
        Raises if the scenario number is invalid.
    """

    is_multi: ClassVar[bool] = True
    """Flag to indicate that this is a multistart NLP."""

    def __init__(self, *args: Any, starts: int, **kwargs: Any) -> None:
        if starts <= 0:
            raise ValueError("Number of scenarios must be positive and > 0.")
        super().__init__(*args, **kwargs)
        self._starts = starts

    def __call__(
        self,
        pars: Union[
            None, dict[str, npt.ArrayLike], Iterable[dict[str, npt.ArrayLike]]
        ] = None,
        vals0: Union[
            None, dict[str, npt.ArrayLike], Iterable[dict[str, npt.ArrayLike]]
        ] = None,
        **kwargs: Any,
    ) -> Union[Solution[SymType], list[Solution[SymType]]]:
        # call solve_multi only if either pars or vals0 is an iterable; otherwise, run
        # the single, base NLP
        if (pars is None or isinstance(pars, dict)) and (
            vals0 is None or isinstance(vals0, dict)
        ):
            return self.solve(pars, vals0)
        return self.solve_multi(pars, vals0, **kwargs)

    @property
    def starts(self) -> int:
        """Gets the number of starts."""
        return self._starts

    def solve_multi(
        self,
        pars: Union[
            None, dict[str, npt.ArrayLike], Iterable[dict[str, npt.ArrayLike]]
        ] = None,
        vals0: Union[
            None, dict[str, npt.ArrayLike], Iterable[dict[str, npt.ArrayLike]]
        ] = None,
        return_all_sols: bool = False,
        **_: Any,
    ) -> Union[Solution[SymType], list[Solution[SymType]]]:
        """Solves the NLP with multiple initial conditions.

        Parameters
        ----------
        pars : dict of (str, array_like) or iterable of, optional
            An iterable that, for each multistart, contains a dictionary with, for each
            parameter in the NLP scheme, the corresponding numerical value. In case a
            single dict is passed, the same is used across all scenarions. Can be
            ``None`` if no parameters are present.
        vals0 : dict of (str, array_like) or iterable of, optional
            An iterable that, for each multistart, contains a dictionary with, for each
            variable in the NLP scheme, the corresponding initial guess. In case a
            single dict is passed, the same is used across all scenarions. By default
            ``None``, in which case  initial guesses are not passed to the solver.
        return_all_sols : bool, optional
            If ``True``, returns the solution of each multistart of the NLP; otherwise,
            only the best solution is returned. By default, ``False``.

        Returns
        -------
        Solution or list of Solutions
            Depending on the flags ``return_all_sols``, returns

            - the best solution out of all multiple starts
            - all the solutions (one per start).
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement `solve_multi`"
        )


class StackedMultistartNlp(MultistartNlp[SymType], Generic[SymType]):
    """A class that models and solves an NLP problem from multiple starting initial
    guesses by automatically stacking the original problem multiple independent times
    in the same, larger-scale NLP. This allows to solve the original problem multiple
    times via a single call to the solver."""

    def __init__(self, *args: Any, starts: int, **kwargs: Any) -> None:
        # this class essentially is a facade that hides an internal nlp in which the
        # problem (variables, parameters, etc.) are duplicated by the requested number
        # of multiple starts. For this reason, all methods are overridden to create
        # multiples of these in the hidden nlp.
        super().__init__(*args, starts=starts, **kwargs)
        self._stacked_nlp = Nlp(*args, **kwargs)  # actual nlp

    @lru_cache(maxsize=32)
    def _vars_i(self, i: int) -> dict[str, SymType]:
        """Internal utility to retrieve the variables of the ``i``-th scenario."""
        nlp = self._stacked_nlp.unwrapped
        return {k: nlp._vars[_n(k, i)] for k in self._vars}

    @lru_cache(maxsize=32)
    def _pars_i(self, i: int) -> dict[str, SymType]:
        """Internal utility to retrieve the parameters of the ``i``-th scenario."""
        nlp = self._stacked_nlp.unwrapped
        return {k: nlp._pars[_n(k, i)] for k in self._pars}

    @lru_cache(maxsize=32)
    def _dual_vars_i(self, i: int) -> dict[str, SymType]:
        """Internal utility to retrieve the dual variables of the ``i``-th scenario."""
        nlp = self._stacked_nlp.unwrapped
        return {k: nlp._dual_vars[_n(k, i)] for k in self._dual_vars}

    @invalidate_cache(_pars_i)
    def parameter(self, name: str, shape: tuple[int, int] = (1, 1)) -> SymType:
        out = super().parameter(name, shape)
        for i in range(self._starts):
            self._stacked_nlp.parameter(_n(name, i), shape)
        return out

    @invalidate_cache(_vars_i, _dual_vars_i)
    def variable(
        self,
        name: str,
        shape: tuple[int, int] = (1, 1),
        discrete: bool = False,
        lb: Union[npt.ArrayLike, cs.DM] = -np.inf,
        ub: Union[npt.ArrayLike, cs.DM] = +np.inf,
    ) -> tuple[SymType, SymType, SymType]:
        out = super().variable(name, shape, discrete, lb, ub)
        for i in range(self._starts):
            self._stacked_nlp.variable(_n(name, i), shape, discrete, lb, ub)
        return out

    @invalidate_cache(_vars_i, _dual_vars_i)
    def constraint(
        self,
        name: str,
        lhs: Union[SymType, np.ndarray, cs.DM],
        op: Literal["==", ">=", "<="],
        rhs: Union[SymType, np.ndarray, cs.DM],
        soft: bool = False,
        simplify: bool = True,
    ) -> tuple[SymType, ...]:
        expr = lhs - rhs
        if simplify:
            expr = cs.cse(cs.simplify(expr))
        out = super().constraint(name, expr, op, 0, soft, False)

        # NOTE: the line above already created the slack variables in each scenario, so
        # below we have to pass both soft=False and the expression with slack included.
        vars = self.variables
        pars = self.parameters
        expr_ = out[0]  # slack-relaxed expression in the form h(x,p)<=0 or g(x,p)==0
        op_: Literal["==", "<="] = "==" if op == "==" else "<="
        for i in range(self._starts):
            expr_i = _chained_substitute(
                expr_, (vars, self._vars_i(i)), (pars, self._pars_i(i))
            )
            self._stacked_nlp.constraint(_n(name, i), expr_i, op_, 0, False, False)
        return out

    def minimize(self, objective: SymType) -> None:
        out = super().minimize(objective)

        vars = self.variables
        pars = self.parameters
        self._fs: list[SymType] = [
            _chained_substitute(
                objective, (vars, self._vars_i(i)), (pars, self._pars_i(i))
            )
            for i in range(self._starts)
        ]
        self._stacked_nlp.minimize(sum(self._fs))
        return out

    def init_solver(
        self,
        opts: Optional[dict[str, Any]] = None,
        solver: str = "ipopt",
        type: Optional[Literal["nlp", "conic"]] = None,
    ) -> None:
        out = super().init_solver(opts, solver, type)
        self._stacked_nlp.init_solver(opts, solver, type)
        return out

    def solve_multi(
        self,
        pars: Union[
            None, dict[str, npt.ArrayLike], Iterable[dict[str, npt.ArrayLike]]
        ] = None,
        vals0: Union[
            None, dict[str, npt.ArrayLike], Iterable[dict[str, npt.ArrayLike]]
        ] = None,
        return_all_sols: bool = False,
        return_stacked_sol: bool = False,
        **_: Any,
    ) -> Union[Solution[SymType], list[Solution[SymType]]]:
        assert not (return_stacked_sol and return_all_sols), (
            "`return_all_sols` and `return_stacked_sol` can't be both true."
        )
        if pars is not None:
            pars_iter = repeat(pars, self.starts) if isinstance(pars, dict) else pars
            pars = {_n(n, i): ps[n] for i, ps in enumerate(pars_iter) for n in ps}
        if vals0 is not None:
            v0_iter = repeat(vals0, self.starts) if isinstance(vals0, dict) else vals0
            vals0 = {_n(n, i): v0[n] for i, v0 in enumerate(v0_iter) for n in v0}
        multi_sol = self._stacked_nlp.solve(pars, vals0)
        if return_stacked_sol:
            return multi_sol

        x_ = self._x
        p_ = self._p
        lam_g_and_h_ = cs.vertcat(self._lam_g, self._lam_h)
        lam_lbx_and_ubx_ = cs.vertcat(self._lam_lbx, self._lam_ubx)
        vars_ = self.variables
        pars_ = self.parameters
        duals_ = self.dual_variables
        fs = [float(multi_sol.value(f)) for f in self._fs]

        all_vars = cs.vertcat(x_, lam_g_and_h_, lam_lbx_and_ubx_, p_)
        splits = np.cumsum(
            (0, x_.size1(), lam_g_and_h_.size1(), lam_lbx_and_ubx_.size1(), p_.size1())
        )
        solver_plugin = self.unwrapped._solver_plugin

        def get_ith_sol(idx: int) -> EagerSolution[SymType]:
            vals = {n: multi_sol.vals[_n(n, idx)] for n in vars_}
            dual_vals = {n: multi_sol.dual_vals[_n(n, idx)] for n in duals_}
            # get the value of p, and, lam g, lam h and lam lbx and lam ubx in a single
            # substitution to save some time
            all_vals = multi_sol.value(
                _chained_substitute(
                    all_vars,
                    (vars_, self._vars_i(idx)),
                    (pars_, self._pars_i(idx)),
                    (duals_, self._dual_vars_i(idx)),
                )
            )
            x, lam_g_and_h, lam_lbx_and_ubx, p = cs.vertsplit(all_vals, splits)
            return EagerSolution(
                fs[idx],
                p_,
                p,
                x_,
                x,
                lam_g_and_h_,
                lam_g_and_h,
                lam_lbx_and_ubx_,
                lam_lbx_and_ubx,
                vars_,
                vals,
                duals_,
                dual_vals,
                multi_sol.stats,
                solver_plugin,
            )

        if return_all_sols:
            return [get_ith_sol(i) for i in range(self._starts)]
        return get_ith_sol(np.argmin(fs).item())

    def remove_variable_bounds(
        self,
        name: str,
        direction: Literal["lb", "ub", "both"],
        idx: Union[None, tuple[int, int], list[tuple[int, int]]] = None,
    ) -> None:
        idx = [idx] if isinstance(idx, tuple) else list(idx)
        super().remove_variable_bounds(name, direction, idx)
        for i in range(self._starts):
            self._stacked_nlp.remove_variable_bounds(_n(name, i), direction, idx)

    def remove_constraints(
        self, name: str, idx: Union[None, tuple[int, int], list[tuple[int, int]]] = None
    ) -> None:
        idx = [idx] if isinstance(idx, tuple) else list(idx)
        super().remove_constraints(name, idx)
        for i in range(self._starts):
            self._stacked_nlp.remove_constraints(_n(name, i), idx)


class ParallelMultistartNlp(MultistartNlp[SymType], Generic[SymType]):
    """A class that solves an NLP problem multiple times, with different initial
    starting conditions, via parallelization of the computations via :mod:`joblib`.

    Parameters
    ----------
    args, kwargs
        See inherited :meth:`csnlp.Nlp.__init__`.
    starts : int
        A positive integer for the number of multiple starting guesses to optimize.
    parallel_kwargs: dict, optional
        A dictionary with keyword arguments to be used to instantiate the parallel
        :class:`joblib.Parallel` backend. By default, ``None``, in which case the
        parallel backend is instantiated with default arguments.

    Raises
    ------
    ValueError
        Raises if the scenario number is invalid.
    """

    def __init__(
        self,
        *args: Any,
        starts: int,
        parallel_kwargs: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, starts=starts, **kwargs)
        self._parallel_kwargs = parallel_kwargs if parallel_kwargs is not None else {}
        self._parallel: Optional[Parallel] = None

    def initialize_parallel(self) -> None:
        """Initializes the parallel backend."""
        self._parallel = Parallel(**self._parallel_kwargs)
        self._parallel.__enter__()

    def terminate_parallel(self) -> None:
        """Terminates the parallel backend."""
        self._parallel.__exit__(None, None, None)

    def solve_multi(
        self,
        pars: Union[
            None, dict[str, npt.ArrayLike], Iterable[dict[str, npt.ArrayLike]]
        ] = None,
        vals0: Union[
            None, dict[str, npt.ArrayLike], Iterable[dict[str, npt.ArrayLike]]
        ] = None,
        return_all_sols: bool = False,
        **_: Any,
    ) -> Union[Solution[SymType], list[Solution[SymType]]]:
        if self._solver is None:
            raise RuntimeError("Solver uninitialized.")
        if self._parallel is None:
            self.initialize_parallel()
        shared_kwargs = {
            "lbx": self._lbx.data,
            "ubx": self._ubx.data,
            "lbg": np.concatenate((np.zeros(self.ng), np.full(self.nh, -np.inf))),
            "ubg": 0,
        }
        pars_iter = (
            repeat(pars, self.starts)
            if pars is None or isinstance(pars, dict)
            else pars
        )
        vals0_iter = (
            repeat(vals0, self.starts)
            if vals0 is None or isinstance(vals0, dict)
            else vals0
        )
        kwargs = (
            self._process_pars_and_vals0(shared_kwargs.copy(), p, v0)
            for p, v0 in zip(pars_iter, vals0_iter)
        )
        sols: Iterator[dict[str, Any]] = self._parallel(
            delayed(_solve_and_get_stats)(self._solver, kw) for kw in kwargs
        )
        if return_all_sols:
            return [LazySolution.from_casadi_solution(sol, self) for sol in sols]
        best_sol = min(sols, key=lambda s: _cmp_key(s, self._solver_plugin))
        return LazySolution.from_casadi_solution(best_sol, self)


class MappedMultistartNlp(MultistartNlp[SymType], Generic[SymType]):
    """A class that solves an NLP problem multiple times, with different initial
    conditions, in parallel via :func:`casadi.Function.map` parallelization.

    See
    `this wiki <https://github.com/casadi/casadi/wiki/FAQ:-How-to-use-map%28%29-and-mapaccum%28%29-to-speed-up-calculations%3F>`_
    for more details on how to enable parallelization via mapping in CasADi.

    Parameters
    ----------
    args, kwargs
        See inherited :meth:`csnlp.Nlp.__init__`.
    starts : int
        A positive integer for the number of multiple starting guesses to optimize.
    parallelization : "serial", "unroll", "inline", "thread", "openmp"
        The type of parallelization to use (see :func:`casadi.Function.map`). By
        default, ``"serial"`` is selected.
    max_num_threads : int, optional
        Maximum number of threads to use in parallelization; if ``None``, the number of
        threads is equal to the number of starts.

    Raises
    ------
    ValueError
        Raises if the scenario number is invalid.
    """

    def __init__(
        self,
        *args: Any,
        starts: int,
        parallelization: Literal[
            "serial", "unroll", "inline", "thread", "openmp"
        ] = "serial",
        max_num_threads: Optional[int] = None,
        tol_stat: float = 1e-8,
        tol_eq: float = 1e-8,
        tol_ineq: float = 1e-8,
        tol_comp: float = 1e-8,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, starts=starts, **kwargs)
        self._mapped_solver: Optional[MemorizedFunc] = None
        self._parallelization = parallelization
        self._max_num_threads = max_num_threads
        self._tol_eq = tol_eq
        self._tol_ineq = tol_ineq
        self._tol_comp = tol_comp
        self._tol_stat = tol_stat

    def init_solver(self, *args: Any, **kwargs: Any) -> None:
        super().init_solver(*args, **kwargs)
        solver: cs.Function = self._solver.func
        mapped_solver = solver.map(
            self._starts, self._parallelization, self._max_num_threads or self._starts
        )
        self._mapped_solver = self._cache.cache(mapped_solver)

        lagrangian = (
            self.f
            + cs.dot(self.lam_g, self.g)
            + cs.dot(self.lam_h, self.h)
            + cs.dot(self.lam_lbx, self.h_lbx)
            + cs.dot(self.lam_ubx, self.h_ubx)
        )
        dL = cs.jacobian(lagrangian, self.x).T
        self._kkt_stationarity = cs.Function(
            "kkt_stationarity",
            [self.x, self.p, self.lam_g, self.lam_h, self.lam_lbx, self.lam_ubx],
            [dL],
            ["x", "p", "lam_g", "lam_h", "lam_lbx", "lam_ubx"],
            ["dL"],
        )

    def solve_multi(
        self,
        pars: Union[
            None, dict[str, npt.ArrayLike], Iterable[dict[str, npt.ArrayLike]]
        ] = None,
        vals0: Union[
            None, dict[str, npt.ArrayLike], Iterable[dict[str, npt.ArrayLike]]
        ] = None,
        return_all_sols: bool = False,
        _return_mapped_sol: bool = False,
        **_: Any,
    ) -> Union[Solution[SymType], list[Solution[SymType]]]:
        assert not (_return_mapped_sol and return_all_sols), (
            "`return_all_sols` and `_return_mapped_sol` can't be both true."
        )
        if self._mapped_solver is None:
            raise RuntimeError("Solver uninitialized.")
        pars_iter = (
            repeat(pars, self.starts)
            if pars is None or isinstance(pars, dict)
            else pars
        )
        vals0_iter = (
            repeat(vals0, self.starts)
            if vals0 is None or isinstance(vals0, dict)
            else vals0
        )
        x0s = []
        ps = []
        default = cs.DM.zeros(0, 1)
        for p, v0 in zip(pars_iter, vals0_iter):
            kwargs = self._process_pars_and_vals0({}, p, v0)
            x0s.append(kwargs.get("x0", default))
            ps.append(kwargs.get("p", default))
        single_kwargs = {
            "x0": cs.hcat(x0s),
            "p": cs.hcat(ps),
            "lbx": self._lbx.data,
            "ubx": self._ubx.data,
            "lbg": np.concatenate((np.zeros(self.ng), np.full(self.nh, -np.inf))),
            "ubg": 0,
        }
        single_sol: dict[str, cs.DM] = self._mapped_solver(**single_kwargs)

        # if the user wants the mapped solution, return it, though it's not a Solution
        # and it's mostly for debugging
        if _return_mapped_sol:
            single_sol["p"] = single_kwargs["p"]
            single_sol["stats"] = self._mapped_solver.func.stats()
            return single_sol  # NOTE: this is NOT a Solution object

        sols: list[dict[str, Any]] = []
        ng = self.ng
        nonmasked_lbx_idx = self.nonmasked_lbx_idx
        nonmasked_ubx_idx = self.nonmasked_ubx_idx
        lbx = self._lbx.data[nonmasked_lbx_idx]
        ubx = self._ubx.data[nonmasked_ubx_idx]
        dlagrangian = self._kkt_stationarity
        tol_eq = self._tol_eq
        tol_ineq = self._tol_ineq
        tol_comp = self._tol_comp
        tol_stat = self._tol_stat
        for i, p in enumerate(ps):
            sol_i = {k: v[:, i] if v.size2() else v for k, v in single_sol.items()}
            sol_i["p"] = p
            sol_i["stats"] = _get_kkt_stats(
                sol_i,
                ng,
                nonmasked_lbx_idx,
                nonmasked_ubx_idx,
                lbx,
                ubx,
                dlagrangian,
                tol_eq,
                tol_ineq,
                tol_comp,
                tol_stat,
            )
            sols.append(sol_i)
        if return_all_sols:
            return [LazySolution.from_casadi_solution(sol, self) for sol in sols]
        # NOTE: since the return status is here hand-crafted, don't pass the plugin name
        # for detecting infeasibility so as to default to searching for "infeas"
        best_sol = min(sols, key=lambda s: _cmp_key(s, None))
        return LazySolution.from_casadi_solution(best_sol, self)
