from collections.abc import Iterable, Iterator
from functools import lru_cache, partial
from itertools import repeat
from typing import Any, ClassVar, Generic, Literal, Optional, TypeVar, Union

import casadi as cs
import numpy as np
import numpy.typing as npt
from joblib import Parallel, delayed
from joblib.memory import MemorizedFunc

from ..core.cache import invalidate_cache
from ..core.solutions import Solution, subsevalf
from ..nlps.nlp import Nlp
from ..nlps.objective import _solve_and_get_stats

SymType = TypeVar("SymType", cs.SX, cs.MX)


def _n(sym_name: str, scenario: int) -> str:
    """Internal utility for the naming convention of the ``i``-th scenario's symbols."""
    return f"{sym_name}__{scenario}"


def _chained_subevalf(
    expr: Union[SymType, np.ndarray],
    old_vars: dict[str, SymType],
    new_vars: dict[str, SymType],
    old_pars: dict[str, SymType],
    new_pars: dict[str, SymType],
    old_dual_vars: Optional[dict[str, SymType]] = None,
    new_dual_vars: Optional[dict[str, SymType]] = None,
    eval: bool = True,
) -> Union[SymType, cs.DM, np.ndarray]:
    """Internal utility to perform :func:`subevalf` on variables, parameters and dual
    variables in chain."""
    expr = subsevalf(expr, old_vars, new_vars, eval=False)
    if old_dual_vars is None:
        return subsevalf(expr, old_pars, new_pars, eval=eval)
    expr = subsevalf(expr, old_pars, new_pars, eval=False)
    return subsevalf(expr, old_dual_vars, new_dual_vars, eval=eval)


def _find_best_sol(sols: Iterator[dict[str, Any]]) -> dict[str, Any]:
    """Picks the best solution out of multiple solutions, with the following logic: the
    current solution should be considered better if

    * it is feasible and the best is not, or
    * both are feasible or infeasible, and the current is successful and the best is
      not, or
    * both are successful or not, and the current has a lower optimal value than the
      best.
    """
    best_sol = next(sols)
    best_f = float(best_sol["f"])
    is_best_successful = best_sol["stats"]["success"]
    is_best_feasible = "infeasib" not in best_sol["stats"]["return_status"].lower()
    for sol in sols:
        f = float(sol["f"])
        is_successful = sol["stats"]["success"]
        is_feasible = "infeasib" not in sol["stats"]["return_status"].lower()
        if (
            (is_feasible and not is_best_feasible)
            or (
                is_feasible == is_best_feasible
                and (is_successful and not is_best_successful)
            )
            or (is_successful == is_best_successful and f < best_f)
        ):
            best_sol = sol
            best_f = f
            is_best_successful = is_successful
            is_best_feasible = is_feasible
    return best_sol


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
        **_,
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
        lb: Union[npt.ArrayLike, cs.DM] = -np.inf,
        ub: Union[npt.ArrayLike, cs.DM] = +np.inf,
    ) -> tuple[SymType, SymType, SymType]:
        out = super().variable(name, shape, lb, ub)
        for i in range(self._starts):
            self._stacked_nlp.variable(_n(name, i), shape, lb, ub)
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
            expr_i = _chained_subevalf(
                expr_, vars, self._vars_i(i), pars, self._pars_i(i), eval=False
            )
            self._stacked_nlp.constraint(_n(name, i), expr_i, op_, 0, False, False)
        return out

    def minimize(self, objective: SymType) -> None:
        out = super().minimize(objective)

        vars = self.variables
        pars = self.parameters
        self._fs: list[SymType] = [
            _chained_subevalf(
                objective, vars, self._vars_i(i), pars, self._pars_i(i), eval=False
            )
            for i in range(self._starts)
        ]
        self._stacked_nlp.minimize(sum(self._fs))
        return out

    def init_solver(
        self,
        opts: Optional[dict[str, Any]] = None,
        solver: str = "ipopt",
        type: Literal["auto", "nlp", "conic"] = "auto",
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
        **_,
    ) -> Union[Solution[SymType], list[Solution[SymType]]]:
        assert not (
            return_stacked_sol and return_all_sols
        ), "`return_all_sols` and `return_stacked_sol` can't be both true."
        if pars is not None:
            pars_iter = repeat(pars, self.starts) if isinstance(pars, dict) else pars
            pars = {
                _n(n, i): pars_i[n]
                for i, pars_i in enumerate(pars_iter)
                for n in pars_i.keys()
            }
        if vals0 is not None:
            v0_iter = repeat(vals0, self.starts) if isinstance(vals0, dict) else vals0
            vals0 = {
                _n(n, i): vals0_i[n]
                for i, vals0_i in enumerate(v0_iter)
                for n in vals0_i.keys()
            }
        multi_sol = self._stacked_nlp.solve(pars, vals0)
        if return_stacked_sol:
            return multi_sol

        vars_ = self.variables
        pars_ = self.parameters
        duals_ = self.dual_variables
        old = cs.vertcat(
            self._x, self._lam_g, self._lam_h, self._lam_lbx, self._lam_ubx, self._p
        )
        fs = [float(multi_sol._get_value(f)) for f in self._fs]

        def get_ith_sol(idx: int) -> Solution[SymType]:
            vals = {n: multi_sol.vals[_n(n, idx)] for n in vars_.keys()}
            dual_vals = {n: multi_sol.dual_vals[_n(n, idx)] for n in duals_.keys()}
            new = multi_sol._get_value(
                _chained_subevalf(
                    old,
                    vars_,
                    self._vars_i(idx),
                    pars_,
                    self._pars_i(idx),
                    duals_,
                    self._dual_vars_i(idx),
                    False,
                )
            )
            get_value = partial(subsevalf, old=old, new=new)
            return Solution(
                fs[idx], vars_, vals, duals_, dual_vals, multi_sol.stats, get_value
            )

        if return_all_sols:
            return [get_ith_sol(i) for i in range(self._starts)]
        return get_ith_sol(np.argmin(fs).item())

    def remove_variable_bounds(
        self,
        name: str,
        direction: Literal["lb", "ub", "both"],
        idx: Union[tuple[int, int], list[tuple[int, int]]] = None,
    ) -> None:
        idx = [idx] if isinstance(idx, tuple) else list(idx)
        super().remove_variable_bounds(name, direction, idx)
        for i in range(self._starts):
            self._stacked_nlp.remove_variable_bounds(_n(name, i), direction, idx)

    def remove_constraints(
        self, name: str, idx: Union[tuple[int, int], list[tuple[int, int]]] = None
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
        self.initialize_parallel()

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
        **_,
    ) -> Union[Solution[SymType], list[Solution[SymType]]]:
        if self._solver is None:
            raise RuntimeError("Solver uninitialized.")
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
            return list(map(self._process_solver_sol, sols))
        best_sol = _find_best_sol(iter(sols))
        return self._process_solver_sol(best_sol)

    def __getstate__(self, fullstate: bool = False) -> dict[str, Any]:
        # joblib.Parallel cannot be pickled or deepcopied
        state = super().__getstate__(fullstate)
        state.pop("_parallel", None)
        return state

    def __setstate__(self, state: Optional[dict[str, Any]]) -> None:
        if state is not None:
            self.__dict__.update(state)
        self.initialize_parallel()


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
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, starts=starts, **kwargs)
        self._mapped_solver: Optional[MemorizedFunc] = None
        self._parallelization = parallelization
        self._max_num_threads = max_num_threads

    def init_solver(self, *args: Any, **kwargs: Any) -> None:
        super().init_solver(*args, **kwargs)
        solver: cs.Function = self._solver.func
        mapped_solver = solver.map(
            self._starts, self._parallelization, self._max_num_threads or self._starts
        )
        self._mapped_solver = self._cache.cache(mapped_solver)

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
        **_,
    ) -> Union[Solution[SymType], list[Solution[SymType]]]:
        assert not (
            _return_mapped_sol and return_all_sols
        ), "`return_all_sols` and `_return_mapped_sol` can't be both true."
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

        # NOTE: the mapped solver does not return the stats, so we have to use the
        # original solver to get them - this likely returns the stats of the last
        # run of the mapped solver, but it's the best we can do up to this point
        stats = self._solver.func.stats()

        # if the user wants the mapped solution, return it, though it's not a Solution
        # and it's mostly for debugging
        if _return_mapped_sol:
            single_sol["p"] = single_kwargs["p"]
            single_sol["stats"] = stats
            return single_sol  # NOTE: this is NOT a Solution object

        # convert the mapped solution to a list of Solution objects
        if return_all_sols:
            sols = []
            for i, p in enumerate(ps):
                sol_i = {k: v[:, i] for k, v in single_sol.items()}
                sol_i["p"] = ps[i]
                sol_i["stats"] = stats.copy()
                sols.append(self._process_solver_sol(sol_i))
            return sols

        # just return the best, as we cannot use stats to exclude failed solutions
        i = np.argmin(single_sol["f"])
        sol = {k: v[:, i] for k, v in single_sol.items()}
        sol["p"] = ps[i]
        sol["stats"] = stats
        return self._process_solver_sol(sol)
