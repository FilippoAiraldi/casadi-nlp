from functools import lru_cache, partial
from itertools import repeat
from typing import (
    Any,
    ClassVar,
    Dict,
    Generic,
    Iterable,
    List,
    Literal,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

import casadi as cs
import numpy as np
import numpy.typing as npt
from joblib import Parallel, delayed

from ..core.cache import invalidate_cache
from ..core.solutions import Solution, subsevalf
from ..nlps.nlp import Nlp
from ..nlps.objective import _solve_and_get_stats

SymType = TypeVar("SymType", cs.SX, cs.MX)


def _n(sym_name: str, scenario: int) -> str:
    """Internal utility for the naming convention of i-scenario's symbols."""
    return f"{sym_name}__{scenario}"


def _chained_subevalf(
    expr: Union[SymType, np.ndarray],
    old_vars: Dict[str, SymType],
    new_vars: Dict[str, SymType],
    old_pars: Dict[str, SymType],
    new_pars: Dict[str, SymType],
    old_dual_vars: Optional[Dict[str, SymType]] = None,
    new_dual_vars: Optional[Dict[str, SymType]] = None,
    eval: bool = True,
) -> Union[SymType, cs.DM, np.ndarray]:
    """Internal utility to perform `subevalf` on vars, pars and duals in chain."""
    expr = subsevalf(expr, old_vars, new_vars, eval=False)
    if old_dual_vars is None:
        return subsevalf(expr, old_pars, new_pars, eval=eval)
    expr = subsevalf(expr, old_pars, new_pars, eval=False)
    return subsevalf(expr, old_dual_vars, new_dual_vars, eval=eval)


class MultistartNlp(Nlp[SymType], Generic[SymType]):
    """Base class for NLP with multistarting."""

    is_multi: ClassVar[bool] = True

    def __init__(self, *args, starts: int, **kwargs) -> None:
        """Initializes the multistart NLP instance.

        Parameters
        ----------
        args, kwargs
            See inherited `csnlp.Nlp`.
        starts : int
            A positive integer for the number of multiple starting guesses to optimize.

        Raises
        ------
        ValueError
            Raises if the scenario number is invalid.
        """
        if starts <= 0:
            raise ValueError("Number of scenarios must be positive and > 0.")
        super().__init__(*args, **kwargs)
        self._starts = starts

    @property
    def starts(self) -> int:
        """Gets the number of starts."""
        return self._starts

    def solve_multi(
        self,
        pars: Union[
            None, Dict[str, npt.ArrayLike], Iterable[Dict[str, npt.ArrayLike]]
        ] = None,
        vals0: Union[
            None, Dict[str, npt.ArrayLike], Iterable[Dict[str, npt.ArrayLike]]
        ] = None,
        return_all_sols: bool = False,
        **_,
    ) -> Union[Solution[SymType], List[Solution[SymType]]]:
        """Solves the NLP with multiple initial conditions.

        Parameters
        ----------
        pars : dict[str, array_like] or iterable of, optional
            An iterable that, for each multistart, contains a dictionary with, for each
            parameter in the NLP scheme, the corresponding numerical value. In case a
            single dict is passed, the same is used across all scenarions. Can be `None`
            if no parameters are present.
        vals0 : dict[str, array_like] or iterable of, optional
            An iterable that, for each multistart, contains a dictionary with, for each
            variable in the NLP scheme, the corresponding initial guess. In case a
            single dict is passed, the same is used across all scenarions. By default
            `None`, in which case  initial guesses are not passed to the solver.
        return_all_sols : bool, optional
            If `True`, returns the solution of each multistart of the NLP; otherwise,
            only the best solution is returned. By default, `False`.

        Returns
        -------
        Solution or list of Solutions
            Depending on the flags `return_all_sols`, returns
             - the best solution out of all multiple starts
             - all the solutions (one per start)
        """
        raise NotImplementedError


class StackedMultistartNlp(MultistartNlp[SymType], Generic[SymType]):
    """A class that models and solves an NLP from multiple starting initial guesses by
    automatically stacking the problem multiple independent times in the same,
    larger-scale NLP. This allows to solve the original problem multiple times via a
    single call to the solver."""

    def __init__(self, *args, starts: int, **kwargs) -> None:
        # this class essentially is a facade that hides an internal nlp in which the
        # problem (variables, parameters, etc.) are duplicated by the requested number
        # of multiple starts. For this reason, all methods are overridden to create
        # multiples of these in the hidden nlp.
        super().__init__(*args, starts=starts, **kwargs)
        self._stacked_nlp = Nlp(*args, **kwargs)  # actual nlp

    @lru_cache(maxsize=32)
    def _vars_i(self, i: int) -> Dict[str, SymType]:
        """Internal utility to retrieve the variables of the i-th scenario."""
        nlp = self._stacked_nlp.unwrapped
        return {k: nlp._vars[_n(k, i)] for k in self._vars}

    @lru_cache(maxsize=32)
    def _pars_i(self, i: int) -> Dict[str, SymType]:
        """Internal utility to retrieve the parameters of the i-th scenario."""
        nlp = self._stacked_nlp.unwrapped
        return {k: nlp._pars[_n(k, i)] for k in self._pars}

    @lru_cache(maxsize=32)
    def _dual_vars_i(self, i: int) -> Dict[str, SymType]:
        """Internal utility to retrieve the dual variables of the i-th scenario."""
        nlp = self._stacked_nlp.unwrapped
        return {k: nlp._dual_vars[_n(k, i)] for k in self._dual_vars}

    @invalidate_cache(_pars_i)
    def parameter(self, name: str, shape: Tuple[int, int] = (1, 1)) -> SymType:
        out = super().parameter(name, shape)
        for i in range(self._starts):
            self._stacked_nlp.parameter(_n(name, i), shape)
        return out

    @invalidate_cache(_vars_i, _dual_vars_i)
    def variable(
        self,
        name: str,
        shape: Tuple[int, int] = (1, 1),
        lb: Union[npt.ArrayLike, cs.DM] = -np.inf,
        ub: Union[npt.ArrayLike, cs.DM] = +np.inf,
    ) -> Tuple[SymType, SymType, SymType]:
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
    ) -> Tuple[SymType, ...]:
        expr = lhs - rhs
        if simplify:
            expr = cs.cse(cs.simplify(expr))
        out = super().constraint(name, expr, op, 0, soft, False)

        # NOTE: the line above already created the slack variables in each scenario, so
        # below we have to pass both soft=False and the expression with slack included.
        vars = self.variables
        pars = self.parameters
        expr_ = out[0]  # slack-relaxed expression in the form h(x)<=0 or ==0
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
        self._fs: List[SymType] = [
            _chained_subevalf(
                objective, vars, self._vars_i(i), pars, self._pars_i(i), eval=False
            )
            for i in range(self._starts)
        ]
        self._stacked_nlp.minimize(sum(self._fs))
        return out

    def init_solver(
        self,
        opts: Optional[Dict[str, Any]] = None,
        solver: str = "ipopt",
        type: Literal["auto", "nlp", "conic"] = "auto",
    ) -> None:
        out = super().init_solver(opts, solver, type)
        self._stacked_nlp.init_solver(opts, solver, type)
        return out

    def solve_multi(
        self,
        pars: Union[
            None, Dict[str, npt.ArrayLike], Iterable[Dict[str, npt.ArrayLike]]
        ] = None,
        vals0: Union[
            None, Dict[str, npt.ArrayLike], Iterable[Dict[str, npt.ArrayLike]]
        ] = None,
        return_all_sols: bool = False,
        return_stacked_sol: bool = False,
        **_,
    ) -> Union[Solution[SymType], List[Solution[SymType]]]:
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

    def __call__(self, *args, **kwargs):
        return self.solve_multi(*args, **kwargs)

    def remove_variable_bounds(
        self,
        name: str,
        direction: Literal["lb", "ub", "both"],
        idx: Union[Tuple[int, int], List[Tuple[int, int]]] = None,
    ) -> None:
        idx = [idx] if isinstance(idx, tuple) else list(idx)
        super().remove_variable_bounds(name, direction, idx)
        for i in range(self._starts):
            self._stacked_nlp.remove_variable_bounds(_n(name, i), direction, idx)

    def remove_constraints(
        self, name: str, idx: Union[Tuple[int, int], List[Tuple[int, int]]] = None
    ) -> None:
        idx = [idx] if isinstance(idx, tuple) else list(idx)
        super().remove_constraints(name, idx)
        for i in range(self._starts):
            self._stacked_nlp.remove_constraints(_n(name, i), idx)


class ParallelMultistartNlp(MultistartNlp[SymType], Generic[SymType]):
    """A class that solves an NLP via parallelization of the computations."""

    def __init__(
        self, *args, starts: int, n_jobs: Optional[int] = None, **kwargs
    ) -> None:
        """Initializes the multistart NLP instance.

        Parameters
        ----------
        args, kwargs
            See inherited `csnlp.Nlp`.
        starts : int
            A positive integer for the number of multiple starting guesses to optimize.
        n_jobs : int, optional
            Number of concurrently running jobs; see `n_job` in `joblib.Parallel`.

        Raises
        ------
        ValueError
            Raises if the scenario number is invalid.
        """
        super().__init__(*args, starts=starts, **kwargs)
        self._n_jobs = n_jobs
        self._parallel = Parallel(n_jobs=n_jobs)
        self.initialize_parallel()

    def initialize_parallel(self) -> None:
        """Initializes the parallel backend."""
        self._parallel.__enter__()

    def terminate_parallel(self) -> None:
        """Terminates the parallel backend."""
        self._parallel.__exit__(None, None, None)

    def solve_multi(
        self,
        pars: Union[
            None, Dict[str, npt.ArrayLike], Iterable[Dict[str, npt.ArrayLike]]
        ] = None,
        vals0: Union[
            None, Dict[str, npt.ArrayLike], Iterable[Dict[str, npt.ArrayLike]]
        ] = None,
        return_all_sols: bool = False,
        **_,
    ) -> Union[Solution[SymType], List[Solution[SymType]]]:
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
        sols: List[Dict[str, Any]] = self._parallel(
            delayed(_solve_and_get_stats)(self._solver, kw) for kw in kwargs
        )
        if return_all_sols:
            return list(map(self._process_solver_sol, sols))

        # pick the best solution, with priority to successful ones
        best_sol = sols[0]
        best_f = float(best_sol["f"])
        best_success = best_sol["stats"]["success"]
        self._failures += not best_success
        for sol in sols[1:]:
            this_f = float(sol["f"])
            this_success = sol["stats"]["success"]
            if (not best_success and this_success) or (
                best_success == this_success and this_f < best_f
            ):
                best_sol = sol
                best_f = this_f
                best_success = this_success
            self._failures += not this_success
        return self._process_solver_sol(best_sol)

    def __getstate__(self, fullstate: bool = False) -> Dict[str, Any]:
        # joblib.Parallel cannot be pickled or deepcopied
        state = super().__getstate__(fullstate)
        state.pop("_parallel", None)
        return state

    def __setstate__(self, state: Optional[Dict[str, Any]]) -> None:
        if state is not None:
            self.__dict__.update(state)
        # re-initialized joblib.Parallel
        if not hasattr(self, "_n_jobs"):
            self._n_jobs = None
        self._parallel = Parallel(n_jobs=self._n_jobs)
        self.initialize_parallel()
