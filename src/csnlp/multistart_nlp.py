from contextlib import contextmanager
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple, Union

import numpy as np

from csnlp.nlp import (
    DMStruct,
    Nlp,
    Solution,
    cache_clearer,
    cs,
    dict2struct,
    partial,
    subsevalf,
)


def _n(sym_name: str, scenario: int) -> str:
    """Internal utility for the naming convention of i-scenario's symbols."""
    return f"{sym_name}__{scenario}"


def _get_value(x, sol: Solution, old, new, eval: bool = True):
    """Internal utility for substituting numerical values in solutions."""
    return sol._get_value(cs.substitute(x, old, new), eval=eval)


class MultistartNlp(Nlp):
    """A class to easily model and solve an NLP from multiple starting
    initial guesses in parallel. This is especially useful in case of strong
    nonlinearities, where the solver's initial conditions play a great role in
    the optimality of the solution (rather, its sub-optimality)."""

    def __init__(
        self,
        *args,
        starts: int,
        **kwargs,
    ) -> None:
        """Initializes a multistart NLP instance.

        Parameters
        ----------
        args, kwargs
            See inherited `csnlp.Nlp`.
        starts : int
            A positive integer for the number of multiple starting guesses to
            optimize over.

        Raises
        ------
        ValueError
            Raises if the scenario number is invalid.
        """
        if starts <= 0:
            raise ValueError("Number of scenarios must be positive and > 0.")
        self._starts = starts
        # this class essentially is a facade that hides an internal nlp in
        # which the problem (variables, parameters, etc.) are duplicated by the
        # requested number of multiple starts. For this reason, all methods are
        # overridden to create multiples of these in the hidden nlp
        super().__init__(*args, **kwargs)
        self._multi_nlp = Nlp(*args, **kwargs)  # actual nlp
        self._vars_per_start: Dict[int, Dict[str, Union[cs.SX, cs.MX]]] = {}

    @contextmanager
    def fullstate(self) -> None:
        with super().fullstate(), self._multi_nlp.fullstate():
            yield

    @contextmanager
    def pickleable(self) -> None:
        with super().pickleable(), self._multi_nlp.pickleable():
            yield

    @property
    def starts(self) -> int:
        """Gets the number of starts."""
        return self._starts

    @lru_cache
    def _symbols(
        self,
        i: Optional[int] = None,
        vars: bool = False,
        pars: bool = False,
        dual: bool = False,
    ) -> Dict[str, Union[cs.SX, cs.MX]]:
        """Internal utility to retrieve the symbols of the i-th scenario."""
        S: Dict[str, Union[cs.SX, cs.MX]] = {}
        if vars:
            S.update(
                self._vars
                if i is None
                else {k: self._multi_nlp.unwrapped._vars[_n(k, i)] for k in self._vars}
            )
        if pars:
            S.update(
                self._pars
                if i is None
                else {k: self._multi_nlp.unwrapped._pars[_n(k, i)] for k in self._pars}
            )
        if dual:
            S.update(
                self._dual_vars
                if i is None
                else {
                    k: self._multi_nlp.unwrapped._dual_vars[_n(k, i)]
                    for k in self._dual_vars
                }
            )
        return S

    @cache_clearer(_symbols)
    def parameter(
        self, name: str, shape: Tuple[int, int] = (1, 1)
    ) -> Union[cs.SX, cs.MX]:
        out = super().parameter(name, shape)
        for i in range(self._starts):
            self._multi_nlp.parameter(_n(name, i), shape)
        return out

    @cache_clearer(_symbols)
    def variable(
        self,
        name: str,
        shape: Tuple[int, int] = (1, 1),
        lb: Union[np.ndarray, cs.DM] = -np.inf,
        ub: Union[np.ndarray, cs.DM] = +np.inf,
    ) -> Union[Tuple[cs.SX, cs.SX, cs.SX], Tuple[cs.MX, cs.MX, cs.MX]]:
        out = super().variable(name, shape, lb, ub)
        for i in range(self._starts):
            self._multi_nlp.variable(_n(name, i), shape, lb, ub)
        return out

    @cache_clearer(_symbols)
    def constraint(
        self,
        name: str,
        lhs: Union[np.ndarray, cs.DM, cs.SX, cs.MX],
        op: Literal["==", ">=", "<="],
        rhs: Union[np.ndarray, cs.DM, cs.SX, cs.MX],
        soft: bool = False,
        simplify: bool = True,
    ) -> Union[
        Tuple[cs.SX, cs.SX],
        Tuple[cs.MX, cs.MX],
        Tuple[cs.SX, cs.SX, cs.SX],
        Tuple[cs.MX, cs.MX, cs.MX],
    ]:
        expr = lhs - rhs
        if simplify:
            expr = cs.simplify(expr)
        out = super().constraint(name, expr, op, 0, soft, simplify=False)

        symbols = self._symbols(vars=True, pars=True)
        for i in range(self._starts):
            symbols_i = self._symbols(i, vars=True, pars=True)
            expr = subsevalf(expr, symbols, symbols_i, eval=False)
            self._multi_nlp.constraint(_n(name, i), expr, op, 0, soft, simplify=False)
        return out

    def minimize(self, objective: Union[cs.SX, cs.MX]) -> None:
        out = super().minimize(objective)
        symbols = self._symbols(vars=True, pars=True)
        self._fs: Union[List[cs.SX], List[cs.MX]] = [
            subsevalf(
                objective, symbols, self._symbols(i, vars=True, pars=True), eval=False
            )
            for i in range(self._starts)
        ]
        self._multi_nlp.minimize(sum(self._fs))
        return out

    def init_solver(self, opts: Optional[Dict[str, Any]] = None) -> None:
        out = super().init_solver(opts)
        self._multi_nlp.init_solver(opts)
        return out

    def solve_multi(
        self,
        pars: Union[None, Iterable[DMStruct], Iterable[Dict[str, np.ndarray]]] = None,
        vals0: Union[None, Iterable[DMStruct], Iterable[Dict[str, np.ndarray]]] = None,
        return_all_sols: bool = False,
        return_multi_sol: bool = False,
    ) -> Union[Solution, List[Solution]]:
        """Solves the NLP with multiple initial conditions.

        Parameters
        ----------
        pars : iterable of DMStruct, dict[str, array_like], optional
            An iterable that, for each multistart, contains a dictionary or
            structure containing, for each parameter in the NLP scheme, the
            corresponding numerical value. Can be `None` if no parameters are
            present.
        vals0 : iterable of DMStruct, dict[str, array_like], optional
            An iterable that, for each multistart, contains a dictionary or
            structure containing, for each variable in the NLP scheme, the
            corresponding initial guess. By default, initial guesses are not
            passed to the solver.
        return_all_sols : bool, optional
            If `True`, returns the solution of each multistart of the NLP;
            otherwise, only the best solution is returned. By default, `False`.
        return_multi_sol : bool, optional
            If `True`, returns the solution of the underlying multistart NLP.
            Generally, only for debugging. By default, `False`.

        Returns
        -------
        Solution or list of Solutions
            Depending on the flags `return_all_sols` and `return_multi_sol`,
            returns
                - the best solution out of all multiple starts
                - all the solutions (one per start)
                - the solution to the underlying (hidden) multistart NLP.

        Raises
        ------
        ValueError
            Raises if `return_multi_sol` and `return_all_sols` are both true at
            the same time.
        """
        if return_multi_sol and return_all_sols:
            raise ValueError(
                "`return_multi_sol` and `return_all_sols` can't be both true."
            )

        if pars is not None:
            pars = {
                _n(n, i): pars_i[n]
                for i, pars_i in enumerate(pars)
                for n in pars_i.keys()
            }
        if vals0 is not None:
            vals0 = {
                _n(n, i): vals0_i[n]
                for i, vals0_i in enumerate(vals0)
                for n in vals0_i.keys()
            }
        multi_sol = self._multi_nlp.solve(pars=pars, vals0=vals0)
        if return_multi_sol:
            return multi_sol

        vars_ = self.variables
        symbols = self._symbols(vars=True, pars=True, dual=True)
        old = cs.vertcat(
            self._p, self._x, self._lam_g, self._lam_h, self._lam_lbx, self._lam_ubx
        )

        sols: List[Solution] = []
        fs = [float(multi_sol.value(f)) for f in self._fs]
        idx = range(self._starts) if return_all_sols else (np.argmin(fs),)
        for i in idx:
            vals = {n: multi_sol.vals[_n(n, i)] for n in vars_.keys()}

            symbols_i = self._symbols(i, vars=True, pars=True, dual=True)
            new = subsevalf(old, symbols, symbols_i, eval=False)
            get_value = partial(_get_value, sol=multi_sol, old=old, new=new)

            sols.append(
                Solution(
                    f=fs[i],
                    vars=vars_,
                    vals=dict2struct(vals),
                    stats=multi_sol.stats,
                    _get_value=get_value,
                )
            )
        return sols if return_all_sols else sols[0]
