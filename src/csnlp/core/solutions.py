"""Contains classes and methods to store the solution of an NLP problem after a call
to :meth:`csnlp.Nlp.solve` or :meth:`csnlp.multistart.MultistartNlp.solve_multi`."""

from collections.abc import Iterable as _Iterable
from functools import cached_property as _cached_property
from itertools import product as _product
from typing import TYPE_CHECKING
from typing import Any as _Any
from typing import Protocol
from typing import TypeVar as _TypeVar
from typing import Union

import casadi as cs
import numpy as np
from casadi.tools.structure3 import CasadiStructured as _CasadiStructured
from casadi.tools.structure3 import DMStruct as _DMStruct

from .data import array2cs as _array2cs
from .data import cs2array as _cs2array

if TYPE_CHECKING:
    from ..nlps.nlp import Nlp

SymType = _TypeVar("SymType", cs.SX, cs.MX)


class Solution(Protocol[SymType]):
    """Class containing information on the solution of a solver's run for an instance of
    :class:`csnlp.Nlp`.

    Notes
    -----
    This class is merely a protocol, and as such it just defines the interface of how a
    solution should look like, plus some minor implementations.
    """

    @property
    def f(self) -> float:
        """Optimal value of the objective function."""
        ...

    @property
    def vars(self) -> dict[str, SymType]:
        """Symbolical primal variables."""
        return self._vars

    @property
    def vals(self) -> dict[str, cs.DM]:
        """Optimal values of the primal variables."""
        ...

    @property
    def dual_vars(self) -> dict[str, SymType]:
        """Symbolical dual variables."""
        return self._dual_vars

    @property
    def dual_vals(self) -> dict[str, cs.DM]:
        """Optimal values of the dual variables."""
        ...

    @property
    def primal_dual_par_vars(self) -> SymType:
        """Symbolical primal and dual variables and parameters in a vector."""
        ...

    @property
    def primal_dual_par_vals(self) -> cs.DM:
        """Optimal values of the primal and dual variables, as well as the parameters
        for which the solution was found, in a vector."""
        ...

    @property
    def stats(self) -> dict[str, _Any]:
        """Statistics of the solver for this solution's run."""
        return self._stats

    @property
    def status(self) -> str:
        """Gets the status of the solver at this solution."""
        return self.stats["return_status"]

    @property
    def success(self) -> bool:
        """Gets whether the solver's run was successful."""
        return self.stats["success"]

    @property
    def barrier_parameter(self) -> float:
        """Gets the IPOPT barrier parameter at the optimal solution"""
        return self.stats["iterations"]["mu"][-1]

    def value(
        self, expr: Union[SymType, np.ndarray], eval: bool = True
    ) -> Union[SymType, cs.DM]:
        """Computes the value of the expression substituting the values of this
        solution in the expression.

        Parameters
        ----------
        expr : casadi.SX, MX or an array of these
            The symbolic expression to be evaluated at the solution's values.
        eval : bool, optional
            Evaluates numerically the new expression. By default, ``True``. See
            :meth:`csnlp.solutions.subsevalf` for more details.

        Returns
        -------
        casadi.SX or MX or DM
            The expression evaluated with the solution's values.

        Raises
        ------
        RuntimeError
            Raises if ``eval=True`` but there are symbolic variables that are still
            free. This can occur when there are symbols that are outside the solution's
            variables, and thus have not been substituted by a numerical value.
        """
        return subsevalf(
            expr, self.primal_dual_par_vars, self.primal_dual_par_vals, eval=eval
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(f={self.f},success={self.success},"
            f"status={self.status})"
        )

    def __lt__(self, other: "Solution[SymType]") -> bool:
        return self.f < other.f


class EagerSolution(Solution[SymType]):
    """Class containing information on the solution of a solver's run for an instance of
    :class:`csnlp.Nlp`.

    Parameters
    ----------
    f : float
        Optimal value of the NLP at the solution.
    vars : dict of (str, cs.SX or MX)
        Symbolic primal variables.
    vals : dict of (str, cs.DM)
        Optimal values of the primal variables.
    dual_vars : dict of (str, cs.SX or MX)
        Symbolic dual variables.
    dual_vals : dict of (str, cs.DM)
        Optimal values of the dual variables.
    primal_dual_par_vars : cs.SX or MX
        Symbolic primal and dual variables and parameters in a vector.
    primal_dual_par_vals : cs.DM
        Optimal values of the primal and dual variables, as well as the parameters for
        which the solution was found, in a vector.
    stats : dict
        Stats of the solver run that generated this solution.

    Notes
    -----
    This class is an eager implementation of the :class:`csnlp.solutions.Solution`,
    where the values and properties of the solution are computed eagerly after the
    solver has finished its run.
    """

    def __init__(
        self,
        f: float,
        vars: dict[str, SymType],
        vals: dict[str, cs.DM],
        dual_vars: dict[str, SymType],
        dual_vals: dict[str, cs.DM],
        primal_dual_par_vars: SymType,
        primal_dual_par_vals: cs.DM,
        stats: dict[str, _Any],
    ) -> None:
        self._f = f
        self._vars = vars
        self._vals = vals
        self._dual_vars = dual_vars
        self._dual_vals = dual_vals
        self._primal_dual_par_vars = primal_dual_par_vars
        self._primal_dual_par_vals = primal_dual_par_vals
        self._stats = stats

    @property
    def f(self) -> float:
        return self._f

    @property
    def vals(self) -> dict[str, cs.DM]:
        return self._vals

    @property
    def dual_vals(self) -> dict[str, cs.DM]:
        return self._dual_vals

    @property
    def primal_dual_par_vars(self) -> SymType:
        return self._primal_dual_par_vars

    @property
    def primal_dual_par_vals(self) -> cs.DM:
        return self._primal_dual_par_vals

    @staticmethod
    def from_casadi_solution(
        sol_with_stats: dict[str, _Any], nlp: "Nlp[SymType]"
    ) -> "EagerSolution[SymType]":
        """Creates a new eager solution from a CasADi solution,

        Parameters
        ----------
        sol_with_stats : dict of (str, cs.DM) and one entry with stats, i.e., Any
            The solution dictionary from the CasADi solver, which contains the optimal
            values of the primal and dual variables, as well as the parameters, and the
            solver's statistics.
        nlp : Nlp[SymType]
            The NLP instance for which the solution was computed.

        Returns
        -------
        EagerSolution[SymType]
            The eager solution corresponding to the CasADi solution.

        Raises
        ------
        RuntimeError
            Raises if a dual variable type is not recognized.
        """
        # stats and objective
        stats = sol_with_stats.pop("stats")
        sol: dict[str, cs.DM] = sol_with_stats  # now the solution has only cs.DMs
        f = float(sol["f"])

        # primal variables and values
        vars = nlp.variables
        vals = {name: subsevalf(var, nlp._x, sol["x"]) for name, var in vars.items()}

        # dual variables and values
        lam_g = sol["lam_g"][: nlp.ng, :]
        lam_h = sol["lam_g"][nlp.ng :, :]
        lam_lbx = -cs.fmin(sol["lam_x"][nlp.nonmasked_lbx_idx, :], 0)
        lam_ubx = cs.fmax(sol["lam_x"][nlp.nonmasked_ubx_idx, :], 0)
        dual_vars = nlp.dual_variables
        dual_vals = {}
        for n, var in dual_vars.items():
            if n.startswith("lam_g"):
                dual_vals[n] = subsevalf(var, nlp._lam_g, lam_g)
            elif n.startswith("lam_h"):
                dual_vals[n] = subsevalf(var, nlp._lam_h, lam_h)
            elif n.startswith("lam_lb"):
                dual_vals[n] = subsevalf(var, nlp._lam_lbx, lam_lbx)
            elif n.startswith("lam_ub"):
                dual_vals[n] = subsevalf(var, nlp._lam_ubx, lam_ubx)
            else:
                raise RuntimeError(f"unknown dual variable type {n}")

        # get_value function
        primal_dual_pars_vars = cs.vertcat(
            nlp._x, nlp._lam_g, nlp._lam_h, nlp._lam_lbx, nlp._lam_ubx, nlp._p
        )
        primal_dual_pars_vals = cs.vertcat(
            sol["x"], lam_g, lam_h, lam_lbx, lam_ubx, sol["p"]
        )
        return EagerSolution(
            f,
            vars,
            vals,
            dual_vars,
            dual_vals,
            primal_dual_pars_vars,
            primal_dual_pars_vals,
            stats,
        )


class LazySolution(Solution[SymType]):
    """Class containing information on the solution of a solver's run for an instance of
    :class:`csnlp.Nlp`.

    Parameters
    ----------
    sol_with_stats : dict of (str, cs.DM) and one entry with stats, i.e., Any
        The solution dictionary from the CasADi solver, which contains the optimal
        values of the primal and dual variables, as well as the parameters, and the
        solver's statistics.
    nlp : Nlp[SymType]
        The NLP instance for which the solution was computed.

    Notes
    -----
    This class is a lazy implementation of the :class:`csnlp.solutions.Solution`, where
    the values and properties of the solution are computed lazily only when requested.
    """

    def __init__(self, sol_with_stats: dict[str, _Any], nlp: "Nlp[SymType]") -> None:
        self._stats = sol_with_stats.pop("stats")
        self._sol: dict[str, cs.DM] = sol_with_stats  # now the solution has only cs.DMs

        # while we want to be as lazy as possible, we need to peform some copies in case
        # the nlp instance is modified after the solution is created
        self._vars = nlp.variables.copy()
        self._dual_vars = nlp.dual_variables.copy()
        self._x = nlp.x
        self._p = nlp.p
        self._lam_g = nlp._lam_g
        self._lam_h = nlp._lam_h
        self._lam_lbx = nlp._lam_lbx
        self._lam_ubx = nlp._lam_ubx
        lbx_idx = nlp.nonmasked_lbx_idx
        self._nonmasked_lbx_idx = (
            lbx_idx.copy() if isinstance(lbx_idx, np.ndarray) else lbx_idx
        )
        ubx_idx = nlp.nonmasked_ubx_idx
        self._nonmasked_ubx_idx = (
            ubx_idx.copy() if isinstance(ubx_idx, np.ndarray) else ubx_idx
        )

    @property
    def original_solution(self) -> dict[str, cs.DM]:
        """Gets the original solution dictionary from the solver."""
        return self._sol

    @_cached_property
    def f(self) -> float:
        return float(self._sol["f"])

    @_cached_property
    def vals(self) -> dict[str, cs.DM]:
        x = self._x
        x_vals = self._sol["x"]
        return {n: cs.evalf(cs.substitute(v, x, x_vals)) for n, v in self._vars.items()}

    @_cached_property
    def grouped_dual_vals(self) -> tuple[cs.DM, cs.DM, cs.DM, cs.DM]:
        """Optimal values of the dual variables."""
        ng = self._lam_g.shape[0]
        lam_g = self._sol["lam_g"][:ng, :]
        lam_h = self._sol["lam_g"][ng:, :]
        lam_lbx = -cs.fmin(self._sol["lam_x"][self._nonmasked_lbx_idx, :], 0)
        lam_ubx = cs.fmax(self._sol["lam_x"][self._nonmasked_ubx_idx, :], 0)
        return lam_g, lam_h, lam_lbx, lam_ubx

    @_cached_property
    def dual_vals(self) -> dict[str, cs.DM]:
        lam_g, lam_h, lam_lbx, lam_ubx = self.grouped_dual_vals
        dual_vals = {}
        for n, v in self._dual_vars.items():
            if n.startswith("lam_g"):
                dual_vals[n] = cs.evalf(cs.substitute(v, self._lam_g, lam_g))
            elif n.startswith("lam_h"):
                dual_vals[n] = cs.evalf(cs.substitute(v, self._lam_h, lam_h))
            elif n.startswith("lam_lb"):
                dual_vals[n] = cs.evalf(cs.substitute(v, self._lam_lbx, lam_lbx))
            elif n.startswith("lam_ub"):
                dual_vals[n] = cs.evalf(cs.substitute(v, self._lam_ubx, lam_ubx))
            else:
                raise RuntimeError(f"unknown dual variable type `{n}`")
        return dual_vals

    @_cached_property
    def primal_dual_par_vars(self) -> SymType:
        return cs.vertcat(
            self._x, self._lam_g, self._lam_h, self._lam_lbx, self._lam_ubx, self._p
        )

    @_cached_property
    def primal_dual_par_vals(self) -> cs.DM:
        return cs.vertcat(self._sol["x"], *self.grouped_dual_vals, self._sol["p"])

    def to_eager(self) -> EagerSolution[SymType]:
        """Converts this lazy solution to an eager solution.

        Returns
        -------
        EagerSolution[SymType]
            The eager solution corresponding to this lazy solution.
        """
        return EagerSolution(
            self.f,
            self._vars,
            self.vals,
            self._dual_vars,
            self.dual_vals,
            self.primal_dual_par_vars,
            self.primal_dual_par_vals,
            self._stats,
        )


def _internal_subsevalf_cs(
    expr: SymType,
    old: Union[SymType, dict[str, SymType], _Iterable[SymType]],
    new: Union[SymType, dict[str, SymType], _Iterable[SymType]],
    eval: bool,
) -> Union[SymType, cs.DM]:
    """Internal utility for substituting and evaluting casadi objects."""
    if isinstance(expr, (cs.DM, _DMStruct)):
        return expr

    if isinstance(old, dict):
        for name, o in old.items():
            expr = cs.substitute(expr, o, new[name])  # type: ignore[index]
    elif isinstance(old, _Iterable) and not isinstance(
        old, (cs.SX, cs.MX, _CasadiStructured)
    ):
        for o, n in zip(old, new):
            expr = cs.substitute(expr, o, n)
    else:
        expr = cs.substitute(expr, old, new)

    if eval:
        expr = cs.evalf(expr)
    return expr


def _internal_subsevalf_np(
    expr: np.ndarray,
    old: Union[SymType, dict[str, SymType], _Iterable[SymType]],
    new: Union[SymType, dict[str, SymType], _Iterable[SymType]],
    eval: bool,
) -> Union[SymType, np.ndarray, cs.DM]:
    """Internal utility for substituting and evaluting arrays of casadi objects."""
    # if not symbolic, return it
    if expr.dtype != object:
        return expr

    # up to 2D, we can get away with only one substitution
    if expr.ndim <= 2:
        return _internal_subsevalf_cs(_array2cs(expr), old, new, eval)

    # for tensors, check if the right end or the left is bigger, then loop over the
    # rest while substituing in the right/left end of the array
    shape = expr.shape
    transposed = (shape[0] * shape[1]) > (shape[-2] * shape[-1])
    if transposed:
        shape = shape[::-1]
        expr = expr.transpose()
    out = np.empty(shape, object)
    shape_iter, shape_cs = shape[:-2], shape[-2:]
    for i in _product(*map(range, shape_iter)):
        out[i] = _cs2array(
            _internal_subsevalf_cs(_array2cs(expr[i]), old, new, eval)
        ).reshape(shape_cs)
    if transposed:
        out = out.transpose()
        expr = expr.transpose()

    if eval:
        out = out.astype(float)
    return out


def subsevalf(
    expr: Union[SymType, np.ndarray],
    old: Union[SymType, dict[str, SymType], _Iterable[SymType]],
    new: Union[SymType, dict[str, SymType], _Iterable[SymType]],
    eval: bool = True,
) -> Union[SymType, cs.DM, np.ndarray]:
    """Substitutes the old variables with the new ones in the symbolic expression,
    and evaluates it, if required.

    Parameters
    ----------
    expr : casadi.SX, MX or an array of these
        Expression for substitution and, possibly, evaluation.
    old : casadi.SX or MX or struct, dict, iterable of these
        Old variable to be substituted.
    new : numpy.array or casadi.SX, MX, DM or struct, dict, iterable of these
        New variable that substitutes the old one. If a collection, it is
        assumed the type is the same of ``old`` (so, old and new should share
        collection type).
    eval : bool, optional
        Evaluates numerically the new expression. By default, ``True``.

    Returns
    -------
    new_expr : casadi.SX, MX, DM or an array of these
        New expression after substitution (SX, MX) and, possibly, evaluation
        (DM).

    Raises
    ------
    Exception
        Raises if ``old`` and ``new`` are not compatible with :func:`casadi.substitute`;
        or if ``eval=True`` but there are symbolic variables that are still free, i.e.,
        the expression cannot be evaluated numerically since it is still (partially)
        symbolic.
    """
    if isinstance(expr, np.ndarray):
        return _internal_subsevalf_np(expr, old, new, eval)
    else:
        return _internal_subsevalf_cs(expr, old, new, eval)
