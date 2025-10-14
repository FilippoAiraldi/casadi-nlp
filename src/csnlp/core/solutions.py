"""Contains classes and methods to store the solution of an NLP problem after a call
to :meth:`csnlp.Nlp.solve` or :meth:`csnlp.multistart.MultistartNlp.solve_multi`."""

from collections.abc import Iterable as _Iterable
from functools import cached_property as _cached_property
from itertools import product as _product
from typing import TYPE_CHECKING, Optional, Union
from typing import Any as _Any
from typing import Protocol as _Protocol
from typing import TypeVar as _TypeVar

import casadi as cs
import numpy as np
import numpy.typing as npt

from .data import array2cs as _array2cs
from .data import cs2array as _cs2array

if TYPE_CHECKING:
    from ..nlps.nlp import Nlp

SymType = _TypeVar("SymType", cs.SX, cs.MX)
SymOrNumType = _TypeVar("SymOrNumType", cs.SX, cs.MX, cs.DM, int, float, np.ndarray)


def _is_infeas(status: str, solver_plugin: str) -> Optional[bool]:
    """Internal utility to compute whether the solver status indicates infeasibility."""
    # NLPs
    if solver_plugin == "ipopt":
        return status == "Infeasible_Problem_Detected"
    if solver_plugin in ("qrsqp", "sqpmethod"):
        return status == "Search_Direction_Becomes_Too_Small"
    # QPs
    if solver_plugin == "osqp":
        return "infeasible" in status
    if solver_plugin == "proxqp":
        return (
            status == "PROXQP_PRIMAL_INFEASIBLE" or status == "PROXQP_DUAL_INFEASIBLE"
        )
    if solver_plugin == "qpoases":
        return "infeasib" in status
    if solver_plugin == "qrqp":
        return status == "Failed to calculate search direction"
    # LPs
    if solver_plugin == "clp":
        return status.endswith("infeasible")
    # MIPs
    if solver_plugin in ("bonmin", "gurobi"):
        return status == "INFEASIBLE"
    if solver_plugin == "cbc":
        return "not feasible" in status
    if solver_plugin == "gurobi":
        return status == "INFEASIBLE" or status == "INF_OR_UNBD"
    if solver_plugin == "knitro":
        return "INFEAS" in status
    return None


class Solution(_Protocol[SymType]):
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

    @property
    def p_sym(self) -> cs.DM:
        """Symbolical  parameters in a vector."""

    @property
    def p(self) -> cs.DM:
        """Values of the parameters for which this solver's solution was generated."""

    @property
    def x_sym(self) -> SymType:
        """Symbolical primal variables in a vector."""

    @property
    def x(self) -> cs.DM:
        """Optimal values of the primal variables in a vector."""

    @property
    def lam_g_and_h_sym(self) -> cs.DM:
        """Symbolical the equality and inequality dual variables in a vector."""

    @property
    def lam_g_and_h(self) -> cs.DM:
        """Optimal values of the equality and inequality dual variables in a vector."""

    @property
    def lam_lbx_and_ubx_sym(self) -> cs.DM:
        """Symbolical lower- and upper-bounds dual variables in a vector."""

    @property
    def lam_lbx_and_ubx(self) -> cs.DM:
        """Optimal values of the lower- and upper-bounds dual variables in a vector."""

    @_cached_property
    def x_and_lam_and_p_sym(self) -> SymType:
        """Symbolical primal and dual variables and parameters in a vector."""
        return cs.vertcat(
            self.x_sym,
            self.lam_g_and_h_sym,
            self.lam_lbx_and_ubx_sym,
            self.p_sym,
        )

    @_cached_property
    def x_and_lam_and_p(self) -> cs.DM:
        """Optimal values of the primal and dual variables, as well as the parameters
        for which the solution was found, in a vector."""
        return cs.vertcat(
            self.x,
            self.lam_g_and_h,
            self.lam_lbx_and_ubx,
            self.p,
        )

    @property
    def vars(self) -> dict[str, SymType]:
        """Symbolical primal variables."""
        return self._vars

    @property
    def vals(self) -> dict[str, cs.DM]:
        """Optimal values of the primal variables."""

    @property
    def dual_vars(self) -> dict[str, SymType]:
        """Symbolical dual variables."""
        return self._dual_vars

    @property
    def dual_vals(self) -> dict[str, cs.DM]:
        """Optimal values of the dual variables."""

    @property
    def solver_plugin(self) -> str:
        """The solver plugin used to generate this solution."""
        return self._solver_plugin

    @property
    def stats(self) -> dict[str, _Any]:
        """Statistics of the solver for this solution's run."""
        return self._stats

    @property
    def status(self) -> str:
        """Gets the status of the solver at this solution."""
        return self.stats["return_status"]

    @property
    def unified_return_status(self) -> str:
        """Gets the unified status of the solver at this solution."""
        return self.stats["unified_return_status"]

    @property
    def success(self) -> bool:
        """Gets whether the solver's run was successful."""
        return self.stats["success"]

    @_cached_property
    def infeasible(self) -> Optional[bool]:
        r"""Gets whether the solver status indicates infeasibility. If ``False``, it
        does not imply feasibility as the solver or its CasADi interface may have not
        detect it.

        For different solvers, the infeasibility status is stored in different ways.
        Here is a list of what I gathered so far. The solvers are grouped based on the
        type of problem they solve. An (F) next to the solver's name indicates that the
        the solver will crash the program if ``"error_on_fail": True`` and the solver
        fails. The ``status`` and ``unified_return_status`` can both be found in
        the solver's stats, or in this solution object.

        * NLPs

          - **fatrop**: unclear; better to return ``None`` for now
          - **ipopt**: ``status == "Infeasible_Problem_Detected"``
          - **qrsqp** (F): ``status == "Search_Direction_Becomes_Too_Small"``
            (dubious)
          - **sqpmethod** (F): ``status == "Search_Direction_Becomes_Too_Small"``
            (dubious)

        * QPs

          - **ipqp** (F): no clear way to detect infeasibility; return ``None`` for now
          - **osqp** (F): ``unified_return_status == "SOLVER_RET_INFEASIBLE"`` or
            ``"infeasible" in status``
          - **proxqp** (F): ``status == "PROXQP_PRIMAL_INFEASIBLE"`` or
            ``status == "PROXQP_DUAL_INFEASIBLE"``
          - **qpoases** (F): ``"infeasib" in status``
          - **qrqp** (F): ``status == "Failed to calculate search direction"``

        * LPs

          - **clp** (F): ``status == "primal infeasible"`` or
            ``status == "dual infeasible"``

        * Mixed-Iteger Problems (MIPs)

          - **bonmin** (F): ``status == "INFEASIBLE"``
          - **cbc** (F): ``"not feasible" in status``
          - **gurobi** (F): ``status == "INFEASIBLE"`` or ``status == "INF_OR_UNBD"``
          - **knitro**: ``"INFEAS" in status``
        """
        return _is_infeas(self.status, self.solver_plugin)

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
            expr, self.x_and_lam_and_p_sym, self.x_and_lam_and_p, eval=eval
        )

    @staticmethod
    def from_casadi_solution(
        sol_with_stats: dict[str, _Any], nlp: "Nlp[SymType]"
    ) -> "Solution[SymType]":
        """Creates a new solution from a CasADi solution,

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
        Solution[SymType]
            The solution corresponding to the CasADi solution.
        """

    @staticmethod
    def cmp_key(sol: "Solution[SymType]") -> tuple[bool, bool, float]:
        """Gets the comparison keys to compare a solution with another. This solution is
        strictly better if

        - it is feasible and the other is not, or
        - both are feasible or infeasible, and the current is successful and the other
            is not, or
        - both are successful or not, and the current has a lower optimal value than the
            other.

        To be used as ``key`` argument in, e.g., :func:`min` or :func:`sorted`.

        Returns
        -------
        tuple of (bool, bool, float)
            A tuple with (is_infeasible, is_unsuccessful, f).
        """
        is_infeas = _is_infeas(sol.status, sol.solver_plugin)
        if is_infeas is None:
            is_infeas = "infeas" in sol.status.lower()
        return is_infeas, not sol.success, sol.f

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(f={self.f},success={self.success},"
            f"status={self.status})"
        )


class EagerSolution(Solution[SymType]):
    """Class containing information on the solution of a solver's run for an instance of
    :class:`csnlp.Nlp`.

    Parameters
    ----------
    f : float
        Optimal value of the NLP at the solution.
    p_sym : cs.SX or MX
        Symbolic parameters in a vector.
    p : cs.DM
        Values of the parameters for which this solver's solution was generated.
    x_sym : SymType
        Symbolic primal variables in a vector.
    x : cs.DM
        Optimal values of the primal variables in a vector.
    lam_g_and_h_sym : cs.SX or MX
        Symbolic the equality and inequality dual variables in a vector.
    lam_g_and_h : cs.DM
        Optimal values of the equality and inequality dual variables in a vector.
    lam_lbx_and_ubx_sym : cs.SX or MX
        Symbolic lower- and upper-bounds dual variables in a vector.
    lam_lbx_and_ubx : cs.DM
        Optimal values of the lower- and upper-bounds dual variables in a vector.
    vars : dict of (str, cs.SX or MX)
        Symbolic primal variables.
    vals : dict of (str, cs.DM)
        Optimal values of the primal variables.
    dual_vars : dict of (str, cs.SX or MX)
        Symbolic dual variables.
    dual_vals : dict of (str, cs.DM)
        Optimal values of the dual variables.
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
        p_sym: SymType,
        p: cs.DM,
        x_sym: SymType,
        x: cs.DM,
        lam_g_and_h_sym: SymType,
        lam_g_and_h: cs.DM,
        lam_lbx_and_ubx_sym: SymType,
        lam_lbx_and_ubx: cs.DM,
        vars: dict[str, SymType],
        vals: dict[str, cs.DM],
        dual_vars: dict[str, SymType],
        dual_vals: dict[str, cs.DM],
        stats: dict[str, _Any],
        solver_plugin: str,
    ) -> None:
        self._f = f

        self._p_sym = p_sym
        self._p = p
        self._x_sym = x_sym
        self._x = x
        self._lam_g_and_h_sym = lam_g_and_h_sym
        self._lam_g_and_h = lam_g_and_h
        self._lam_lbx_and_ubx_sym = lam_lbx_and_ubx_sym
        self._lam_lbx_and_ubx = lam_lbx_and_ubx

        self._vars = vars
        self._vals = vals
        self._dual_vars = dual_vars
        self._dual_vals = dual_vals

        self._stats = stats
        self._solver_plugin = solver_plugin

    @property
    def f(self) -> float:
        return self._f

    @property
    def p_sym(self) -> cs.DM:
        return self._p_sym

    @property
    def p(self) -> cs.DM:
        return self._p

    @property
    def x_sym(self) -> SymType:
        return self._x_sym

    @property
    def x(self) -> cs.DM:
        return self._x

    @property
    def lam_g_and_h_sym(self) -> cs.DM:
        return self._lam_g_and_h_sym

    @property
    def lam_g_and_h(self) -> cs.DM:
        return self._lam_g_and_h

    @property
    def lam_lbx_and_ubx_sym(self) -> cs.DM:
        return self._lam_lbx_and_ubx_sym

    @property
    def lam_lbx_and_ubx(self) -> cs.DM:
        return self._lam_lbx_and_ubx

    @property
    def vals(self) -> dict[str, cs.DM]:
        return self._vals

    @property
    def dual_vals(self) -> dict[str, cs.DM]:
        return self._dual_vals

    @staticmethod
    def from_casadi_solution(
        sol_with_stats: dict[str, _Any], nlp: "Nlp[SymType]"
    ) -> "EagerSolution[SymType]":
        # stats and objective
        stats = sol_with_stats.pop("stats")
        sol: dict[str, cs.DM] = sol_with_stats  # now the solution has only cs.DMs
        f = float(sol["f"])

        # primal variables and values
        x = nlp._x
        x_opt = sol["x"]
        vars = nlp._vars.copy()
        vals = {name: subsevalf(var, x, x_opt) for name, var in vars.items()}

        # dual variables and values
        lam_g_and_h = sol["lam_g"]
        lam_g = lam_g_and_h[: nlp.ng, :]
        lam_h = lam_g_and_h[nlp.ng :, :]
        lam_x = sol["lam_x"]
        lam_lbx = -cs.fmin(lam_x[nlp.nonmasked_lbx_idx, :], 0)
        lam_ubx = cs.fmax(lam_x[nlp.nonmasked_ubx_idx, :], 0)
        dual_vars = nlp._dual_vars.copy()
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

        return EagerSolution(
            f,
            nlp._p,
            sol["p"],
            x,
            x_opt,
            cs.vertcat(nlp._lam_g, nlp._lam_h),
            lam_g_and_h,
            cs.vertcat(nlp._lam_lbx, nlp._lam_ubx),
            cs.vertcat(lam_lbx, lam_ubx),
            vars,
            vals,
            dual_vars,
            dual_vals,
            stats,
            nlp.unwrapped._solver_plugin,
        )


class LazySolution(Solution[SymType]):
    """Class containing information on the solution of a solver's run for an instance of
    :class:`csnlp.Nlp`.

    Parameters
    ----------
    f : float
        Optimal value of the NLP at the solution.
    p_sym : cs.SX or MX
        Symbolic parameters in a vector.
    x_sym : SymType
        Symbolic primal variables in a vector.
    lam_g_and_h_sym : cs.SX or MX
        Symbolic the equality and inequality dual variables in a vector.
    lam_lbx_and_ubx_sym : cs.SX or MX
        Symbolic lower- and upper-bounds dual variables in a vector.
    vars : dict of (str, cs.SX or MX)
        Symbolic primal variables.
    dual_vars : dict of (str, cs.SX or MX)
        Symbolic dual variables.
    nonmasked_lbx_idx : slice or 1D array of int
        Indexes of the non-masked (i.e., valid, finite) lower-bounds.
    nonmasked_ubx_idx : slice or 1D array of int
        Indexes of the non-masked (i.e., valid, finite) upper-bounds.
    stats : dict
        Stats of the solver run that generated this solution.

    Notes
    -----
    This class is a lazy implementation of the :class:`csnlp.solutions.Solution`, where
    the values and properties of the solution are computed lazily only when requested.
    """

    def __init__(
        self,
        solution: dict[str, cs.DM],
        p_sym: SymType,
        x_sym: SymType,
        lam_g_and_h_sym: SymType,
        lam_lbx_and_ubx_sym: SymType,
        vars: dict[str, SymType],
        dual_vars: dict[str, SymType],
        nonmasked_lbx_idx: Union[slice, npt.NDArray[np.int64]],
        nonmasked_ubx_idx: Union[slice, npt.NDArray[np.int64]],
        stats: dict[str, _Any],
        solver_plugin: str,
    ) -> None:
        self._sol = solution
        self._p_sym = p_sym
        self._x_sym = x_sym
        self._lam_g_and_h_sym = lam_g_and_h_sym
        self._lam_lbx_and_ubx_sym = lam_lbx_and_ubx_sym
        self._vars = vars
        self._dual_vars = dual_vars
        self._nonmasked_lbx_idx = nonmasked_lbx_idx
        self._nonmasked_ubx_idx = nonmasked_ubx_idx
        self._stats = stats
        self._solver_plugin = solver_plugin

    @_cached_property
    def f(self) -> float:
        return float(self._sol["f"])

    @property
    def p_sym(self) -> SymType:
        return self._p_sym

    @_cached_property
    def p(self) -> cs.DM:
        return self._sol["p"]

    @property
    def x_sym(self) -> SymType:
        return self._x_sym

    @_cached_property
    def x(self) -> cs.DM:
        return self._sol["x"]

    @property
    def lam_g_and_h_sym(self) -> cs.DM:
        return self._lam_g_and_h_sym

    @_cached_property
    def lam_g_and_h(self) -> cs.DM:
        return self._sol["lam_g"]

    @property
    def lam_lbx_and_ubx_sym(self) -> cs.DM:
        return self._lam_lbx_and_ubx_sym

    @_cached_property
    def lam_lbx_and_ubx(self) -> cs.DM:
        lam_x = self._sol["lam_x"]
        lam_lbx = -cs.fmin(lam_x[self._nonmasked_lbx_idx, :], 0)
        lam_ubx = cs.fmax(lam_x[self._nonmasked_ubx_idx, :], 0)
        return cs.vertcat(lam_lbx, lam_ubx)

    @_cached_property
    def vals(self) -> dict[str, cs.DM]:
        x = self._x_sym
        x_vals = self._sol["x"]
        return {n: cs.evalf(cs.substitute(v, x, x_vals)) for n, v in self._vars.items()}

    @_cached_property
    def dual_vals(self) -> dict[str, cs.DM]:
        lam_g_and_h_sym = self.lam_g_and_h_sym
        lam_g_and_h = self.lam_g_and_h
        lam_lbx_and_ubx_sym = self.lam_lbx_and_ubx_sym
        lam_lbx_and_ubx = self.lam_lbx_and_ubx
        dual_vals = {}
        for n, v in self._dual_vars.items():
            if n.startswith(("lam_g", "lam_h")):
                dual_vals[n] = cs.evalf(cs.substitute(v, lam_g_and_h_sym, lam_g_and_h))
            elif n.startswith(("lam_lb", "lam_ub")):
                dual_vals[n] = cs.evalf(
                    cs.substitute(v, lam_lbx_and_ubx_sym, lam_lbx_and_ubx)
                )
            else:
                raise RuntimeError(f"unknown dual variable type `{n}`")
        return dual_vals

    @staticmethod
    def from_casadi_solution(
        sol_with_stats: dict[str, _Any], nlp: "Nlp[SymType]"
    ) -> "LazySolution[SymType]":
        stats = sol_with_stats.pop("stats")
        sol: dict[str, cs.DM] = sol_with_stats
        x = nlp._x
        p = nlp._p
        lam_g_and_h = cs.vertcat(nlp._lam_g, nlp._lam_h)
        lam_lbx_and_ubx = cs.vertcat(nlp._lam_lbx, nlp._lam_ubx)
        vars = nlp._vars.copy()
        dual_vars = nlp._dual_vars.copy()
        idx = nlp.nonmasked_lbx_idx
        nonmasked_lbx_idx = idx.copy() if isinstance(idx, np.ndarray) else idx
        idx = nlp.nonmasked_ubx_idx
        nonmasked_ubx_idx = idx.copy() if isinstance(idx, np.ndarray) else idx
        return LazySolution(
            sol,
            p,
            x,
            lam_g_and_h,
            lam_lbx_and_ubx,
            vars,
            dual_vars,
            nonmasked_lbx_idx,
            nonmasked_ubx_idx,
            stats,
            nlp.unwrapped._solver_plugin,
        )


def _broadcast_like(x: SymOrNumType, other: SymOrNumType) -> Union[SymType, np.ndarray]:
    """Internal utility to broadcast a value, if numerical, to the other's shape."""
    if isinstance(x, (np.ndarray, cs.DM)):
        target_shape = other.shape
        if x.shape == target_shape or (
            other.is_vector()
            and (x.size if isinstance(x, np.ndarray) else x.numel()) == other.numel()
        ):
            return x
        return np.broadcast_to(x, target_shape)
    if isinstance(x, (int, float)):
        return np.full(other.shape, x)
    return x  # cs.SX or cs.MX


def _internal_subsevalf_cs(
    expr: SymType,
    old: Union[SymType, dict[str, SymType], _Iterable[SymType]],
    new: Union[SymOrNumType, dict[str, SymOrNumType], _Iterable[SymOrNumType]],
    eval: bool,
) -> Union[SymType, cs.DM]:
    """Internal utility for substituting and evaluting casadi objects."""
    if isinstance(expr, cs.DM):
        return expr
    if isinstance(old, (cs.SX, cs.MX)):
        new_expr = cs.substitute(expr, old, new)
    else:
        old_vals = []
        new_vals = []
        if isinstance(old, dict):
            assert isinstance(new, dict), "old and new should be of the same type"
            for name, old_val in old.items():
                new_val = new[name]
                old_vals.append(old_val)
                new_vals.append(_broadcast_like(new_val, old_val))
        else:  # iterable
            for old_val, new_val in zip(old, new):
                old_vals.append(old_val)
                new_vals.append(_broadcast_like(new_val, old_val))
        new_expr = cs.substitute(expr, cs.vvcat(old_vals), cs.vvcat(new_vals))
    return cs.evalf(new_expr) if eval else new_expr


def _internal_subsevalf_np(
    expr: np.ndarray,
    old: Union[SymType, dict[str, SymType], _Iterable[SymType]],
    new: Union[SymOrNumType, dict[str, SymOrNumType], _Iterable[SymOrNumType]],
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
    new: Union[SymOrNumType, dict[str, SymOrNumType], _Iterable[SymOrNumType]],
    eval: bool = True,
) -> Union[SymType, cs.DM, np.ndarray]:
    """Substitutes the old variables with the new ones in the symbolic expression,
    and evaluates it, if required.

    Parameters
    ----------
    expr : casadi.SX, MX or an array of these
        Expression for substitution and, possibly, evaluation.
    old : casadi.SX or MX or dict/iterable of these
        Old variable to be substituted.
    new : numpy.array or casadi.SX, MX, DM, int, float, or dict/iterable of these
        New variable that substitutes the old one. If a collection, it is
        assumed the type is the same of ``old`` (so, old and new should share
        collection type), e.g., if ``old`` is a dict, ``new`` should be a dict.
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
    return _internal_subsevalf_cs(expr, old, new, eval)
