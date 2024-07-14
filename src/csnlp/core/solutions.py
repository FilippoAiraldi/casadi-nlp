"""Contains classes and methods to store the solution of an NLP problem after a call
to :meth:`csnlp.Nlp.solve` or :meth:`csnlp.multistart.MultistartNlp.solve_multi`."""

from collections.abc import Iterable as _Iterable
from dataclasses import dataclass as _dataclass
from functools import partial as _partial
from itertools import product as _product
from typing import Any
from typing import Generic as _Generic
from typing import TypeVar as _TypeVar
from typing import Union

import casadi as cs
import numpy as np
from casadi.tools.structure3 import CasadiStructured as _CasadiStructured
from casadi.tools.structure3 import DMStruct as _DMStruct

from .data import array2cs as _array2cs
from .data import cs2array as _cs2array

SymType = _TypeVar("SymType", cs.SX, cs.MX)


@_dataclass(frozen=True, repr=False, order=True)
class Solution(_Generic[SymType]):
    """Class containing information on the solution of a solver's run for an instance of
    :class:`csnlp.Nlp`."""

    f: float
    """Optimal value of the objective function."""

    vars: dict[str, SymType]
    """Symbolical primal variables."""

    vals: dict[str, cs.DM]
    """Optimal values of the primal variables."""

    dual_vars: dict[str, SymType]
    """Symbolical dual variables."""

    dual_vals: dict[str, cs.DM]
    """Optimal values of the dual variables."""

    stats: dict[str, Any]
    """Statistics of the solver for this solution's run."""

    _get_value: _partial

    @property
    def all_vars(self) -> SymType:
        """Gets all the variables of the solution in a vector."""
        return self._get_value.keywords["old"]

    @property
    def all_vals(self) -> cs.DM:
        """Gets all the values of the solution in a vector."""
        return self._get_value.keywords["new"]

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

    def value(self, x: SymType, eval: bool = True) -> Union[SymType, cs.DM]:
        """Computes the value of the expression substituting the values of this
        solution in the expression.

        Parameters
        ----------
        x : casadi.SX or MX
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
        return self._get_value(x, eval=eval)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(f={self.f},success={self.success},"
            f"status={self.status})"
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
    """
    Substitutes the old variables with the new ones in the symbolic expression,
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
