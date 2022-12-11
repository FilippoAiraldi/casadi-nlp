from dataclasses import dataclass
from typing import Any, Callable, Dict, Generic, Iterable, TypeVar, Union

import casadi as cs
import numpy as np
from casadi.tools.structure3 import CasadiStructured, DMStruct

from csnlp.core.data import array2cs, cs2array

T = TypeVar("T", cs.SX, cs.MX)


@dataclass(frozen=True)
class Solution(Generic[T]):
    """Class containing information on the solution of an NLP solver's run."""

    f: float
    vars: Dict[str, T]
    vals: Dict[str, cs.DM]
    stats: Dict[str, Any]
    _get_value: Callable[[T, bool], cs.DM]

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

    def value(self, x: T, eval: bool = True) -> Union[T, cs.DM]:
        """Computes the value of the expression substituting the values of this
        solution in the expression.

        Parameters
        ----------
        x : casadi.SX or MX
            The symbolic expression to be evaluated at the solution's values.
        eval : bool, optional
            Evaluates numerically the new expression. By default, `True`. See
            `csnlp.solutions.subsevalf` for more details.

        Returns
        -------
        casadi.SX or MX or DM
            The expression evaluated with the solution's values.

        Raises
        ------
        RuntimeError
            Raises if `eval=True` but there are symbolic variables that are
            still free since they are outside the solution's variables.
        """
        return self._get_value(x, eval=eval)  # type: ignore


def _internal_subsevalf_cs(
    expr: T,
    old: Union[T, Dict[str, T], Iterable[T]],
    new: Union[T, Dict[str, T], Iterable[T]],
    eval: bool,
) -> Union[T, cs.DM]:
    """Internal utility for substituting and evaluting casadi objects."""
    if isinstance(expr, (cs.DM, DMStruct)):
        return expr

    if isinstance(old, dict):
        for name, o in old.items():
            expr = cs.substitute(expr, o, new[name])  # type: ignore
    elif isinstance(old, Iterable) and not isinstance(
        old, (cs.SX, cs.MX, CasadiStructured)
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
    old: Union[T, Dict[str, T], Iterable[T]],
    new: Union[T, Dict[str, T], Iterable[T]],
    eval: bool,
) -> Union[T, np.ndarray, cs.DM]:
    # sourcery skip: remove-redundant-slice-index
    """Internal utility for substituting and evaluting arrays of casadi objects."""
    # if not symbolic, return it
    if expr.dtype != object:
        return expr

    # up to 2D, we can get away with only one substitution
    if expr.ndim <= 2:
        return _internal_subsevalf_cs(array2cs(expr), old, new, eval)

    # for tensors, check if the right end or the left is bigger, then loop over the
    # rest while substituing in the right/left end of the array
    shape = expr.shape
    transposed = (shape[0] * shape[1]) > (shape[-2] * shape[-1])
    if transposed:
        shape = shape[::-1]
        expr = expr.transpose()
    out = np.empty(shape, dtype=object)
    for i in np.ndindex(shape[0:-2:]):  # 0 is needed to avoid wrapping around the shape
        shape_i = expr[i].shape
        out[i] = cs2array(
            _internal_subsevalf_cs(array2cs(expr[i]), old, new, eval)
        ).reshape(shape_i)
    if transposed:
        out = out.transpose()
        expr = expr.transpose()

    if eval:
        out = out.astype(float)
    return out


def subsevalf(
    expr: Union[T, np.ndarray],
    old: Union[T, Dict[str, T], Iterable[T]],
    new: Union[T, Dict[str, T], Iterable[T]],
    eval: bool = True,
) -> Union[T, cs.DM, np.ndarray]:
    """
    Substitutes the old variables with the new ones in the symbolic expression,
    and evaluates it, if required.

    Parameters
    ----------
    expr : casadi.SX, MX or an array of these
        Expression for substitution and, possibly, evaluation.
    old : casadi.SX, MX (or struct, dict, iterable of)
        Old variable to be substituted.
    new : numpy.array or casadi.SX, MX, DM (or struct, dict, iterable of)
        New variable that substitutes the old one. If a collection, it is
        assumed the type is the same of `old` (so, old and new should share
        collection type).
    eval : bool, optional
        Evaluates numerically the new expression. By default, `True`.

    Returns
    -------
    new_expr : casadi.SX, MX, DM or an array of these
        New expression after substitution (SX, MX) and, possibly, evaluation
        (DM).

    Raises
    ------
    Exception
        Raises if `old` and `new` are not compatible with `casadi.substitute`; or if
        `eval=True` but there are symbolic variables that are still free, i.e., the
        expression cannot be evaluated numerically since it is still (partially)
        symbolic.
    """
    if isinstance(expr, np.ndarray):
        return _internal_subsevalf_np(expr, old, new, eval)
    else:
        return _internal_subsevalf_cs(expr, old, new, eval)
