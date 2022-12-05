from dataclasses import dataclass
from typing import Any, Callable, Dict, Generic, Iterable, TypeVar, Union

import casadi as cs
import numpy as np
from casadi.tools.structure3 import CasadiStructured, DMStruct

T = TypeVar("T", cs.SX, cs.MX)


@dataclass(frozen=True)
class Solution(Generic[T]):
    """Class containing information on the solution of an NLP solver's run."""

    f: float
    vars: Union[CasadiStructured, Dict[str, T]]
    vals: Union[DMStruct, Dict[str, cs.DM]]
    stats: Dict[str, Any]
    _get_value: Callable[[T, bool], cs.DM]

    @property
    def status(self) -> str:
        """Gets the status of the solver at this solution."""
        return self.stats["return_status"]

    @property
    def success(self) -> bool:
        """Gets whether the run was run successfully."""
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
        return self._get_value(x, eval=eval)


def subsevalf(
    expr: Union[T, np.ndarray],
    old: Union[
        T,
        Dict[str, T],
        Iterable[T],
        CasadiStructured,
    ],
    new: Union[
        T,
        Dict[str, T],
        Iterable[T],
        CasadiStructured,
    ],
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
    TypeError
        Raises if the `old` and `new` are neither SX, MX, dict, Iterable.
    RuntimeError
        Raises if `eval=True` but there are symbolic variables that are still
        free, i.e., the expression cannot be evaluated numerically since it is
        still (partially) symbolic.
    """
    if isinstance(expr, np.ndarray):
        out = np.empty(expr.shape, dtype=object)
        for i in np.ndindex(expr.shape):
            out[i] = subsevalf(expr[i], old, new, eval=eval)
        if eval:
            out = out.astype(float)
        return out

    if isinstance(old, (cs.SX, cs.MX, CasadiStructured)):
        expr = cs.substitute(expr, old, new)
    elif isinstance(old, dict):
        for name, o in old.items():
            expr = cs.substitute(expr, o, new[name])
    elif isinstance(old, Iterable):
        for o, n in zip(old, new):
            expr = cs.substitute(expr, o, n)
    else:
        raise TypeError(f"Invalid type {old.__class__.__name__} for old.")

    if eval:
        expr = cs.evalf(expr)
    return expr
