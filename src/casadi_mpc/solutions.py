from typing import Dict, Union, Iterable
import casadi as cs
from casadi.tools.structure3 import CasadiStructured

# https://casadi.sourceforge.net/tutorials/tools/structure.pdf
# https://github.com/do-mpc/do-mpc/blob/master/do_mpc/tools/casstructure.py


def subsevalf(
    expr: Union[cs.SX, cs.MX],
    old: Union[Union[cs.SX, cs.MX],
               Dict[str, Union[cs.SX, cs.MX]],
               Iterable[Union[cs.SX, cs.MX]],
               CasadiStructured],
    new: Union[Union[cs.SX, cs.MX],
               Dict[str, Union[cs.SX, cs.MX]],
               Iterable[Union[cs.SX, cs.MX]],
               CasadiStructured],
    eval: bool = True
) -> Union[cs.SX, cs.DM]:
    '''
    Substitutes the old variables with the new ones in the symbolic expression,
    and evaluates it, if required.

    Parameters
    ----------
    expr : casadi.SX, MX
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
    new_expr : casadi.SX, MX, DM
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
    '''
    if isinstance(old, (cs.SX, cs.MX, CasadiStructured)):
        expr = cs.substitute(expr, old, new)
    elif isinstance(old, dict):
        for name, o in old.items():
            expr = cs.substitute(expr, o, new[name])
    elif isinstance(old, Iterable):
        for o, n in zip(old, new):
            expr = cs.substitute(expr, o, n)
    else:
        raise TypeError(f'Invalid type {old.__class__.__name__} for old.')

    if eval:
        expr = cs.evalf(expr)
    return expr
