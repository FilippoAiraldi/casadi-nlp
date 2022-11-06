from dataclasses import dataclass
from typing import Any, Callable, Dict, Union, Iterable
import casadi as cs
from casadi.tools.structure3 import CasadiStructured, ssymStruct, DMStruct


# https://casadi.sourceforge.net/tutorials/tools/structure.pdf
# https://github.com/do-mpc/do-mpc/blob/master/do_mpc/tools/casstructure.py


@dataclass(frozen=True)
class Solution:
    '''Class containing information on the solution of an MPC solver's run.'''

    f: float
    vars: Union[ssymStruct, Dict[str, cs.MX]]
    vals: DMStruct
    stats: Dict[str, Any]
    _get_value: Callable[[Union[cs.SX, cs.MX], bool], cs.DM]

    @property
    def status(self) -> str:
        '''Gets the status of the solver at this solution.'''
        return self.stats['return_status']

    @property
    def success(self) -> bool:
        '''Gets whether the MPC was run successfully.'''
        return self.stats['success']

    def value(
        self,
        x: Union[cs.SX, cs.MX],
        eval: bool = True
    ) -> Union[cs.SX, cs.MX, cs.DM]:
        '''Computes the value of the expression substituting the values of this
        solution in the expression.

        Parameters
        ----------
        x : Union[cs.SX, cs.MX]
            The symbolic expression to be evaluated at the solution's values.
        eval : bool, optional
            Evaluates numerically the new expression. By default, `True`. See 
            `casadi_mpc.solutions.subsevalf` for more details.

        Returns
        -------
        cs.SX or MX or DM
            The expression evaluated with the solution's values.

        Raises
        ------
        RuntimeError
            Raises if `eval=True` but there are symbolic variables that are 
            still free since they are outside the solution's variables.
        '''
        return self._get_value(x, eval=eval)


def subsevalf(
    expr: Union[cs.SX, cs.MX],
    old: Union[cs.SX, cs.MX,
               Dict[str, Union[cs.SX, cs.MX]],
               Iterable[Union[cs.SX, cs.MX]],
               CasadiStructured],
    new: Union[cs.SX, cs.MX,
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
