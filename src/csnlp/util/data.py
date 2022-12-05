from typing import Any, Dict, Literal, Union

import casadi as cs
import casadi.tools as cst
import numpy as np
from casadi.tools.structure3 import DMStruct


def is_casadi_object(obj: Any) -> bool:
    """Checks if the object belongs to the CasADi module.

    See https://stackoverflow.com/a/52783240/19648688 for more details.

    Parameters
    ----------
    obj : Any
        Any type of object.

    Returns
    -------
    bool
        A flag that states whether the object belongs to CasADi or not.
    """
    if not hasattr(obj, "__module__"):
        return False
    module: str = obj.__module__.split(".")[0]
    return module == cs.__name__


class struct_symSX(cst.struct_symSX):
    """Updated structure class for CasADi structures (SX). This class fixes a bug that
    prevents unpickeling of the structure.

    Implementation taken from
    https://github.com/do-mpc/do-mpc/blob/master/do_mpc/tools/casstructure.py"""

    def __init__(self, *args, **kwargs):
        kwargs.pop("order", None)
        super().__init__(*args, **kwargs)


class struct_SX(cst.struct_SX):
    """Updated structure class for CasADi structures (SX). This class fixes a bug that
    prevents unpickeling of the structure.

    Implementation taken from
    https://github.com/do-mpc/do-mpc/blob/master/do_mpc/tools/casstructure.py"""

    def __init__(self, *args, **kwargs):
        kwargs.pop("order", None)
        super().__init__(*args, **kwargs)


def dict2struct(
    dict: Dict[str, Union[cs.DM, cs.SX, cs.MX]],
    entry_type: Literal["sym", "expr"] = "sym",
) -> Union[DMStruct, cst.struct_symSX, cst.struct_SX, Dict[str, cs.MX]]:
    """Attempts to convert a dictionary of CasADi matrices to a struct. The
    algorithm is inferred from the type of the first element of `dict`:
     - if `DM`, then a numerical `DMStruct` is returned
     - if `SX`, then a symbolical `struct_symSX` is returned
     - if `MX` (or any other, for this matter), only a copy of `dict` is
       returned.

    Parameters
    ----------
    dict : Dict[str, Union[cs.DM, cs.SX, cs.MX]]
        Dictionary of names and their corresponding symbolic variables.
    entry_type : 'sym', 'expr'
        SX struct entry type. By default, `'sym'`.

    Returns
    -------
    Union[DMStruct, struct_symSX, struct_SX, Dict[str, cs.MX]]
        Either a structure generated from `dict`, or a copy of `dict` itself.
    """
    if not dict:  # in case of empty dict
        return {}
    o = next(iter(dict.values()))
    if isinstance(o, cs.DM):
        dummy = struct_symSX(
            [cst.entry(name, shape=p.shape) for name, p in dict.items()]
        )
        return dummy(cs.vertcat(*map(cs.vec, dict.values())))
    elif isinstance(o, cs.SX):
        struct = struct_symSX if entry_type == "sym" else struct_SX
        return struct([cst.entry(name, **{entry_type: p}) for name, p in dict.items()])
    else:
        return dict.copy()


def array2cs(x: np.ndarray) -> Union[cs.SX, cs.MX]:
    """Converts numpy array `x` of scalar symbolic variable to a single
    symbolic instance. Opposite to `array2cs`. Note that all entries in `x`
    must have the same type, either SX or MX.

    Parameters
    ----------
    x : np.ndarray
        Array whose entries are either MX or SX.

    Returns
    -------
    Union[cs.SX, cs.MX]
        A single SX or MX instance.

    Raises
    ------
    ValueError
        Raises if the array is empty (zero dimensions), or if it has more than
        2 dimensions.
    """
    if isinstance(x, (cs.SX, cs.MX, cs.DM)):
        return x
    ndim = x.ndim
    if ndim == 0:
        raise ValueError("Cannot convert empty arrays.")
    elif ndim == 1:
        o = x[0]
        x = x.reshape(-1, 1)
    elif ndim == 2:
        o = x[0, 0]
    else:
        raise ValueError("Can only convert 1D and 2D arrays to CasADi SX or MX.")
    # infer type from first element
    sym_type = type(o)
    if sym_type is cs.SX:
        return cs.SX(x)
    shape = x.shape
    m = cs.MX(*shape)
    for i in np.ndindex(shape):
        m[i] = x[i]
    return m


def cs2array(x: Union[cs.MX, cs.SX]) -> np.ndarray:
    """Converts casadi symbolic variable `x` to a numpy array of scalars.
    Opposite to `array2cs`.

    Inspired by https://bitbucket.org/rawlings-group/mpc-tools-casadi/src/master/mpctools/tools.py

    Parameters
    ----------
    x : Union[cs.MX, cs.SX]
        _description_

    Returns
    -------
    np.ndarray
        The array containing the symbolic variable scalars.
    """
    if isinstance(x, np.ndarray):
        return x
    shape = x.shape
    y = np.empty(shape, dtype=object)
    for i in np.ndindex(shape):
        y[i] = x[i]
    return y
