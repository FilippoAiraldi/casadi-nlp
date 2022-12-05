from typing import Any, List, Optional, Union

import casadi as cs
import numpy as np


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

    Inspired by
    https://bitbucket.org/rawlings-group/mpc-tools-casadi/src/master/mpctools/tools.py

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


def jaggedstack(
    arrays,
    axis: int = 0,
    out: Optional[np.ndarray] = None,
    constant_values: Union[float, np.ndarray] = np.nan,
) -> np.ndarray:
    """Joins a sequence of arrays with different shapes along a new axis. To do
    so, each array is padded with `constant_values` (see `numpy.pad`) to the
    right to even out the shapes. Then, the same-shape-arrays are stacked via
    `numpy.stack`.

    Parameters
    ----------
    arrays, axis, out
        See `numpy.stack`.
    constant_values
        See `numpy.pad`.

    Returns
    -------
    stacked : ndarray
        The stacked array has one more dimension than the input arrays.

    Raises
    ------
    ValueError
        Raises if no array is passed as input.
    """
    arraylist: List[np.ndarray] = [np.asanyarray(a) for a in arrays]
    if not arraylist:
        raise ValueError("Need at least one array to stack.")
    maxndim = max(map(lambda a: a.ndim, arraylist))
    newarrays: List[np.ndarray] = []
    maxshape = arraylist[0].shape
    for a in arraylist:
        if a.ndim < maxndim:
            a = np.expand_dims(a, tuple(range(a.ndim, maxndim)))
        maxshape = np.maximum(maxshape, a.shape)
        newarrays.append(a)
    newarrays = [
        np.pad(
            a,
            [(0, d_max - d) for d, d_max in zip(a.shape, maxshape)],
            mode="constant",
            constant_values=constant_values,
        )
        for a in newarrays
    ]
    return np.stack(newarrays, axis, out)
