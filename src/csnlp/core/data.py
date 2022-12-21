from typing import Union

import casadi as cs
import numpy as np


def array2cs(x: np.ndarray) -> Union[cs.SX, cs.MX]:
    """Converts numpy array `x` of scalar symbolic variable to a single symbolic
    instance. Opposite to `array2cs`. Note that all entries in `x` must have the same
    type, either SX or MX.

    Parameters
    ----------
    x : np.ndarray
        Array whose entries are either MX or SX. In case `x` is SX, MX or DM, it is
        returned immediately.

    Returns
    -------
    casadi.SX or MX
        A single SX or MX instance whose entries are `x`'s entries.

    Raises
    ------
    NotImplementedError
        Raises if the array is empty (zero dimensions), or if it has more than 2 dims.
    """
    if isinstance(x, (cs.SX, cs.MX, cs.DM)):
        return x

    first_item = next(x.flat)
    if x.dtype != object or isinstance(first_item, cs.DM):
        return cs.DM(x)
    if isinstance(first_item, cs.SX):
        return cs.SX(x)
    shape = x.shape
    m = cs.MX(*shape)
    for i in np.ndindex(shape):
        m[i] = x[i]
    return m


def cs2array(x: Union[cs.MX, cs.SX]) -> np.ndarray:
    """Converts casadi symbolic variable `x` to a numpy array of scalars. Opposite to
    `array2cs`.

    Inspired by
    https://bitbucket.org/rawlings-group/mpc-tools-casadi/src/master/mpctools/tools.py

    Parameters
    ----------
    x : casadi.SX or MX
        A symbolic variable (with multiple entries).

    Returns
    -------
    np.ndarray
        The array containing the symbolic variable scalars.
    """
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, cs.DM):
        return x.full()

    shape = x.shape
    y = np.empty(shape, dtype=object)
    for i in np.ndindex(shape):
        y[i] = x[i]
    return y
