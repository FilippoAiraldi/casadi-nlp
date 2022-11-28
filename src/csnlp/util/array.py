from typing import List, Tuple, Union

import casadi as cs
import numpy as np

from csnlp.util.data import array2cs, cs2array


def hojacobian(ex: Union[cs.MX, cs.SX], x: Union[cs.MX, cs.SX]) -> np.ndarray:
    """Computes jacobian on higher-order matrices, i.e., with an output in 4D.

    Parameters
    ----------
    ex : casadi.MX, SX
        The expression to compute the jacobian for. Can be a matrix.
    x : casadi.MX, SX
        The variable to differentiate with respect to. Can be a matrix.

    Returns
    -------
    np.ndarray
        A 4D array of objects, where each entry `(i,j,k,m)` is the derivative
        of `ex[i,j]` w.r.t. `x[k,m]`.
    """
    return cs2array(cs.jacobian(cs.vec(ex), cs.vec(x))).reshape(
        ex.shape + x.shape, order="F"
    )


def hohessian(
    ex: Union[cs.MX, cs.SX], x: Union[cs.MX, cs.SX], y: Union[cs.MX, cs.SX, None] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes hessian on higher-order matrices, i.e., with an output in 6D.

    Parameters
    ----------
    ex : casadi.MX, SX
        The expression to compute the hessin for. Can be a matrix.
    x : casadi.MX, SX
        The variable to differentiate with respect to. Can be a matrix.
    y : casadi.MX, SX, optional
        Use this argument to specify the second partial derivative, i.e., if
        the output requested is not the hessian of `ex` w.r.t. `x`, but rather
        the second-order partial derivatives of `ex` w.r.t. `x` and `y`.

    Returns
    -------
    np.ndarray, np.ndarray
        The first element is a 6D array of objects, where each entry
        `(i,j,k,m,l,p)` is the hessian of `ex[i,j]` w.r.t. `x[k,m], y[l,p]`,
        while the second element contains the jacobian instead (see
        `hojacobian` for details).
    """
    if y is None:
        y = x
    J = hojacobian(ex, x)
    H = np.empty(ex.shape + x.shape + y.shape, dtype=object)
    for i in np.ndindex(ex.shape):
        H[i] = hojacobian(array2cs(J[i]), y)
    return H, J


def jaggedstack(
    arrays,
    axis: int = 0,
    out: np.ndarray = None,
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
