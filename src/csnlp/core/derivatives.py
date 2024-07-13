from itertools import product
from typing import Union

import casadi as cs
import numpy as np

from .data import array2cs, cs2array


def hojacobian(ex: Union[cs.MX, cs.SX], x: Union[cs.MX, cs.SX]) -> np.ndarray:
    """Computes jacobian on higher-order matrices, i.e.,

    Parameters
    ----------
    ex : casadi.MX or SX
        The expression to compute the jacobian for. Can be a matrix.
    x : casadi.MX or SX
        The variable to differentiate with respect to. Can be a matrix.

    Returns
    -------
    numpy array of symbolic variables
        A 4D array of objects, where each entry ``(i,j,k,m)`` is the derivative
        of ``ex[i,j]`` w.r.t. ``x[k,m]``.
    """
    return cs2array(cs.jacobian(cs.vec(ex), cs.vec(x))).reshape(
        ex.shape + x.shape, order="F"
    )


def hohessian(
    ex: Union[cs.MX, cs.SX], x: Union[cs.MX, cs.SX], y: Union[cs.MX, cs.SX, None] = None
) -> tuple[np.ndarray, np.ndarray]:
    """Computes hessian on higher-order matrices, similar to :func:`hojacobian`.

    Parameters
    ----------
    ex : casadi.MX or SX
        The expression to compute the hessin for. Can be a matrix.
    x : casadi.MX or SX
        The variable to differentiate with respect to. Can be a matrix.
    y : casadi.MX or SX, optional
        Use this argument to specify the second partial derivative, i.e., if
        the output requested is not the hessian of ``ex`` w.r.t. ``x``, but rather
        the second-order partial derivatives of ``ex`` w.r.t. ``x`` and ``y``.

    Returns
    -------
    tuple of 2 numpy arrays of symbolic variables
        The first element is a 6D array of objects, where each entry
        ``(i,j,k,m,l,p)`` is the hessian of ``ex[i,j]`` w.r.t. ``x[k,m], y[l,p]``,
        while the second element contains the jacobian instead (see :func:`hojacobian`
        for details).
    """
    if y is None:
        y = x
    J = hojacobian(ex, x)
    H = np.empty(ex.shape + x.shape + y.shape, object)
    for i in product(*map(range, ex.shape)):
        H[i] = hojacobian(array2cs(J[i]), y)
    return H, J
