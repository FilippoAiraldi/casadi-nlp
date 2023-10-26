from itertools import product
from typing import Tuple, Union

import casadi as cs
import numpy as np

from .data import array2cs, cs2array


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
    H = np.empty(ex.shape + x.shape + y.shape, object)
    for i in product(*map(range, ex.shape)):
        H[i] = hojacobian(array2cs(J[i]), y)
    return H, J
