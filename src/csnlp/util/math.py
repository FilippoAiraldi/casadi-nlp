"""A collection of stand-alone functions that implement some of the basic mathematical
operations that are not available in CasADi. The implementations are simple and thus
not optimized for performance. They are meant to be used as a fallback when the CasADi
really does not provide the required functionality.
"""

from math import sqrt as _sqrt
from typing import Literal, Union

import casadi as cs

SQRT2 = _sqrt(2)


def log(
    x: Union[cs.SX, cs.MX, cs.DM], base: Union[None, cs.SX, cs.MX, cs.DM] = None
) -> Union[cs.SX, cs.MX, cs.DM]:
    r"""Computes the logarithm. With one argument, return the natural logarithm of
    :math:`x` (to base :math:`e`). With two arguments, return the logarithm of :math:`x`
    to the given base :math:`b`, calculated as :math:`\frac{\log x}{\log b}`.

    Parameters
    ----------
    x : casadi.SX, MX or DM
        Value to compute the logarithm of.
    base : casadi.SX, MX or DM, optional
        Base of the logarithm, by default ``None``, so that the natural logarithm is
        computed.

    Returns
    -------
    casadi.SX, MX or DM
        The logarithm of ``x`` with base ``base``, i.e., :math:`\log_{base} x`.
    """
    if base is None:
        return cs.log(x)
    if base == 10 or base == 10.0:
        return cs.log10(x)
    return cs.log(x) / cs.log(base)


def prod(
    x: Union[cs.SX, cs.MX, cs.DM], axis: Literal[0, 1, None, -1, -2] = None
) -> Union[cs.SX, cs.MX, cs.DM]:
    r"""Computes the product of all the elements in ``x`` (CasADi version of
    :func:`numpy.prod`).

    Parameters
    ----------
    x : casadi.SX, MX or DM
        The variable whose entries must be multiplied together.
    axis : {0, 1, None, -1, -2}
        Axis or along which a product is performed. The default, ``axis=None``, will
        calculate the product of all the elements in the matrix. If axis is negative it
        counts from the last to the first axis.

    Returns
    -------
    casadi.SX, MX or DM
        Product of the elements, i.e., :math:`\prod_{i=1}^{|x|}{x_i}`.
    """
    # sourcery skip: merge-comparisons
    if axis is None:
        x = cs.vec(x)
        axis = 0

    # prod of vector elements
    if x.is_vector():
        if x.shape[axis] == 1:
            return x
        if not isinstance(x, cs.MX):
            return cs.det(cs.diag(x))  # does not work with MX
        p = x[0]
        for i in range(1, x.shape[axis]):
            p *= x[i]
        return p

    # prod of matrix elements
    sum_ = cs.sum1 if (axis == 0 or axis == -2) else cs.sum2
    n_negatives = sum_(x < 0)
    p = cs.exp(sum_(cs.log(cs.fabs(x))))
    return cs.if_else(cs.remainder(n_negatives, 2) == 0.0, 1, -1) * p


def norm_cdf(
    x: Union[cs.SX, cs.MX, cs.DM],
    loc: Union[cs.SX, cs.MX, cs.DM] = 0,
    scale: Union[cs.SX, cs.MX, cs.DM] = 1,
) -> Union[cs.SX, cs.MX, cs.DM]:
    """Computes the cdf of a normal distribution (CasADi version of
    :data:`scipy.stats.norm`'s ``cdf`` method).

    Parameters
    ----------
    x : casadi.SX, MX or DM
        The value at which the cdf is computed.
    loc : casadi.SX, MX or DM, optional
        Mean of the normal distribution. By default, ``loc=0``.
    scale : casadi.SX, MX or DM, optional
        Standard deviation of the normal distribution. By default, ``scale=0``.

    Returns
    -------
    casadi.SX, MX or DM
        The cdf of the normal distribution.
    """
    return 0.5 * (1 + cs.erf((x - loc) / SQRT2 / scale))


def norm_ppf(
    p: Union[cs.SX, cs.MX, cs.DM],
    loc: Union[cs.SX, cs.MX, cs.DM] = 0,
    scale: Union[cs.SX, cs.MX, cs.DM] = 1,
) -> Union[cs.SX, cs.MX, cs.DM]:
    """Computes the quantile (invese of :func:`norm_cdf`) of a normal distribution
    (CasADi version of :data:`scipy.stats.norm`'s ``ppf`` method).

    Parameters
    ----------
    x : casadi.SX, MX or DM
        The value at which the quantile is computed.
    loc : casadi.SX, MX or DM, optional
        Mean of the normal distribution. By default, ``loc=0``.
    scale : casadi.SX, MX or DM, optional
        Standard deviation of the normal distribution. By default, ``scale=0``.

    Returns
    -------
    casadi.SX, MX or DM
        The quantile of the normal distribution.
    """
    return SQRT2 * scale * cs.erfinv(2 * p - 1) + loc


def repeat(
    a: Union[cs.SX, cs.MX, cs.DM], repeats: Union[int, tuple[int, int]]
) -> Union[cs.SX, cs.MX, cs.DM]:
    """Repeats elements in array.

    Parameters
    ----------
    a : casadi.SX or MX or DM
        The array/matrix whose elements are to be repeated.
    repeats : int or tuple of 2 ints
        The number of repeats, in the first and/or second axis.

    Returns
    -------
    casadi.SX, MX or DM
        Output array with repeated elements.
    """
    return cs.kron(a, cs.GenDM_ones(repeats))
