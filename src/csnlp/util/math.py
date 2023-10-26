from math import sqrt
from typing import Optional, Union

import casadi as cs

SQRT2 = sqrt(2)


def log(
    x: Union[cs.SX, cs.MX, cs.DM], base: Union[None, cs.SX, cs.MX, cs.DM] = None
) -> Union[cs.SX, cs.MX, cs.DM]:
    """Logarithm. With one argument, return the natural logarithm of `x` (to base `e`).
    With two arguments, return the logarithm of x to the given base, calculated as
    `log(x) / log(base)`.

    Parameters
    ----------
    x : Union[cs.SX, cs.MX, cs.DM]
        Value to compute the logarithm of.
    base : Union[None, cs.SX, cs.MX, cs.DM], optional
        Base of the logarithm, by default `None`

    Returns
    -------
    Union[cs.SX, cs.MX, cs.DM]
        The logarithm. of `x` with base `base`.
    """
    if base is None:
        return cs.log(x)
    if base == 10 or base == 10.0:
        return cs.log10(x)
    return cs.log(x) / cs.log(base)


def prod(
    x: Union[cs.SX, cs.MX, cs.DM], axis: Optional[int] = None
) -> Union[cs.SX, cs.MX, cs.DM]:
    """Computes the product of all the elements in `x` (CasADi version of `numpy.prod`).

    Parameters
    ----------
    x : Union[cs.SX, cs.MX, cs.DM]
        The variable whose entries must be multiplied together.
    axis : {0, 1, None, -1, -2}
        Axis or along which a product is performed. The default, `axis=None`, will
        calculate the product of all the elements in the matrix. If axis is negative it
        counts from the last to the first axis.

    Returns
    -------
    Union[cs.SX, cs.MX, cs.DM]
        Product of the elements.
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
    loc: Union[cs.SX, cs.MX, cs.DM] = cs.DM(0),
    scale: Union[cs.SX, cs.MX, cs.DM] = cs.DM(1),
) -> Union[cs.SX, cs.MX, cs.DM]:
    """Computes the cdf of a normal distribution. See `scipy.stats.norm.cdf`.

    Parameters
    ----------
    x : Union[cs.SX, cs.MX, cs.DM]
        The value at which the cdf is computed.
    loc : Union[cs.SX, cs.MX, cs.DM], optional
        Mean of the normal distribution. By default, `loc=0`.
    scale : Union[cs.SX, cs.MX, cs.DM], optional
        Standard deviation of the normal distribution. By default, `scale=0`.

    Returns
    -------
    Union[cs.SX, cs.MX, cs.DM]
        The cdf of the normal distribution.
    """
    return 0.5 * (1 + cs.erf((x - loc) / SQRT2 / scale))


def norm_ppf(
    p: Union[cs.SX, cs.MX, cs.DM],
    loc: Union[cs.SX, cs.MX, cs.DM] = cs.DM(0),
    scale: Union[cs.SX, cs.MX, cs.DM] = cs.DM(1),
) -> Union[cs.SX, cs.MX, cs.DM]:
    """Computes the quantile (invese of cdf) of a normal distribution. See
    `scipy.stats.norm.ppf`.

    Parameters
    ----------
    x : Union[cs.SX, cs.MX, cs.DM]
        The value at which the quantile is computed.
    loc : Union[cs.SX, cs.MX, cs.DM], optional
        Mean of the normal distribution. By default, `loc=0`.
    scale : Union[cs.SX, cs.MX, cs.DM], optional
        Standard deviation of the normal distribution. By default, `scale=0`.

    Returns
    -------
    Union[cs.SX, cs.MX, cs.DM]
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
    repeats : int or tuple[int, int]
        The number of repeats, in the first and/or second axis.

    Returns
    -------
    Union[cs.SX, cs.MX, cs.DM]
        Output array with repeated elements.
    """
    return cs.kron(a, cs.GenDM_ones(repeats))
