import math
from typing import Union, Optional
import casadi as cs


SQRT2 = math.sqrt(2)


def prod(
    x: Union[cs.SX, cs.MX, cs.DM],
    axis: Optional[int] = None
) -> Union[cs.SX, cs.MX, cs.DM]:
    '''Computes the product of all the elements in `x` (CasADi version of
    `numpy.prod`).

    Parameters
    ----------
    x : Union[cs.SX, cs.MX, cs.DM]
        The variable whose entries must be multiplied together.
    axis : {0, 1, None}
        Axis or along which a product is performed. The default, `axis=None`,
        will calculate the product of all the elements in the matrix. If axis
        is negative it counts from the last to the first axis.

    Returns
    -------
    Union[cs.SX, cs.MX, cs.DM]
        Product of the elements.
    '''
    if axis is None:
        x = cs.vec(x)
        axis = 0
    elif axis < 0:
        axis = 2 + axis
    sum_ = cs.sum1 if axis == 0 else cs.sum2
    n_negatives = sum_(x < 0)
    p = cs.exp(sum_(cs.log(cs.fabs(x))))
    return cs.if_else(cs.mod(n_negatives, 2) == 0.0, 1, -1) * p


def quad_form(
    A: Union[cs.SX, cs.MX, cs.DM],
    x: Union[cs.SX, cs.MX, cs.DM]
) -> Union[cs.SX, cs.MX, cs.DM]:
    '''Calculates quadratic form `x.T*A*x`.

    Parameters
    ----------
    A : Union[cs.SX, cs.MX, cs.DM]
        The matrix of weights in the quadratic form. If a vector, a matrix with
        the vector as diagonal is used instead.
    x : Union[cs.SX, cs.MX, cs.DM]
        The vector in the quadratic form.

    Returns
    -------
    Union[cs.SX, cs.MX, cs.DM]
        The result of the quadratic form.

    Raises
    ------
    RuntimeError
        Raises if
         - `A` is not squared
         - `A` and `x` cannot be multiplied together
         - `x` is not a vector.
    '''
    if A.is_vector():
        A = cs.diag(A)
    return cs.bilin(A, x, x)


def norm_cdf(
    x: Union[cs.SX, cs.MX, cs.DM],
    loc: Union[cs.SX, cs.MX, cs.DM] = cs.DM(0),
    scale: Union[cs.SX, cs.MX, cs.DM] = cs.DM(1)
) -> Union[cs.SX, cs.MX, cs.DM]:
    '''Computes the cdf of a normal distribution. See `scipy.stats.norm.cdf`.

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
    '''
    return 0.5 * (1 + cs.erf((x - loc) / SQRT2 / scale))


def norm_ppf(
    p: Union[cs.SX, cs.MX, cs.DM],
    loc: Union[cs.SX, cs.MX, cs.DM] = cs.DM(0),
    scale: Union[cs.SX, cs.MX, cs.DM] = cs.DM(1)
) -> Union[cs.SX, cs.MX, cs.DM]:
    '''Computes the quantile (invese of cdf) of a normal distribution. See
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
    '''
    return SQRT2 * scale * cs.erfinv(2 * p - 1) + loc
