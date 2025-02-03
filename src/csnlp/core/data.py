"""A collection of functions for manipulating data in CasADi, in particular, on how
to convert to and from numpy arrays and CasADi symbolic variables, and how to find
the indices of a symbolic variable in a vector of symbols."""

from typing import Union

import casadi as cs
import numpy as np
import numpy.typing as npt


def array2cs(x: np.ndarray) -> Union[cs.SX, cs.MX]:
    """Converts the numpy array ``x`` containing scalar symbolic variables to a single
    symbolic variable (see :func:`cs2array` for the opposite functionality).

    Parameters
    ----------
    x : numpy array of casadi.SX or MX
        Array whose entries are either MX or SX.

    Returns
    -------
    casadi.SX or MX
        A single SX or MX instance whose entries are the ``x``'s entries.

    Raises
    ------
    ValueError
        Raises if the array is empty (zero dimensions), or if it has more than 2
        dimensions.

    Notes
    -----
    Note that all entries in ``x`` must have the same type, either SX or MX.
    """
    if isinstance(x, (cs.SX, cs.MX, cs.DM)):
        return x
    cls = type(next(x.flat))
    if x.dtype != object or cls is cs.DM:
        return cs.DM(x)  # meaning it is not SX or MX, thus a number
    ndim = x.ndim
    if ndim == 0:
        return x.item()
    if ndim == 1:
        numel = x.size
        shape = (numel, 1)
        indices = range(numel)
    elif ndim == 2:
        shape = x.shape
        indices = np.ndindex(shape)
    else:
        raise ValueError("Can only convert 1D and 2D arrays to CasADi SX or MX.")
    m: Union[cs.SX, cs.MX] = cls(*shape)
    for idx in indices:
        m[idx] = x[idx]
    return m


def cs2array(x: Union[cs.MX, cs.SX]) -> np.ndarray:
    """Converts the symbolic variable ``x`` to a numpy array of scalar symbolic
    variables (see :func:`array2cs` for the opposite functionality).

    Inspired by the implementation in
    `MPCTools <https://bitbucket.org/rawlings-group/mpc-tools-casadi/src/master/mpctools/tools.py>`_.

    Parameters
    ----------
    x : casadi.SX or MX
        A symbolic variable (with multiple entries).

    Returns
    -------
    numpy array of casadi.SX or MX
        The array containing the symbolic variable scalars.
    """
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, cs.DM):
        return x.toarray()

    shape = x.shape
    y = np.empty(shape, object)
    for idx in np.ndindex(shape):
        y[idx] = x[idx]
    return y


def find_index_in_vector(
    V: Union[cs.SX, cs.MX], a: Union[cs.SX, cs.MX]
) -> npt.NDArray[np.int_]:
    """Finds the indices of ``a`` in ``V``.

    Parameters
    ----------
    V : casadi.SX or MX
        The vector in which to search.
    a : casadi.SX or MX
        The vector to search for.

    Returns
    -------
    array of integers
        The indices of ``a`` in ``V``.

    Raises
    ------
    ValueError
        Raises if ``V`` or ``a`` are not vectors.
    """
    if not V.is_vector() or not a.is_vector():
        raise ValueError("`V` and `a` must be vectors.")

    sp: cs.Sparsity = cs.jacobian_sparsity(a, V)
    idx = np.asarray(sp.get_crs()[1], int)
    if idx.size != a.numel():
        raise RuntimeError("invalid subset: some entries of `a` were not found in `V`")
    return idx
