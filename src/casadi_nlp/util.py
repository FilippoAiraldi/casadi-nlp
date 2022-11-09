from functools import cached_property, wraps
from typing import Any, Callable, Dict, Tuple, Union, Optional, Literal
import numpy as np
import casadi as cs
from casadi.tools import struct_symSX, struct_SX, entry
from casadi.tools.structure3 import DMStruct


def is_casadi_object(obj: Any) -> bool:
    '''Checks if the object belongs to the CasADi module.

    Parameters
    ----------
    obj : Any
        Any type of object.

    Returns
    -------
    bool
        A flag that states whether the object belongs to CasADi or not.
    '''

    # see https://stackoverflow.com/a/52783240/19648688

    if not hasattr(obj, '__module__'):
        return False
    module: str = obj.__module__.split('.')[0]
    return module == cs.__name__


def cache_clearer(*properties: cached_property) -> Callable:
    '''Decorator that allows to enhance a method with the ability, when
    called, to clear the cached of some target properties. This is especially
    useful to reset the cache of a given cached property when another method
    makes changes to the underlying data, thus compromising the cached results.

    Parameters
    ----------
    properties : cached_property
        The cached properties to be reset in this decorator.

    Returns
    -------
    decorated_func : Callable
        Returns the function wrapped with this decorator.

    Raises
    ------
    TypeError
        Raises if the given properties are not instances of
        `functools.cached_property`.
    '''
    # for now, the class handles only cached_properties, but it can be extended
    # to reset also other types of caches.
    if any(not isinstance(p, cached_property) for p in properties):
        raise TypeError('The specified properties must be an instance of '
                        '`functools.cached_property`')

    # use a double decorator as it is a trick to allow passing arguments to it
    def actual_decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            self = args[0]
            for property in properties:
                n = property.attrname
                if n is not None and n in self.__dict__:
                    del self.__dict__[n]
            return func(*args, **kwargs)
        return wrapper
    return actual_decorator


def dict2struct(
    dict: Dict[str, Union[cs.DM, cs.SX, cs.MX]],
    entry_type: Literal['sym', 'expr'] = 'sym'
) -> Union[DMStruct, struct_symSX, struct_SX, Dict[str, cs.MX]]:
    '''Attempts to convert a dictionary of CasADi matrices to a struct. The
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
    '''

    o = next(iter(dict.values()))
    if isinstance(o, cs.DM):
        dummy = struct_symSX([
            entry(name, shape=p.shape) for name, p in dict.items()])
        return dummy(cs.vertcat(*map(cs.vec, dict.values())))
    elif isinstance(o, cs.SX):
        struct = struct_symSX if entry_type == 'sym' else struct_SX
        return struct([
            entry(name, **{entry_type: p}) for name, p in dict.items()])
    else:
        return dict.copy()


def np_random(
    seed: Optional[int] = None
) -> Tuple[np.random.Generator, int]:
    '''Generates a random number generator from the seed and returns the
    Generator and seed.

    Full credit to [OpenAI implementation](https://github.com/openai/gym/blob/6a04d49722724677610e36c1f92908e72f51da0c/gym/utils/seeding.py).

    Parameters
    ----------
    seed : int, optional
        The seed used to create the generator.

    Returns
    -------
    Tuple[Generator, int]
        The generator and resulting seed.

    Raises
    ------
    ValueError
        Seed must be a non-negative integer or omitted.
    '''
    if seed is not None and not (isinstance(seed, int) and seed >= 0):
        raise ValueError(
            f'Seed must be a non-negative integer or omitted, not {seed}')
    seed_seq = np.random.SeedSequence(seed)
    np_seed = seed_seq.entropy
    rng = np.random.Generator(np.random.PCG64(seed_seq))
    return rng, np_seed


def array2cs(x: np.ndarray) -> Union[cs.SX, cs.MX]:
    '''Converts numpy array `x` of scalar symbolic variable to a single 
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
    '''
    ndim = x.ndim
    if ndim == 0:
        raise ValueError('Cannot convert empty arrays.')
    elif ndim == 1:
        o = x[0]
        x = x.reshape(-1, 1)
    elif ndim == 2:
        o = x[0, 0]
    else:
        raise ValueError(
            'Can only convert 1D and 2D arrays to CasADi SX or MX.')
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
    '''Converts casadi symbolic variable `x` to a numpy array of scalars. 
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
    '''
    shape = x.shape
    y = np.empty(shape, dtype=object)
    for i in np.ndindex(shape):
        y[i] = x[i]
    return y


def hojacobian(ex: Union[cs.MX, cs.SX], x: Union[cs.MX, cs.SX]) -> np.ndarray:
    '''Computes jacobian on higher-order matrices, i.e., with an output in 4D.

    Parameters
    ----------
    ex : Union[cs.MX, cs.SX]
        The expression to compute the jacobian for. Can be a matrix.
    x : Union[cs.MX, cs.SX]
        The variable to differentiate with respect to. Can be a matrix.

    Returns
    -------
    np.ndarray
        A 4D array of objects, where each entry `(i,j,k,m)` is the derivative
        of `ex[i,j]` w.r.t. `x[k,m]`.
    '''
    return cs2array(
        cs.jacobian(cs.vec(ex), cs.vec(x))
    ).reshape(ex.shape + x.shape, order='F')
