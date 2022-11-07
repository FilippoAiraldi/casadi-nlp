from functools import cached_property, wraps
from typing import Any, Callable, Dict, Tuple, Union, Optional
import numpy as np
import casadi as cs
from casadi.tools import struct_symSX, entry
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


def cached_property_reset(*properties: cached_property) -> Callable:
    '''Decorator that allows to enhance a method with the ability, when called,
    to clear the cached of some target properties. This is especially useful to 
    reset the cache of a given cached property when a method makes changes to 
    the underlying data, thus compromising the cached results.

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
    dict: Dict[str, Union[cs.DM, cs.SX, cs.MX]]
) -> Union[DMStruct, struct_symSX, Dict[str, cs.MX]]:
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

    Returns
    -------
    Union[DMStruct, struct_symSX, Dict[str, cs.MX]]
        Either a structure generated from `dict`, or a copy of `dict` itself.
    '''

    o = next(iter(dict.values()))
    if isinstance(o, cs.DM):
        dummy = struct_symSX([
            entry(name, shape=p.shape) for name, p in dict.items()])
        return dummy(cs.vertcat(*map(cs.vec, dict.values())))
    elif isinstance(o, cs.SX):
        return struct_symSX([entry(name, sym=p) for name, p in dict.items()])
    else:
        return dict.copy()


def np_random(seed: int = None) -> Tuple[np.random.Generator, Optional[int]]:
    # https://github.com/openai/gym/blob/6a04d49722724677610e36c1f92908e72f51da0c/gym/utils/seeding.py
    '''Generates a random number generator from the seed and returns the 
    Generator and seed.

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
