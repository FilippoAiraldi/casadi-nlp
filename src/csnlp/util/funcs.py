from functools import _lru_cache_wrapper, cached_property, wraps
from typing import Callable, List, Optional, Tuple, Union

import numpy as np


def cache_clearer(
    *callables: Union[cached_property, _lru_cache_wrapper]
) -> Callable:
    '''Decorator that allows to enhance a method with the ability, when
    called, to clear the cached of some target methods/properties. This is
    especially useful to reset the cache of a given cached method/property when
    another method makes changes to the underlying data, thus compromising the
    cached results.

    Parameters
    ----------
    callables : cached_property or lru cache wrapper
        The cached properties or methods to be reset in this decorator.

    Returns
    -------
    decorated_func : Callable
        Returns the function wrapped with this decorator.

    Raises
    ------
    TypeError
        Raises if the given inputs are not instances of
        `functools.cached_property` or `functools._lru_cache_wrapper`.
    '''
    cps: List[cached_property] = []
    lrus: List[_lru_cache_wrapper] = []
    for p in callables:
        if isinstance(p, cached_property):
            cps.append(p)
        elif isinstance(p, _lru_cache_wrapper):
            lrus.append(p)
        else:
            raise TypeError('Expected cached properties or lru wrappers; got '
                            f'{p.__class__.__name__} instead.')

    def actual_decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            self = args[0]
            for prop in cps:
                n = prop.attrname
                if n is not None and n in self.__dict__:
                    del self.__dict__[n]
            for lru in lrus:
                lru.cache_clear()
            return func(*args, **kwargs)
        return wrapper

    return actual_decorator


def np_random(seed: Optional[int] = None) -> Tuple[np.random.Generator, int]:
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
