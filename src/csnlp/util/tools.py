from functools import cached_property, wraps
from typing import Callable, Tuple, Optional
import numpy as np


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
