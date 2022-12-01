from functools import _lru_cache_wrapper, cached_property, wraps
from typing import Callable, List, Optional, Tuple

import numpy as np


def invalidate_cache(*callables: Callable) -> Callable:
    """Decorator that allows to enhance a function or method with the ability, when
    called, to invalidate and clear the cached of some other target methods/properties.
    This is especially useful to reset the cache of a given cached method/property when
    another method makes changes to the underlying data, thus compromising the cached
    results.

    Note: the wrapper can invalidate other cached properties, but for doing so it
    assumes the instance of the object (which the property to invalidate belongs to) is
    the first argument of the wrapped method. For lru caches the issue does not subsist.

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
    """
    cached_properties: List[cached_property] = []
    lru_caches: List[_lru_cache_wrapper] = []
    for p in callables:
        if isinstance(p, cached_property):
            cached_properties.append(p)
        elif hasattr(p, "cache_clear"):
            lru_caches.append(p)
        else:
            raise TypeError(
                "Expected cached properties or lru wrappers; got "
                f"{p.__class__.__name__} instead."
            )

    def decorating_function(func):
        def invalidate_cached_properties(self):
            for prop in cached_properties:
                n = prop.attrname
                if n is not None and n in self.__dict__:
                    del self.__dict__[n]

        def invalidate_lru_caches():
            for lru in lru_caches:
                lru.cache_clear()

        @wraps(func)
        def wrapper(*args, **kwargs):
            if args:
                invalidate_cached_properties(args[0])  # assume self is args[0]
            invalidate_lru_caches()
            return func(*args, **kwargs)

        wrapper.cached_properties = cached_properties
        wrapper.lru_caches = lru_caches
        wrapper.invalidate_cached_properties = invalidate_cached_properties
        wrapper.invalidate_lru_caches = invalidate_lru_caches
        return wrapper

    return decorating_function


def np_random(seed: Optional[int] = None) -> Tuple[np.random.Generator, int]:
    """Generates a random number generator from the seed and returns the Generator and
    seed.

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
    """
    if seed is not None and not (isinstance(seed, int) and seed >= 0):
        raise ValueError(f"Seed must be a non-negative integer or omitted, not {seed}")
    seed_seq = np.random.SeedSequence(seed)
    np_seed = seed_seq.entropy
    rng = np.random.Generator(np.random.PCG64(seed_seq))
    return rng, np_seed
