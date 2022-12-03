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
    decorating_function : Callable
        Returns the function wrapped with this decorator.

    Raises
    ------
    ValueError
        Raises if no callable is passed to the function.
    TypeError
        Raises if the given inputs are not instances of `functools.cached_property` or
        `functools._lru_cache_wrapper`.
    """
    if not callables:
        raise ValueError("No callables were passed for cache invalidation.")
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

    Ncp = len(cached_properties)
    if Ncp == 0:
        invalidate_cached_properties = None
    elif Ncp == 1:
        prop = cached_properties[0]

        def invalidate_cached_properties(self):
            propname = prop.attrname
            if propname in self.__dict__:
                del self.__dict__[propname]

    else:

        def invalidate_cached_properties(self):
            for prop in cached_properties:
                propname = prop.attrname
                if propname in self.__dict__:
                    del self.__dict__[propname]

    Nlru = len(lru_caches)
    if Nlru == 0:
        invalidate_lru_caches = None
    elif Nlru == 1:
        lru_cache = lru_caches[0]

        def invalidate_lru_caches():
            lru_cache.cache_clear()

    else:

        def invalidate_lru_caches():
            for lru_cache in lru_caches:
                lru_cache.cache_clear()

    def decorating_function(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if invalidate_cached_properties is not None and args:
                invalidate_cached_properties(args[0])
            if invalidate_lru_caches is not None:
                invalidate_lru_caches()
            return func(*args, **kwargs)

        return wrapper

    return decorating_function


def invoke(callable: Callable, after: bool = True) -> Callable:
    """Decorator that, when the decorated method is called, invokes the given callable.

    Parameters
    ----------
    callable : cached_property or lru cache wrapper
        The callable to be invoked whenever the decorated function is called. It is
        called with the same `*args` and `**kwargs` of the decorated function.
    after : bool, optional
        If `after=True`, `callable` is invoked after the wrapped function; otherwise, it
        is called before. By default, `True`.

    Returns
    -------
    decorating_function : Callable
        Returns the function wrapped with this decorator.
    """

    def decorating_function(func):
        if after:

            @wraps(func)
            def wrapper(*args, **kwargs):
                out = func(*args, **kwargs)
                callable(*args, **kwargs)
                return out

            return wrapper

        else:

            @wraps(func)
            def wrapper(*args, **kwargs):
                callable(*args, **kwargs)
                return func(*args, **kwargs)

            return wrapper

    return decorating_function


def np_random(seed: Optional[int] = None) -> Tuple[np.random.Generator, int]:
    """Generates a random number generator from the seed and returns the Generator and
    seed.

    Full credit to OpenAI implementation at
    https://github.com/openai/gym/blob/master/gym/utils/seeding.py.


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
