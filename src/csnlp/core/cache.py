"""A collection of methods to handle caching in the package. In particular, it offers a
decorator :func:`invalidate_cache` that allows to invalidate the cache of a given set of
other cached properties or methods when the decorated method is invoked, as well as a
function :func:`invalidate_caches_of` that allows to invalidate the cache of a given
object on the fly."""

import functools
import inspect
from typing import Callable


def _is_cached_property(c: Callable) -> bool:
    """Returns True if the callable is a cached property."""
    return isinstance(c, functools.cached_property)


def _is_lru_cache(c: Callable) -> bool:
    """Returns True if the callable is a lru cache."""
    return hasattr(c, "cache_info") or hasattr(c, "cache_clear")


def invalidate_cache(*callables: Callable) -> Callable:
    r"""Decorator that allows to enhance a function or method with the ability, when
    called, to invalidate and clear the cached of some other target methods/properties.
    This is especially useful to reset the cache of a given cached method/property when
    another method makes changes to the underlying data, thus compromising the cached
    results.

    The decorator works only for the built-in :func:`functools.cached_property` and
    :func:`functools.lru_cache`.

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
        Raises if the given inputs are not instances of
        :func:`functools.cached_property` or :func:`functools.lru_cache`.

    Notes
    -----
    The wrapper can invalidate other :func:`functools.cached_property` instances, but
    for doing so it assumes the instance of the object (which the property to invalidate
    belongs to) is the first argument of the wrapped method. For lru caches the issue
    does not subsist.
    """
    if not callables:
        raise ValueError("No callables were passed for cache invalidation.")
    cached_properties: list[functools.cached_property] = []
    lru_caches: list[functools._lru_cache_wrapper] = []
    for p in callables:
        if _is_cached_property(p):
            cached_properties.append(p)
        elif _is_lru_cache(p):
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

        def invalidate_cached_properties(self: object) -> None:
            propname = prop.attrname
            if propname in self.__dict__:
                del self.__dict__[propname]

    else:

        def invalidate_cached_properties(self: object) -> None:
            for prop in cached_properties:
                propname = prop.attrname
                if propname in self.__dict__:
                    del self.__dict__[propname]

    Nlru = len(lru_caches)
    if Nlru == 0:
        invalidate_lru_caches = None
    elif Nlru == 1:
        lru_cache = lru_caches[0]

        def invalidate_lru_caches() -> None:
            lru_cache.cache_clear()

    else:

        def invalidate_lru_caches() -> None:
            for lru_cache in lru_caches:
                lru_cache.cache_clear()

    def decorating_function(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
            # if type annotated, breaks docs in `mpcrl`
            if invalidate_cached_properties is not None and args:
                invalidate_cached_properties(args[0])
            if invalidate_lru_caches is not None:
                invalidate_lru_caches()
            return func(*args, **kwargs)

        return wrapper

    return decorating_function


def invalidate_caches_of(obj: object) -> None:
    """Similar to the decorator :func:`invalidate_cache`, but clears the case of
    the given object only once.

    Parameters
    ----------
    obj : object
        The object whose caches are to be cleared. If no cached properties or lru caches
        are found, the function does nothing.
    """
    # basically do again what csnlp.core.cache.invalidate_cache does
    for membername, member in inspect.getmembers(
        type(obj), predicate=lambda m: _is_cached_property(m) or _is_lru_cache(m)
    ):
        if _is_cached_property(member):
            if membername in obj.__dict__:
                del obj.__dict__[membername]
        elif _is_lru_cache(member):
            getattr(obj, membername).cache_clear()
