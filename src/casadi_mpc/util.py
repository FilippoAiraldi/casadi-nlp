from functools import cached_property, wraps
from typing import Any, Callable
import casadi as cs


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


def cached_property_reset(cproperty: cached_property) -> Callable:
    '''Decorator that allows to enhance a method with the ability, when called,
    to clear the cached of a target property. This is especially useful to 
    reset the cache of a given cached property when a method makes changes to 
    the underlying data, thus compromising the cached results.

    Returns
    -------
    decorated_func : Callable
        Returns the function wrapped with this decorator.

    Raises
    ------
    TypeError
        Raises if the given property is not an instance of 
        `functools.cached_property`.
    '''

    if not isinstance(cproperty, cached_property):
        raise TypeError(
            'The specified property must be an instance of '
            f'`functools.cached_property`; got {type(cproperty)} instead.')

    # use a double decorator as it is a trick to allow passing arguments to it
    def actual_decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            self = args[0]
            n = cproperty.attrname
            if n is not None and n in self.__dict__:
                del self.__dict__[n]
            return func(*args, **kwargs)
        return wrapper
    return actual_decorator
