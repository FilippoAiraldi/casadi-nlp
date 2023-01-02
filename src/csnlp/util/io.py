import pickle
from contextlib import contextmanager
from copy import deepcopy
from functools import cached_property
from inspect import getmembers
from itertools import chain
from pickletools import optimize
from typing import Any, Dict, Iterable, Iterator, Optional, Tuple, TypeVar, Union
from warnings import warn


def is_casadi_object(obj: Any) -> bool:
    """Checks if the object belongs to the CasADi module.

    See https://stackoverflow.com/a/52783240/19648688 for more details.

    Parameters
    ----------
    obj : Any
        Any type of object.

    Returns
    -------
    bool
        A flag that states whether the object belongs to CasADi or not.
    """
    if not hasattr(obj, "__module__"):
        return False
    module: str = obj.__module__.split(".")[0]
    return module == "casadi"


def is_pickleable(obj: Any) -> bool:
    """
    Checks whether the object is pickeable.

    Parameters
    ----------
    obj : Any
        The object to test against.

    Returns
    -------
    pickleable : bool
        A flag indicating if pickleable or not.
    """
    try:
        pickle.dumps(obj)
        return True
    except Exception:
        return False


def _get_dict_state(
    obj: "SupportsDeepcopyAndPickle",
    attributes: Iterable[str],
    remove_None: bool,
) -> Dict[str, Any]:
    """Internal utility for SupportsDeepcopyAndPickle."""
    state: Dict[str, Any] = {attr: getattr(obj, attr, None) for attr in attributes}
    if "_GETFULLSTATE" in state:
        state["_GETFULLSTATE"] = None
    if not obj._GETFULLSTATE:
        for attr in list(state.keys()):
            val = state[attr]
            if (
                (remove_None and val is None)
                or is_casadi_object(val)
                or not is_pickleable(val)
            ):
                state.pop(attr, None)
    return state


SymType = TypeVar("SymType", bound="SupportsDeepcopyAndPickle")


class SupportsDeepcopyAndPickle:
    """Class that defines a `__getstate__` that is compatible with both
    `deepcopy` and `pickle`, as well as any other operation that requires the instance's
    state.

    When pickled, use the context manager `pickleable` in order to automatically remove
    states that cannot be pickled (e.g., CasADi objects); otherwise, use `fullstate`
    for, e.g., `deepcopy`ing the class instance."""

    _GETFULLSTATE: Optional[bool] = None

    @contextmanager
    def pickleable(self) -> Iterator[None]:
        """Context manager that makes the class pickleable by automatically removing
        unpickleable states (opposite of `fullstate`)."""
        self._GETFULLSTATE = False
        yield
        self._GETFULLSTATE = None

    @contextmanager
    def fullstate(self) -> Iterator[None]:
        """Context manager that makes the class return the full state without removing
        unpickleable states (opposite of `pickleable`)."""
        self._GETFULLSTATE = True
        yield
        self._GETFULLSTATE = None

    def copy(self: SymType, invalidate_caches: bool = True) -> SymType:
        """Creates a deepcopy of this instance.

        Parameters
        ----------
        invalidate_caches : bool, optional
            If `True`, methods decorated with `csnlp.util.funcs.invalidate_cache` are
            called to clear cached properties/lru caches in the copied instance.
            Otherwise, caches in the copy are not invalidated. By default, `True`.

        Returns
        -------
        `SupportsDeepcopyAndPickle` or its subclass
            A deepcopy of this instance.
        """
        with self.fullstate():
            new = deepcopy(self)
        if invalidate_caches:
            # basically do again what csnlp.util.funcs.invalidate_cache does
            for membername, member in getmembers(type(new)):
                if hasattr(member, "cache_clear"):
                    getattr(new, membername).cache_clear()
                elif isinstance(member, cached_property):
                    if membername in new.__dict__:
                        del new.__dict__[membername]
        return new

    def __getstate__(
        self,
    ) -> Union[None, Dict[str, Any], Tuple[Optional[Dict[str, Any]], Dict[str, Any]]]:
        """Returns the instance's state to be pickled or copied."""
        # https://docs.python.org/3/library/pickle.html#pickle-inst
        if self._GETFULLSTATE is None:
            raise RuntimeError(
                f"Trying to get the state of {self.__class__.__name__} without using "
                "context manager `pickleable` or `fullstate`."
            )
        elif not self._GETFULLSTATE:
            warn(
                f"to deepcopy/pickle {self.__class__.__name__}, all references to "
                "CasADi and unpickleable objects are removed from the state.",
                RuntimeWarning,
            )
        dictstate = (
            _get_dict_state(self, self.__dict__.keys(), False)
            if hasattr(self, "__dict__")
            else None
        )
        slots = chain.from_iterable(
            getattr(cls, "__slots__", []) for cls in type(self).__mro__
        )
        slotstate = _get_dict_state(self, slots, True) if slots else None
        return (dictstate, slotstate) if slotstate is not None else dictstate


def save(filename: str, **data: Any) -> str:
    """
    Saves data to a pickle file.

    Parameters
    ----------
    filename : str
        The name of the file to save to. If the filename does not end in
        `'.pkl'`, then this extension is automatically added.
    **data : dict
        Any data to be saved to the pickle file.

    Returns
    -------
    filename : str
        The complete name of the file where the data was written to.
    """
    if not filename.endswith(".pkl"):
        filename = f"{filename}.pkl"
    with open(filename, "wb") as f:
        pickled = pickle.dumps(data)
        optimized = optimize(pickled)
        f.write(optimized)
    return filename


def load(filename: str) -> Dict[str, Any]:
    """
    Loads data from pickle.

    Parameters
    ----------
    filename : str, optional
        The name of the file to load. If the filename does not end in `'.pkl'`,
        then this extension is automatically added.

    Returns
    -------
    data : dict
        The saved data in the shape of a dictionary.
    """
    if not filename.endswith(".pkl"):
        filename = f"{filename}.pkl"
    with open(filename, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict) and len(data.keys()) == 1:
        data = data[next(iter(data.keys()))]
    return data
