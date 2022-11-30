import pickle
from contextlib import contextmanager
from copy import deepcopy
from pickletools import optimize
from typing import Any, Dict, Optional, TypeVar
from warnings import warn

from csnlp.util.data import is_casadi_object


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


T = TypeVar("T", bound="SupportsDeepcopyAndPickle")


class SupportsDeepcopyAndPickle:
    """Class that defines a `__getstate__` that is compatible with both
    `deepcopy` and `pickle`, as well as any other operation that requires the instance's
    state.

    When pickled, use the context manager `pickleable` in order to automatically remove
    states that cannot be pickled (e.g., CasADi objects); otherwise, use `fullstate`
    for, e.g., `deepcopy`ing the class instance."""

    _GETFULLSTATE: Optional[bool] = None

    @contextmanager
    def pickleable(self) -> None:
        """Context manager that makes the class pickleable by automatically removing
        unpickleable states (opposite of `fullstate`)."""
        self._GETFULLSTATE = False
        yield
        self._GETFULLSTATE = None

    @contextmanager
    def fullstate(self) -> None:
        """Context manager that makes the class return the full state without removing
        unpickleable states (opposite of `pickleable`)."""
        self._GETFULLSTATE = True
        yield
        self._GETFULLSTATE = None

    def copy(self: T) -> T:
        """Creates a deepcopy of this instance.

        Returns
        -------
        `SupportsDeepcopyAndPickle` or its subclass
            A deepcopy of this instance.
        """
        with self.fullstate():
            return deepcopy(self)

    def __getstate__(self) -> Dict[str, Any]:
        """Returns the instance's state to be pickled or copied."""
        if self._GETFULLSTATE is None:
            raise RuntimeError(
                f"Trying to get the state of {self.__class__.__name__} without using "
                "context manager `pickleable` or `fullstate`."
            )
        state = self.__dict__.copy()
        state["_GETFULLSTATE"] = None
        if not self._GETFULLSTATE:
            warn(
                f"to pickle {self.__class__.__name__} all references to CasADi and "
                "unpickleable objects are removed.",
                RuntimeWarning,
            )
            for attr, val in self.__dict__.items():
                if is_casadi_object(val) or not is_pickleable(val):
                    state.pop(attr)
        return state


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
