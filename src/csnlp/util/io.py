import pickle
from pickletools import optimize
from traceback import walk_stack
from typing import Any, Dict
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


class SupportsDeepcopyAndPickle:
    """Class that defines a `__getstate__` that is compatible both with
    `deepcopy` and `pickle`.

    When pickled, it automatically removes states that
    cannot be pickled (e.g., CasADi objects); otherwise, the full state is
    returned. Exceptions for which the full state must be returned by
    `__getstate__` can be extended via the attribute `nonpickle_exceptions`."""

    nonpickle_exceptions = {"deepcopy"}

    def __getstate__(self) -> Dict[str, Any]:
        """Returns the instance's state to be pickled or copied."""
        state = self.__dict__.copy()
        for frame, _ in walk_stack(None):
            if frame.f_code.co_name in self.nonpickle_exceptions:
                return state
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
