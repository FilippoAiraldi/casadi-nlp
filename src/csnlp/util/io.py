import pickle
from pickletools import optimize
from typing import Any, Dict


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
