import pickle
from copy import _reconstruct, deepcopy
from os.path import splitext
from pickletools import optimize
from typing import TYPE_CHECKING, Any, Callable, Literal, Optional, TypeVar

from ..core.cache import invalidate_caches_of

if TYPE_CHECKING:
    from scipy.io.matlab import mat_struct


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
    """Checks whether the object is pickeable.

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
    """Class that defines a `__getstate__` that is compatible with both `deepcopy` and
    `pickle`, as well as any other operation that requires the instance's state.

    When pickled, states that cannot be pickled (e.g., CasADi objects) are automatically
    removed."""

    def copy(self: T, invalidate_caches: bool = True) -> T:
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
        new = deepcopy(self)
        if invalidate_caches:
            invalidate_caches_of(new)
        return new

    def __deepcopy__(self: T, memo: Optional[dict[int, list[Any]]] = None) -> T:
        """Returns a deepcopy of the object."""
        rv = self.__reduce_ex__(4)
        if isinstance(rv, str):
            return self
        assert len(rv) < 6 or rv[5] is None, "Unexpected reductor callable."
        # overwrite the filtered state with its full version
        fullstate = self.__getstate__(True)
        new_rv = (*rv[:2], fullstate, *rv[3:])
        return _reconstruct(self, memo, *new_rv)

    def __getstate__(self: T, fullstate: bool = False) -> Optional[dict[str, Any]]:
        """Returns the instance's state to be pickled/deepcopied."""
        # https://docs.python.org/3/library/pickle.html#pickle-inst
        if not (hasattr(self, "__dict__") and self.__dict__.keys()):
            return None
        state = self.__dict__.copy()
        if fullstate:
            return state
        state_items_copy = list(state.items())
        for attr, val in state_items_copy:
            if is_casadi_object(val) or not is_pickleable(val):
                state.pop(attr, None)
        return state


_COMPRESSION_EXTS: dict[str, Optional[str]] = {
    ".pkl": None,
    ".xz": "lzma",
    ".pbz2": "bz2",
    ".gz": "gzip",
    ".bt": "brotli",
    ".bl2": "blosc2",
    ".mat": "matlab",
}


def save(
    filename: str,
    compression: Optional[
        Literal["lzma", "bz2", "gzip", "brotli", "blosc2", "matlab"]
    ] = None,
    **data: Any,
) -> str:
    """Saves data to a (possibly compressed) file. Inspired by
     - https://stackoverflow.com/a/57983757/19648688,
     - https://stackoverflow.com/a/8832212/19648688.

    Parameters
    ----------
    filename : str
        The name of the file to save to. If the filename does not end in the correct
        extension, then it is automatically added. The extensions are
         - "pickle": .pkl
         - "lzma": .xz
         - "bz2": .pbz2
         - "gzip": .gz
         - "brotli": .bt
         - "blosc2": .bl2
         - "matlab": .mat
    **data : dict
        Any data to be saved to a file.
    compression : {"lzma", "bz2", "gzip", "brotli", "blosc2", "matlab"]}
        Type of compression to apply to the file. Note that `brotli` and `blosc2`
        require the installation of the corresponding pip package. `matlab` requires the
        installation of `scipy` to save as .mat file. By default, pickle is used.

    Returns
    -------
    filename : str
        The complete name of the file where the data was written to.
    """

    actual_ext = splitext(filename)[1]
    if compression is None:
        compression = _COMPRESSION_EXTS.get(actual_ext)

    open_fun: Callable
    compress_fun: Callable
    if compression is None:
        expected_ext = ".pkl"
        open_fun = open
        compress_fun = lambda o: o  # noqa E731
    elif compression == "lzma":
        import lzma

        expected_ext = ".xz"
        open_fun = lzma.open
        compress_fun = lambda o: o  # noqa E731
    elif compression == "bz2":
        import bz2

        expected_ext = ".pbz2"
        open_fun = bz2.BZ2File
        compress_fun = lambda o: o  # noqa E731
    elif compression == "gzip":
        import gzip

        expected_ext = ".gz"
        open_fun = gzip.open
        compress_fun = lambda o: o  # noqa E731
    elif compression == "brotli":
        import brotli

        expected_ext = ".bt"
        open_fun = open
        compress_fun = brotli.compress
    elif compression == "blosc2":
        import blosc2

        expected_ext = ".bl2"
        open_fun = open
        compress_fun = blosc2.compress
    elif compression == "matlab":
        expected_ext = ".mat"
    else:
        raise ValueError(f"Unknown compression method {compression}.")

    if expected_ext != actual_ext:
        filename += expected_ext

    # address first special cases that do not adhere to the open/compress scheme
    if compression == "matlab":
        import scipy.io as spio

        spio.savemat(filename, data, do_compression=True, oned_as="column")

    # address all other cases that do adhere to the open/compress scheme
    else:
        pickled = pickle.dumps(data)
        optimized = optimize(pickled)
        compressed = compress_fun(optimized)
        with open_fun(filename, "wb") as f:
            f.write(compressed)
    return filename


def load(filename: str) -> dict[str, Any]:
    """Loads data from a (possibly compressed) file.

    Parameters
    ----------
    filename : str, optional
        The name of the file to load. If the filename does not end in a known extension,
        then it fails. The known extensions are
         - "pickle": .pkl
         - "lzma": .xz
         - "bz2": .pbz2
         - "gzip": .gz
         - "brotli": .bt
         - "blosc2": .bl2
         - "matlab": .mat

    Returns
    -------
    data : dict
        The saved data in the shape of a dictionary.
    """
    ext = splitext(filename)[1]
    compression = _COMPRESSION_EXTS[ext]

    open_fun: Callable
    decompress_fun: Callable
    if compression is None:
        open_fun = open
        decompress_fun = pickle.loads
    elif compression == "lzma":
        import lzma

        open_fun = lzma.open
        decompress_fun = pickle.loads
    elif compression == "bz2":
        import bz2

        open_fun = bz2.BZ2File
        decompress_fun = pickle.loads
    elif compression == "gzip":
        import gzip

        open_fun = gzip.open
        decompress_fun = pickle.loads
    elif compression == "brotli":
        import brotli

        open_fun = open
        decompress_fun = lambda o: pickle.loads(brotli.decompress(o))  # noqa E731
    elif compression == "blosc2":
        import blosc2

        open_fun = open
        decompress_fun = lambda o: pickle.loads(blosc2.decompress(o))  # noqa E731
    elif compression != "matlab":
        raise ValueError(f"Unknown file extension {ext}.")

    # address first special cases that do not adhere to the open/decompress scheme
    if compression == "matlab":
        import scipy.io as spio

        data = _check_mat_keys(
            spio.loadmat(filename, struct_as_record=False, squeeze_me=True),
            spio.matlab.mat_struct,
        )

    # address all other cases that do adhere to the open/decompress scheme
    else:
        with open_fun(filename, "rb") as f:
            data = decompress_fun(f.read())

    # if it is only a dict with one key, return the value of the key directly.
    if isinstance(data, dict) and len(data.keys()) == 1:
        data = data[next(iter(data.keys()))]
    return data


def _check_mat_keys(dictionary: dict, mat_struct_type: type) -> dict:
    """Internal utility to check if entries in dictionary are mat-objects. If yes,
    todict is called to change them to nested dictionaries."""

    def _todict_recursive(matobj: "mat_struct") -> dict:
        dictionary = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            dictionary[strg] = (
                _todict_recursive(elem) if isinstance(elem, mat_struct_type) else elem
            )
        return dictionary

    for bad_key in ("__header__", "__version__", "__globals__"):
        dictionary.pop(bad_key, None)
    for key in dictionary:
        if isinstance(dictionary[key], mat_struct_type):
            dictionary[key] = _todict_recursive(dictionary[key])
    return dictionary
