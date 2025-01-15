"""A collection of utilities for input/output operations. The goals of this module are:

- compatibility of pickling/deepcopying with CasADi objects and classes that hold such
  objects (since these are often not picklable).
- saving and loading data to/from files, possibly compressed.
"""

import pickle
from copy import _reconstruct
from copy import deepcopy as _deepcopy
from functools import partial
from os.path import splitext as _splitext
from pickletools import optimize as _optimize
from typing import TYPE_CHECKING
from typing import Any as _Any
from typing import Callable, Literal, Optional
from typing import TypeVar as _TypeVar

from ..core.cache import invalidate_caches_of as _invalidate_caches_of

if TYPE_CHECKING:
    from scipy.io.matlab import mat_struct


def is_casadi_object(obj: _Any) -> bool:
    """Checks if the object belongs to the CasADi module.

    See `this thread <https://stackoverflow.com/a/52783240/19648688>`_ for more
    details.

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


def is_pickleable(obj: _Any) -> bool:
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


T = _TypeVar("T", bound="SupportsDeepcopyAndPickle")


class SupportsDeepcopyAndPickle:
    """Class that defines a :meth:`__getstate__` that is compatible with both
    :func:`deepcopy` and :mod:`pickle`, as well as any other operation that requires the
    instance's state.

    When pickled, states that cannot be pickled (e.g., CasADi objects) are automatically
    removed.
    """

    def copy(self: T, invalidate_caches: bool = True) -> T:
        """Creates a deepcopy of this instance.

        Parameters
        ----------
        invalidate_caches : bool, optional
            If ``True``, methods decorated with
            :func:`csnlp.core.cache.invalidate_cache` are called to clear cached
            properties/lru caches in the copied instance. Otherwise, caches in the copy
            are not invalidated. By default, ``True``.

        Returns
        -------
        Instance of :class:`SupportsDeepcopyAndPickle` or its subclass
            A deepcopy of this instance.
        """
        new = _deepcopy(self)
        if invalidate_caches:
            _invalidate_caches_of(new)
        return new

    def __deepcopy__(self: T, memo: Optional[dict[int, list[_Any]]] = None) -> T:
        """Returns a deepcopy of the object."""
        rv = self.__reduce_ex__(4)
        if isinstance(rv, str):
            return self
        assert len(rv) < 6 or rv[5] is None, "Unexpected reductor callable."
        # overwrite the filtered state with its full version
        fullstate = self.__getstate__(True)
        new_rv = (*rv[:2], fullstate, *rv[3:])
        return _reconstruct(self, memo, *new_rv)

    def __getstate__(self: T, fullstate: bool = False) -> Optional[dict[str, _Any]]:
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
    ".npz": "numpy",
}


def save(
    filename: str,
    compression: Optional[
        Literal["lzma", "bz2", "gzip", "brotli", "blosc2", "matlab", "numpy"]
    ] = None,
    **data: _Any,
) -> str:
    """Saves data to a (possibly compressed) file. Inspired by
    `this discussion <https://stackoverflow.com/a/57983757/19648688>`_
    and `this other discussion <https://stackoverflow.com/a/8832212/19648688>`_.

    Parameters
    ----------
    filename : str
        The name of the file to save to. If the filename does not end in the correct
        extension, then it is automatically added. The extensions are

        - ``"pickle"``: .pkl
        - ``"lzma"``: .xz
        - ``"bz2"``: .pbz2
        - ``"gzip"``: .gz
        - ``"brotli"``: .bt
        - ``"blosc2"``: .bl2
        - ``"matlab"``: .mat
        - ``"numpy"``: .npz.
    **data : dict
        Any data to be saved to a file.
    compression : {"lzma", "bz2", "gzip", "brotli", "blosc2", "matlab", "npz"}
        Type of compression to apply to the file. By default, `pickle` is used.

    Returns
    -------
    filename : str
        The complete name of the file where the data was written to.

    Notes
    -----
    Note that the compression types ``brotli`` and ``blosc2`` require the installation
    of the corresponding pip packages (see `Brotli <https://github.com/google/brotli>`_
    and `Blosc2 <https://www.blosc.org/python-blosc/python-blosc.html>`_).
    ``matlab`` requires instead the installation of :mod:`scipy` to save as .mat file
    (see :func:`scipy.io.savemat` and :func:`scipy.io.loadmat` for more details).
    """

    actual_ext = _splitext(filename)[1]
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
        compress_fun = partial(blosc2.compress, typesize=None)
    elif compression == "matlab":
        expected_ext = ".mat"
    elif compression == "numpy":
        expected_ext = ".npz"
    else:
        raise ValueError(f"Unknown compression method {compression}.")

    if expected_ext != actual_ext:
        filename += expected_ext

    # address first special cases that do not adhere to the open/compress scheme
    if compression == "matlab":
        import scipy.io as spio

        spio.savemat(filename, data, do_compression=True, oned_as="column")

    elif compression == "numpy":
        import numpy as np

        np.savez_compressed(filename, **data)

    # address all other cases that do adhere to the open/compress scheme
    else:
        pickled = pickle.dumps(data)
        optimized = _optimize(pickled)
        compressed = compress_fun(optimized)
        with open_fun(filename, "wb") as f:
            f.write(compressed)
    return filename


def load(filename: str) -> dict[str, _Any]:
    """Loads data from a (possibly compressed) file.

    Parameters
    ----------
    filename : str, optional
        The name of the file to load. If the filename does not end in a known extension,
        then it fails. The known extensions are

        - ``"pickle"``: .pkl
        - ``"lzma"``: .xz
        - ``"bz2"``: .pbz2
        - ``"gzip"``: .gz
        - ``"brotli"``: .bt
        - ``"blosc2"``: .bl2
        - ``"matlab"``: .mat
        - ``"numpy"``: .npz.

    Returns
    -------
    data : dict
        The saved data in the shape of a dictionary.
    """
    ext = _splitext(filename)[1]
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
    elif compression not in ("matlab", "numpy"):
        raise ValueError(f"Unknown file extension {ext}.")

    # address first special cases that do not adhere to the open/decompress scheme
    if compression == "matlab":
        import scipy.io as spio

        data = _check_mat_keys(
            spio.loadmat(filename, struct_as_record=False, squeeze_me=True),
            spio.matlab.mat_struct,
        )
    elif compression == "numpy":
        import numpy as np

        with np.load(filename, allow_pickle=True) as file:
            data = dict(file)

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
