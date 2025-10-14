"""A collection of utilities for input/output operations. The goals of this module are:

- compatibility of pickling/deepcopying with CasADi objects and classes that hold such
  objects (since these are often not picklable).
- saving and loading data to/from files, possibly compressed.
"""

import pickle
from functools import partial as _partial
from os.path import splitext as _splitext
from pickletools import optimize as _optimize
from typing import TYPE_CHECKING, Callable, Literal, Optional
from typing import Any as _Any

if TYPE_CHECKING:
    from scipy.io.matlab import mat_struct


_COMPRESSION_EXTS: dict[
    str, Optional[Literal["lzma", "bz2", "gzip", "brotli", "blosc2", "matlab", "numpy"]]
] = {
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
        compress_fun = lambda o: o
    elif compression == "lzma":
        import lzma

        expected_ext = ".xz"
        open_fun = lzma.open
        compress_fun = lambda o: o
    elif compression == "bz2":
        import bz2

        expected_ext = ".pbz2"
        open_fun = bz2.BZ2File
        compress_fun = lambda o: o
    elif compression == "gzip":
        import gzip

        expected_ext = ".gz"
        open_fun = gzip.open
        compress_fun = lambda o: o
    elif compression == "brotli":
        import brotli

        expected_ext = ".bt"
        open_fun = open
        compress_fun = brotli.compress
    elif compression == "blosc2":
        import blosc2

        expected_ext = ".bl2"
        open_fun = open
        compress_fun = _partial(blosc2.compress, typesize=None)
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
        decompress_fun = lambda o: pickle.loads(brotli.decompress(o))
    elif compression == "blosc2":
        import blosc2

        open_fun = open
        decompress_fun = lambda o: pickle.loads(blosc2.decompress(o))
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
    for key, value in dictionary.items():
        if isinstance(value, mat_struct_type):
            dictionary[key] = _todict_recursive(value)
    return dictionary
