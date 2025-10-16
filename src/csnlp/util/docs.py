"""A collection of stand-alone functions to extract information from the CasADi
documentation via code. In particular, this module offers a way to get the solvers that
are available in CasADi (i.e., they have an interface) as well as and their options. The
functions are taken from the
`MPCTools <https://bitbucket.org/rawlings-group/mpc-tools-casadi/src/master/mpctools/util.py>`_
repository by the Rawlings' group.
"""

import contextlib
import itertools
import warnings
from typing import Any as _Any
from typing import Callable
from typing import NamedTuple as _NamedTuple

import casadi as cs


class _LambdaType:
    def __init__(self, func: Callable[[_Any], _Any], typerepr: str) -> None:
        self.__typerepr = typerepr
        self.__func = func

    def __call__(self, val: _Any) -> _Any:
        return self.__func(val)

    def __repr__(self) -> str:
        return f"<type '{self.__typerepr}'>"

    def __str__(self) -> str:
        return repr(self)


class _DocCell(_NamedTuple):
    id: str
    default: str
    doc: str


_TABLE_START = "+="  # string prefix that starts the table
_CELL_END = "+-"  # string prefix that ends the cell
_CELL_CONTENTS = "|"  # string prefix that continues the cell
_TYPES: dict[str, Callable] = {  # casadi types to python types
    "OT_INTEGER": int,
    "OT_STRING": str,
    "OT_REAL": float,
    "OT_INT": int,
    "OT_DICT": dict,
    "OT_DOUBLE": float,
    "OT_BOOL": bool,
    "OT_STR": str,
    "OT_INTVECTOR": _LambdaType(lambda x: [int(i) for i in x], "list[int]"),
    "OT_STRINGVECTOR": _LambdaType(lambda x: [str(i) for i in x], "list[str]"),
}


def _get_doc_cell(lines: list[str]) -> _DocCell:
    """Returns a DocCell tuple for the set of lines.

    joins is a tuple of strings to say how to join multiple lines in a given
    cell. It must have exactly one entry for each cell.
    """
    ncol = lines[0].count(" | ") + 1
    assert ncol in (3, 4), f"Expected 3 or 4 columns in the docstring table; got {ncol}"

    fields: tuple[list[str], ...] = tuple([] for _ in range(ncol))
    for line in lines:
        cells = [c.strip() for c in line.split(" | ") if c]
        cells[0] = cells[0].lstrip("|").rstrip()
        cells[-1] = cells[-1].rstrip("|").rstrip()
        for i, c in enumerate(cells):
            fields[i].append(c.strip())
            if i == ncol - 1:
                break
        else:
            raise ValueError("Wrong number of columns.")

    if ncol == 3:
        id, type, doc = (j.join(f) for j, f in zip(("", "", " "), fields))
        default = None

    else:
        id, type, default, doc = (j.join(f) for j, f in zip(("", "", "", " "), fields))
        if typefunc := _TYPES.get(type):
            try:
                default = typefunc(default)
            except (ValueError, TypeError):
                includetype = True
                if default in ("None", "GenericType()"):
                    default = None
                elif typefunc is int:
                    with contextlib.suppress(ValueError, TypeError):
                        default = int(float(default))
                        includetype = False
                if includetype:
                    default = (default, typefunc)
        else:
            warnings.warn(f"Unknown type for '{id}', '{type}'.", stacklevel=2)

    return _DocCell(id, default, doc)


def _get_doc_dict(docstring: str) -> dict[str, tuple[_Any, str]]:
    lineiter = itertools.dropwhile(
        lambda x: not x.startswith(_TABLE_START), docstring.split("\n")
    )
    try:
        next(lineiter)
    except StopIteration as e:
        raise ValueError("No table found!") from e
    thiscell: list[str] = []
    allcells: list[_DocCell] = []
    for line in lineiter:
        if line.startswith(_CELL_END):
            allcells.append(_get_doc_cell(thiscell))
            thiscell.clear()
        elif line.startswith(_CELL_CONTENTS):
            thiscell.append(line)
        else:
            break
    return {c.id: ("N/A" if c.default is None else c.default, c.doc) for c in allcells}


def get_casadi_plugins() -> dict[str, list[str]]:
    """Returns the available CasADi plugins.

    Returns
    -------
    dict of (str, list of str)
        A dictionary containing for each type of problem a list of plugin names that
        are available.

    Raises
    ------
    RuntimeError
        Raises in case the plugins cannot be retrieved because the functions
        :func:`cs.CasadiMeta_getPlugins` or :func:`cs.CasadiMeta_plugins` are not
        available.
    """

    func = getattr(cs, "CasadiMeta_getPlugins", getattr(cs, "CasadiMeta_plugins", None))
    if func is None:
        raise RuntimeError("Unable to get Casadi plugins.")

    all_plugins: str = func()
    plugins = (p.split("::") for p in all_plugins.split(";"))
    plugin_dict: dict[str, list[str]] = {}
    for group, name in plugins:
        if group not in plugin_dict:
            plugin_dict[group] = []
        plugin_dict[group].append(name)
    return plugin_dict


def list_available_solvers() -> dict[str, list[str]]:
    """Returns the available CasADi solvers.

    Returns
    -------
    dict of (str, list of str)
        A dictionary containing for each type of problem a list of plugin names that
        are available.

    Raises
    ------
    RuntimeError
        Raises in case the plugins cannot be retrieved because the functions
        :func:`cs.CasadiMeta_getPlugins` or :func:`cs.CasadiMeta_plugins` are not
        available.
    """
    availablesolvers = get_casadi_plugins()
    return {
        "nlp": availablesolvers.get("Nlpsol", []),
        "qp": availablesolvers.get("Conic", []) + availablesolvers.get("Qpsol", []),
    }


def get_solver_options(
    solver: str, display: bool = True
) -> dict[str, tuple[_Any, str]]:
    """Returns the solver-specific options, with default value and description whenever
    available.

    Parameters
    ----------
    solver : str
        The solver name.
    display : bool, optional
        Whether to print the options, by default ``True``

    Returns
    -------
    dict of (str, tuple of (Any, str)
        A dictionary containing for each option the default value and a description.

    Raises
    ------
    ValueError
        Raises in case ``solver`` is not included in the available solvers.
    """
    availablesolvers = list_available_solvers()
    if solver in availablesolvers["nlp"]:
        docstring = cs.doc_nlpsol(solver)
    elif solver in availablesolvers["qp"]:
        docstring = cs.doc_conic(solver)
    else:
        raise ValueError(f"Unknown solver: '{solver}'.")
    options = _get_doc_dict(docstring)
    if display:
        print(f"Available options [default] for {solver}:\n")
        for k in sorted(options.keys()):
            default, doc = options[k]
            print(f"{k} [{default!r}]: {doc}\n")
    return options
