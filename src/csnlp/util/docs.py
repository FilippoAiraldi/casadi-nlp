"""Functions to extract information from CasADi documentation.

Taken from
https://bitbucket.org/rawlings-group/mpc-tools-casadi/src/master/mpctools/util.py.
"""

from collections import namedtuple
from contextlib import suppress
from itertools import dropwhile
from typing import Any, Callable, Dict, List, Tuple
from warnings import warn

import casadi as cs


class _LambdaType:
    def __init__(self, func, typerepr):
        self.__typerepr = typerepr
        self.__func = func

    def __call__(self, val):
        return self.__func(val)

    def __repr__(self):
        return f"<type '{self.__typerepr}'>"

    def __str__(self):
        return repr(self)


_DocCell = namedtuple("_DocCell", ["id", "default", "doc"])
_TABLE_START = "+="  # string prefix that starts the table
_CELL_END = "+-"  # string prefix that ends the cell
_CELL_CONTENTS = "|"  # string prefix that continues the cell
_JOINS = ("", "", "", " ")  # how to join multiple lines in a given cell
_TYPES: Dict[str, Callable] = {  # casadi types to python types
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


def _get_doc_cell(lines) -> _DocCell:
    """
    Returns a DocCell tuple for the set of lines.

    joins is a tuple of strings to say how to join multiple lines in a given
    cell. It must have exactly one entry for each cell
    """
    ncol = 4
    fields: Tuple[List[str], ...] = ([], [], [], [])
    for line in lines:
        cells = line.split(" | ", ncol - 1)
        cells[0] = cells[0].lstrip().lstrip("|")
        cells[-1] = cells[-1].rstrip().rstrip("|")
        if len(cells) != ncol:
            raise ValueError("Wrong number of columns.")
        for i, c in enumerate(cells):
            fields[i].append(c.strip())

    id, type, default, doc = (j.join(f) for j, f in zip(_JOINS, fields))

    if typefunc := _TYPES.get(type):
        try:
            default = typefunc(default)
        except (ValueError, TypeError):
            includetype = True
            if default in ("None", "GenericType()"):
                default = None  # type: ignore[assignment]
            elif typefunc is int:
                with suppress(ValueError, TypeError):
                    default = int(float(default))  # type: ignore[assignment]
                    includetype = False
            if includetype:
                default = (default, typefunc)  # type: ignore[assignment]
    else:
        warn(f"Unknown type for '{id}', '{type}'.")
    return _DocCell(id, default, doc)


def _get_doc_dict(docstring: str) -> Dict[str, Tuple[Any, str]]:
    lineiter = dropwhile(
        lambda x: not x.startswith(_TABLE_START), docstring.split("\n")
    )
    try:
        next(lineiter)
    except StopIteration as e:
        raise ValueError("No table found!") from e
    thiscell: List[str] = []
    allcells: List[_DocCell] = []
    for line in lineiter:
        if line.startswith(_CELL_END):
            allcells.append(_get_doc_cell(thiscell))
            thiscell.clear()
        elif line.startswith(_CELL_CONTENTS):
            thiscell.append(line)
        else:
            break
    return {c.id: (c.default, c.doc) for c in allcells}


def get_casadi_plugins() -> Dict[str, List[str]]:
    """Returns a dictionary of available casadi plugin as a dict of (type, name)."""

    func = getattr(cs, "CasadiMeta_getPlugins", getattr(cs, "CasadiMeta_plugins", None))
    if func is None:
        raise RuntimeError("Unable to get Casadi plugins.")

    all_plugins: str = func()
    plugins = (p.split("::") for p in all_plugins.split(";"))
    plugin_dict: Dict[str, List[str]] = {}
    for group, name in plugins:
        if group not in plugin_dict:
            plugin_dict[group] = []
        plugin_dict[group].append(name)
    return plugin_dict


def list_available_solvers() -> Dict[str, List[str]]:
    """Returns available solvers as a string or a dictionary."""
    availablesolvers = get_casadi_plugins()
    return {
        "nlp": availablesolvers.get("Nlpsol", []),
        "qp": availablesolvers.get("Conic", []) + availablesolvers.get("Qpsol", []),
    }


def get_solver_options(solver, display: bool = True) -> Dict[str, Tuple[Any, str]]:
    """Returns a dictionary of solver-specific options, with default value and
    description."""
    availablesolvers = list_available_solvers()
    if solver in availablesolvers["nlp"]:
        docstring = cs.doc_nlpsol(solver)
    elif solver in availablesolvers["qp"]:
        docstring = cs.doc_conic(solver)
    else:
        raise ValueError(f"Unknown solver: '{solver}'.")
    options: Dict[str, Tuple[Any, str]] = _get_doc_dict(docstring)
    if display:
        print("Available options [default] for %s:\n" % solver)
        for k in sorted(options.keys()):
            print(k, "[%r]: %s\n" % options[k])
    return options
