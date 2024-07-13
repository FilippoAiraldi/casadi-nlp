from inspect import getframeinfo
from itertools import dropwhile
from traceback import walk_stack
from types import MappingProxyType
from typing import Literal, NamedTuple

from numpy import prod


class NlpDebugEntry(NamedTuple):
    """Class representing a single entry of the debug information for an
    :class:`csnlp.Nlp` instance."""

    name: str
    type: str
    shape: tuple[int, ...]
    filename: str
    function: str
    lineno: int
    context: str

    def __str__(self) -> str:
        shape = "x".join(str(d) for d in self.shape)
        return (
            f"{self.type} '{self.name}' of shape {shape} defined at\n"
            f"  filename: {self.filename}\n"
            f"  function: {self.function}:{self.lineno}\n"
            f"  context:  {self.context}"
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.__str__()})"


class NlpDebug:
    """NLP debug class for information about variables and constraints in an instance
    of the :class:`csnlp.Nlp` class. In particular, it records information on

    - the parameters ``p``
    - the decision variable ``x``
    - the equality constraints ``g``
    - the inequality constraints ``h``.
    """

    types = MappingProxyType(
        {
            "p": "Parameter",
            "x": "Decision variable",
            "g": "Equality constraint",
            "h": "Inequality constraint",
        }
    )

    def __init__(self) -> None:
        self._p_info: list[tuple[range, NlpDebugEntry]] = []
        self._x_info: list[tuple[range, NlpDebugEntry]] = []
        self._g_info: list[tuple[range, NlpDebugEntry]] = []
        self._h_info: list[tuple[range, NlpDebugEntry]] = []

    def p_describe(self, index: int) -> NlpDebugEntry:
        """Returns debug information on the parameter at the given ``index``.

        Parameters
        ----------
        index : int
            Index of the parameter p to query information about.

        Returns
        -------
        NlpDebugEntry
            A class instance containing debug information on the parameter p
            at the given index.

        Raises
        ------
        IndexError
            Index not found, or outside bounds of p.
        """
        return self.__describe(self._p_info, index)

    def x_describe(self, index: int) -> NlpDebugEntry:
        """Returns debug information on the variable at the given ``index``.

        Parameters
        ----------
        index : int
            Index of the variable x to query information about.

        Returns
        -------
        NlpDebugEntry
            A class instance containing debug information on the variable x
            at the given index.

        Raises
        ------
        IndexError
            Index not found, or outside bounds of x.
        """
        return self.__describe(self._x_info, index)

    def g_describe(self, index: int) -> NlpDebugEntry:
        """Returns debug information on the constraint at the given ``index``.

        Parameters
        ----------
        index : int
            Index of the constraint g to query information about.

        Returns
        -------
        NlpDebugEntry
            A class instance containing debug information on the constraint g
            at the given index.

        Raises
        ------
        IndexError
            Index not found, or outside bounds of g.
        """
        return self.__describe(self._g_info, index)

    def h_describe(self, index: int) -> NlpDebugEntry:
        """Returns debug information on the constraint at the given ``index``.

        Parameters
        ----------
        index : int
            Index of the constraint h to query information about.

        Returns
        -------
        NlpDebugEntry
            A class instance containing debug information on the constraint h
            at the given index.

        Raises
        ------
        IndexError
            Index not found, or outside bounds of h.
        """
        return self.__describe(self._h_info, index)

    def register(
        self, group: Literal["p", "x", "g", "h"], name: str, shape: tuple[int, ...]
    ) -> None:
        """Registers debug information on new object name under the specific
        group.

        Parameters
        ----------
        group : {"p", "x", "g", "h"}
            Indentifies the group the object belongs to: parameters, variables,
            equality constraints or inequality constraints.
        name : str
            Name of the object.
        shape : Tuple[int, ...]
            Shape of the object.

        Raises
        ------
        AttributeError
            Raises in case the given group is invalid.
        """
        stack = dropwhile(
            lambda f: f[0].f_globals["__name__"].startswith("csnlp."), walk_stack(None)
        )
        frame, lineno = next(stack)
        traceback = getframeinfo(frame, context=3)
        info: list[tuple[range, NlpDebugEntry]] = getattr(self, f"_{group}_info")
        last = info[-1][0].stop if info else 0
        info.append(
            (
                range(last, last + prod(shape)),
                NlpDebugEntry(
                    name,
                    self.types[group],
                    shape,
                    traceback.filename,
                    traceback.function,
                    lineno,
                    (
                        "".join(traceback.code_context)
                        if traceback.code_context is not None
                        else ""
                    ),
                ),
            )
        )

    def __describe(
        self, info: list[tuple[range, NlpDebugEntry]], index: int
    ) -> NlpDebugEntry:
        for range_, description in info:
            if index in range_:
                return description
        raise IndexError(f"Index {index} not found.")
