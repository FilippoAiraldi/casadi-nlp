from contextlib import contextmanager
from typing import Any, Generic, Iterator, List, Type, TypeVar, Union

import casadi as cs

from csnlp.core.solutions import Solution
from csnlp.nlps.nlp import Nlp
from csnlp.util.io import SupportsDeepcopyAndPickle

SymType = TypeVar("SymType", cs.SX, cs.MX)


class Wrapper(SupportsDeepcopyAndPickle, Generic[SymType]):
    """Wraps an NLP to allow a modular transformation of its methods. This class is the
    base class for all wrappers. The subclass could override some methods to change the
    behavior of the original environment without touching the original code.

    The base class is retroactive, in the sense that it can be applied to any NLP
    instance that already defines variables, parameters, and/or objective. Use
    `NonRetroactiveWrapper` for wrappers that need to wrap an NLP before it is defined.
    """

    __slots__ = ("nlp",)

    def __init__(self, nlp: Nlp[SymType]) -> None:
        """Wraps an NLP instance.

        Parameters
        ----------
        nlp : Nlp or subclass
            The NLP to wrap.
        """
        super().__init__()
        self.nlp = nlp

    @property
    def unwrapped(self) -> Nlp[SymType]:
        """'Returns the original NLP of the wrapper."""
        return self.nlp.unwrapped

    def is_wrapped(self, wrapper_type: Type["Wrapper[SymType]"]) -> bool:
        """Gets whether the NLP instance is wrapped or not by the given wrapper type.

        Parameters
        ----------
        wrapper_type : type of Wrapper
            Type of wrapper to check if the NLP is wrapped with.

        Returns
        -------
        bool
            `True` if wrapped by an instance of `wrapper_type`; `False`, otherwise.
        """
        if isinstance(self, wrapper_type):
            return True
        return self.nlp.is_wrapped(wrapper_type)

    @contextmanager
    def fullstate(self) -> Iterator[None]:
        with super().fullstate(), self.nlp.fullstate():
            yield

    @contextmanager
    def pickleable(self) -> Iterator[None]:
        with super().pickleable(), self.nlp.pickleable():
            yield

    def __getattr__(self, name: str) -> Any:
        """Reroutes attributes to the wrapped NLP instance."""
        if name.startswith("_"):
            raise AttributeError(f"Accessing private attribute '{name}' is prohibited.")
        return getattr(self.nlp, name)

    def __call__(
        self, *args: Any, **kwds: Any
    ) -> Union[Solution[SymType], List[Solution[SymType]]]:
        return (self.solve_multi if self.nlp.is_multi else self.solve)(*args, **kwds)

    def __str__(self) -> str:
        """Returns the wrapped NLP string."""
        return f"<{self.__class__.__name__}: {self.nlp.__str__()}>"

    def __repr__(self) -> str:
        """Returns the wrapped NLP representation."""
        return f"<{self.__class__.__name__}: {self.nlp.__repr__()}>"


class NonRetroactiveWrapper(Wrapper[SymType], Generic[SymType]):
    """Same as `Wrapper`, but the wrapped NLP instance must have no variable, parameter
    or objective specified; in other words, the wrapper must wrap the NLP before it gets
    defined."""

    def __init__(self, nlp: Nlp[SymType]) -> None:
        """Initializes the non-retroactive wrapper around an NLP.

        Parameters
        ----------
        nlp : Nlp
            The NLP instance to be wrapped.

        Raises
        ------
        ValueError
            Raises if the objective, variables, dual variables, parameters or
            constraints are already defined in this NLP instance.
        """
        super().__init__(nlp)
        unlp = nlp.unwrapped
        if (
            unlp._f is not None
            or unlp._vars
            or unlp._dual_vars
            or unlp._pars
            or unlp._cons
        ):
            raise ValueError("Nlp already defined.")
