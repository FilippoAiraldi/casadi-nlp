from collections.abc import Iterable
from typing import Any, Generic, TypeVar, Union

import casadi as cs
from numpy import typing as npt

from ..core.solutions import Solution
from ..nlps.nlp import Nlp

SymType = TypeVar("SymType", cs.SX, cs.MX)


class Wrapper(Generic[SymType]):
    """Wraps an instance of :class:`csnlp.Nlp` to allow a modular transformation of its
    methods. This class is the base class for all wrappers. The subclass can then
    override some methods to change the behavior of the original environment without
    touching the original code.

    The base class is retroactive, in the sense that it can be applied to any NLP
    instance that already defines variables, parameters, and/or objective. Use
    :class:`NonRetroactiveWrapper` for wrappers that need to wrap an NLP before it is
    defined.

    Parameters
    ----------
    nlp : Nlp or subclass
        The NLP to wrap.
    """

    def __init__(self, nlp: Nlp[SymType]) -> None:
        super().__init__()
        self.nlp = nlp

    @property
    def unwrapped(self) -> Nlp[SymType]:
        """'Returns the original NLP of the wrapper."""
        return self.nlp.unwrapped

    def is_wrapped(self, wrapper_type: type["Wrapper[SymType]"]) -> bool:
        """Gets whether the NLP instance is wrapped or not by the given wrapper type.

        Parameters
        ----------
        wrapper_type : type of Wrapper
            Type of wrapper to check if the NLP is wrapped with.

        Returns
        -------
        bool
            ``True`` if wrapped by an instance of ``wrapper_type``; ``False``,
            otherwise.
        """
        if isinstance(self, wrapper_type):
            return True
        return self.nlp.is_wrapped(wrapper_type)

    def __getattr__(self, name: str) -> Any:
        """Reroutes attributes to the wrapped NLP instance."""
        if name.startswith("_"):
            raise AttributeError(f"Accessing private attribute '{name}' is prohibited.")
        return getattr(self.nlp, name)

    def __call__(
        self,
        pars: Union[
            None, dict[str, npt.ArrayLike], Iterable[dict[str, npt.ArrayLike]]
        ] = None,
        vals0: Union[
            None, dict[str, npt.ArrayLike], Iterable[dict[str, npt.ArrayLike]]
        ] = None,
        **kwargs: Any,
    ) -> Union[Solution[SymType], list[Solution[SymType]]]:
        # Similar logic to `MultiStartNlp.__call__`: call solve_multi only if either
        # pars or vals0 is an iterable; otherwise, run the single, base NLP
        if not self.nlp.is_multi or (
            (pars is None or isinstance(pars, dict))
            and (vals0 is None or isinstance(vals0, dict))
        ):
            return self.solve(pars, vals0)
        return self.solve_multi(pars, vals0, **kwargs)

    def __str__(self) -> str:
        """Returns the wrapped NLP string."""
        return f"<{self.__class__.__name__}: {self.nlp.__str__()}>"

    def __repr__(self) -> str:
        """Returns the wrapped NLP representation."""
        return f"<{self.__class__.__name__}: {self.nlp.__repr__()}>"


class NonRetroactiveWrapper(Wrapper[SymType], Generic[SymType]):
    """Same as :class:`Wrapper`, but the wrapped NLP instance must have no variable,
    parameter or objective specified; in other words, the wrapper must wrap the NLP
    before it gets defined.

    Parameters
    ----------
    nlp : Nlp
        The NLP instance to be wrapped.

    Raises
    ------
    ValueError
        Raises if the objective, variables, dual variables, parameters or constraints
        are already defined in this NLP instance.
    """

    def __init__(self, nlp: Nlp[SymType]) -> None:
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
