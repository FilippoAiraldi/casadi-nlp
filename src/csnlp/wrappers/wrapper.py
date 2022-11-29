from typing import Any, Type

from csnlp.nlp import Nlp


class Wrapper:
    """
    Wraps an NLP to allow a modular transformation of its methods. This class
    is the base class for all wrappers. The subclass could override some
    methods to change the behavior of the original environment without touching
    the original code.
    """

    def __init__(self, nlp: Nlp) -> None:
        """Wraps an NLP instance.

        Parameters
        ----------
        nlp : Nlp or subclass
            The NLP to wrap.
        """
        self.nlp = nlp

    @property
    def unwrapped(self) -> Nlp:
        """'Returns the original NLP of the wrapper."""
        return self.nlp.unwrapped

    def is_wrapped(self, wrapper_type: Type["Wrapper"]) -> bool:
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

    def __getattr__(self, name) -> Any:
        """Reroutes attributes to the wrapped NLP instance."""
        return getattr(self.nlp, name)

    def __str__(self) -> str:
        """Returns the wrapped NLP string."""
        return f"<{self.__class__.__name__}: {self.nlp.__str__()}>"

    def __repr__(self) -> str:
        """Returns the wrapped NLP representation."""
        return f"<{self.__class__.__name__}: {self.nlp.__repr__()}>"
