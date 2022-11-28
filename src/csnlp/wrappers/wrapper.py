from typing import Any

from csnlp.nlp import Nlp


class Wrapper:
    '''
    Wraps an NLP to allow a modular transformation of its methods. This class
    is the base class for all wrappers. The subclass could override some
    methods to change the behavior of the original environment without touching
    the original code.
    '''

    def __init__(self, nlp: Nlp) -> None:
        '''Wraps an NLP instance.

        Parameters
        ----------
        nlp : Nlp or subclass
            The NLP to wrap.
        '''
        self.nlp = nlp

    @property
    def unwrapped(self) -> Nlp:
        ''''Returns the original NLP of the wrapper.'''
        return self.nlp.unwrapped

    def __getattr__(self, name) -> Any:
        '''Reroutes attributes to the wrapped NLP instance.'''
        return getattr(self.nlp, name)

    def __str__(self) -> str:
        '''Returns the wrapped NLP string.'''
        return f'<{self.__class__.__name__}: {self.nlp.__str__()}>'

    def __repr__(self) -> str:
        '''Returns the wrapped NLP representation.'''
        return f'<{self.__class__.__name__}: {self.nlp.__repr__()}>'
