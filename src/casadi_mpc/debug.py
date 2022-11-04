from dataclasses import dataclass
from inspect import getframeinfo, stack
from typing import List, Tuple
from types import MappingProxyType


@dataclass(frozen=True)
class MpcDebugEntry:
    '''Class representing a single entry of debug information for the MPC.'''

    name: str
    type: str
    shape: Tuple[int, int]
    filename: str
    function: str
    lineno: int
    context: str

    def __str__(self) -> str:
        shape = 'x'.join(str(d) for d in self.shape)
        return \
            f'{self.type} \'{self.name}\' of shape {shape} defined at\n' \
            f'  filename: {self.filename}\n' \
            f'  function: {self.function}:{self.lineno}\n' \
            f'  context:  {self.context}\n'

    def __repr__(self) -> str:
        return f'{self.__class__.__class__}({self.__str__()})'


class MpcDebug:
    '''
    MPC debug class for information about variables and constraints. In 
    particular, it records information on
     - the decision variable `x`
     - the equality constraints `g`
     - the inequality constraints `h`
    '''

    __slots__ = ['_x_info', '_g_info', '_h_info']

    types = MappingProxyType({
        'x': 'Decision variable',
        'g': 'Equality constraint',
        'h': 'Inequality constraint'
    })

    def __init__(self) -> None:
        '''Initializes the debug information collector.'''
        self._x_info: List[Tuple[range, MpcDebugEntry]] = []
        self._g_info: List[Tuple[range, MpcDebugEntry]] = []
        self._h_info: List[Tuple[range, MpcDebugEntry]] = []

    def x_describe(self, index: int) -> MpcDebugEntry:
        '''Returns debug information on the variable x at the given index.

        Parameters
        ----------
        index : int
            Index of the variable x to query information about.

        Returns
        -------
        MpcDebugEntry
            A class instance containing debug information on the variable x
            at the given index.

        Raises
        ------
        IndexError
            Index not found, or outside bounds of x.
        '''
        return self.__describe(self._x_info, index)

    def g_describe(self, index: int) -> MpcDebugEntry:
        '''Returns debug information on the constraint g at the given index.

        Parameters
        ----------
        index : int
            Index of the constraint g to query information about.

        Returns
        -------
        MpcDebugEntry
            A class instance containing debug information on the constraint g
            at the given index.

        Raises
        ------
        IndexError
            Index not found, or outside bounds of g.
        '''
        return self.__describe(self._g_info, index)

    def h_describe(self, index: int) -> MpcDebugEntry:
        '''Returns debug information on the constraint h at the given index.

        Parameters
        ----------
        index : int
            Index of the constraint h to query information about.

        Returns
        -------
        MpcDebugEntry
            A class instance containing debug information on the constraint h
            at the given index.

        Raises
        ------
        IndexError
            Index not found, or outside bounds of h.
        '''
        return self.__describe(self._h_info, index)

    def register(self, group: str, name: str, shape: Tuple[int, int]) -> None:
        '''Registers debug information on new object name under the specific 
        group.

        Parameters
        ----------
        group : {'x', 'g', 'h'}
            Whether the object belongs to variables, equality or inequality 
            constraints.
        name : str
            Name of the object.
        shape : Tuple[int, int]
            Shape of the object.

        Raises
        ------
        AttributeError
            Raises in case the given group is invalid.
        '''
        trackback = getframeinfo(stack()[2][0])
        info: List[Tuple[range, MpcDebugEntry]] = getattr(self,
                                                          f'_{group}_info')
        last = info[-1][0].stop if info else 0
        info.append((
            range(last, last + shape[0] * shape[1]),
            MpcDebugEntry(
                name=name,
                type=self.types[group],
                shape=shape,
                filename=trackback.filename,
                function=trackback.function,
                lineno=trackback.lineno,
                context=('; '.join(trackback.code_context).strip()
                         if trackback.code_context is not None else
                         '')
            )))

    def __describe(
        self,
        info: List[Tuple[range, MpcDebugEntry]], index: int
    ) -> MpcDebugEntry:
        for range_, description in info:
            if index in range_:
                return description
        raise IndexError(f'Index {index} not found.')
