from functools import cached_property
from typing import Literal, TypeVar

import casadi as cs
import numpy as np
from numpy.typing import NDArray

from ..core.cache import invalidate_cache
from .parameters import HasParameters

SymType = TypeVar("SymType", cs.SX, cs.MX)


class HasVariables(HasParameters[SymType]):
    """Class for the creation and storage of symbolic variables in an NLP problem. It
    builds on top of :class:`HasParameters`, which handles parameters.

    Parameters
    ----------
    sym_type : {"SX", "MX"}, optional
        The CasADi symbolic variable type to use in the NLP, by default ``"SX"``.
    """

    def __init__(self, sym_type: Literal["SX", "MX"] = "SX") -> None:
        super().__init__(sym_type)
        self._vars: dict[str, SymType] = {}
        self._x = self._sym_type()
        self._has_discrete = False
        self._discrete: dict[str, bool] = {}

    @property
    def x(self) -> SymType:
        """Gets the primary variables of the NLP scheme in vector form."""
        return self._x

    @property
    def nx(self) -> int:
        """Number of variables in the NLP scheme."""
        return self._x.shape[0]

    @property
    def variables(self) -> dict[str, SymType]:
        """Gets the primal variables of the NLP scheme."""
        return self._vars

    @property
    def has_discrete(self) -> bool:
        """Flags if the NLP has discrete variables."""
        return self._has_discrete

    @cached_property
    def discrete(self) -> NDArray[np.bool_]:
        """Gets the boolean array indicating which variables are discrete."""
        if not self._has_discrete:
            return np.zeros(self.nx, dtype=bool)
        vars = self._vars
        return np.concatenate(
            [
                (np.ones if is_discrete else np.zeros)(
                    np.prod(vars[name].numel()), dtype=bool
                )
                for name, is_discrete in self._discrete.items()
            ]
        )

    @invalidate_cache(discrete)
    def variable(
        self,
        name: str,
        shape: tuple[int, int] = (1, 1),
        discrete: bool = False,
    ) -> SymType:
        """Adds a variable to the NLP problem.

        Parameters
        ----------
        name : str
            Name of the new variable. Must not be already in use.
        shape : tuple[int, int], optional
            Shape of the new parameter. By default a scalar, i.e., ``(1, 1)``.
        discrete : bool, optional
            Flag indicating if the variable is discrete. Defaults to ``False``.

        Returns
        -------
        var : casadi.SX
            The symbol of the new variable.

        Raises
        ------
        ValueError
            Raises if there is already another variable with the same name ``name``.
        """
        if name in self._vars:
            raise ValueError(f"Variable name '{name}' already exists.")
        var = self._sym_type.sym(name, *shape)
        self._vars[name] = var
        self._x = cs.veccat(self._x, var)
        self._has_discrete |= discrete
        self._discrete[name] = discrete
        return var
