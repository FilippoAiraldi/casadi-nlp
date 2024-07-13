from typing import Literal, TypeVar

import casadi as cs

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

    def variable(self, name: str, shape: tuple[int, int] = (1, 1)) -> SymType:
        """Adds a variable to the NLP problem.

        Parameters
        ----------
        name : str
            Name of the new variable. Must not be already in use.
        shape : tuple[int, int], optional
            Shape of the new parameter. By default a scalar, i.e., ``(1, 1)``.

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
        return var
