from typing import Generic, Literal, TypeVar

import casadi as cs

SymType = TypeVar("SymType", cs.SX, cs.MX)


class HasParameters(Generic[SymType]):
    """Class for creating and storing symbolic parameters of an NLP problem."""

    def __init__(self, sym_type: Literal["SX", "MX"] = "SX") -> None:
        """Instantiate the class.

        Parameters
        ----------
        sym_type : "SX" or "MX", optional
            The CasADi symbolic variable type to use in the NLP, by default "SX".
        """
        super().__init__()
        self._sym_type: type[SymType] = getattr(cs, sym_type)
        self._pars: dict[str, SymType] = {}
        self._p = self._sym_type()

    @property
    def p(self) -> SymType:
        """Gets the parameters of the NLP scheme."""
        return self._p

    @property
    def np(self) -> int:
        """Number of parameters in the NLP scheme."""
        return self._p.shape[0]

    @property
    def parameters(self) -> dict[str, SymType]:
        """Gets the parameters of the NLP scheme."""
        return self._pars

    def parameter(self, name: str, shape: tuple[int, int] = (1, 1)) -> SymType:
        """Adds a parameter to the NLP scheme.

        Parameters
        ----------
        name : str
            Name of the new parameter. Must not be already in use.
        shape : tuple[int, int], optional
            Shape of the new parameter. By default, a scalar.

        Returns
        -------
        casadi.SX or MX
            The symbol for the new parameter.

        Raises
        ------
        ValueError
            Raises if there is already another parameter with the same name.
        """
        if name in self._pars:
            raise ValueError(f"Parameter name '{name}' already exists.")
        par = self._sym_type.sym(name, *shape)
        self._pars[name] = par
        self._p = cs.veccat(self._p, par)
        return par
