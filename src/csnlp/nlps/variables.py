from typing import Dict, Generic, Literal, Tuple, Type, TypeVar

import casadi as cs

T = TypeVar("T", cs.SX, cs.MX)


class HasVariables(Generic[T]):
    """Class for creating and storing symbolic variables of an NLP problem."""

    def __init__(self, sym_type: Literal["SX", "MX"] = "SX") -> None:
        """Instantiate the class.

        Parameters
        ----------
        sym_type : "SX" or "MX", optional
            The CasADi symbolic variable type to use in the NLP, by default "SX".
        """
        super().__init__()
        self._csXX: Type[T] = getattr(cs, sym_type)
        self._vars: Dict[str, T] = {}
        self._x = self._csXX()

    @property
    def x(self) -> T:
        """Gets the primary variables of the NLP scheme in vector form."""
        return self._x

    @property
    def nx(self) -> int:
        """Number of variables in the NLP scheme."""
        return self._x.shape[0]

    @property
    def variables(self) -> Dict[str, T]:
        """Gets the primal variables of the NLP scheme."""
        return self._vars

    def variable(self, name: str, shape: Tuple[int, int] = (1, 1)) -> T:
        """Adds a variable to the NLP problem.

        Parameters
        ----------
        name : str
            Name of the new variable. Must not be already in use.
        shape : tuple[int, int], optional
            Shape of the new variable. By default, a scalar.

        Returns
        -------
        var : casadi.SX
            The symbol of the new variable.

        Raises
        ------
        ValueError
            Raises if there is already another variable with the same name.
        """
        if name in self._vars:
            raise ValueError(f"Variable name '{name}' already exists.")
        var = self._csXX.sym(name, *shape)
        self._vars[name] = var
        self._x = cs.vertcat(self._x, cs.vec(var))
        return var
