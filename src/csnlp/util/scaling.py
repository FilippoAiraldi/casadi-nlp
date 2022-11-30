from collections import UserDict
from typing import Dict, Tuple, TypeVar

import numpy as np

T = TypeVar("T")


class Scaling(UserDict):
    """Class for scaling of quantities. It suffices to register the scale of each the
    variable, and then it can be easily (un-)scaled according to
    ```
            x_scaled = (x - loc) / scale
    ```
    Quantities need not be numerical, and can be of any type that supports basic
    algebraic operations.
    """

    def __init__(self, d: Dict[str, Tuple[T, T]] = None):
        """Initializes the scaling class.

        Parameters
        ----------
        d : Dict[str, Tuple[T, T]], optional
            A possible non-empty dict of variable group names with the corresponding
            scaling range.
        """
        super().__init__()
        if d is not None:
            for k, v in d.items():
                self.register(k, *v)

    def register(self, name: str, loc: T = 0, scale: T = 1) -> None:
        """Registers a new variable's loc and scale for normalization, but raises if
        duplicates occur.

        Parameters
        ----------
        name : str
            Variable group to register for normalization.
        loc : array_like or support algebraic operations, optional
            Mean value of the variable, by default 0.
        scale : array_like or supports algebraic operations, optional
            Standard deviation of the variable, by default 1.

        Raises
        ------
        KeyError
            Raises if a duplicate name is detected.
        ValueError
        """
        if name in self.data:
            raise KeyError(f"'{name}' already registered for normalization.")
        self.data[name] = (loc, scale)

    def can_scale(self, name: str) -> bool:
        """Checks whether the given variable's name was previously registered for
        scaling.

        Parameters
        ----------
        name : str
            Name of the variable to scale.

        Returns
        -------
        bool
            Whether the variable `name` has been registered.
        """
        return name in self.data

    def scale(self, name: str, x: T) -> T:
        """Scales the value `x` according to the ranges of `name`.

        Parameters
        ----------
        name : str
            Variable group `x` belongs to.
        x : array_like or supports algebraic operations
            Value to be scaled.

        Returns
        -------
        array_like or supports algebraic operations
            Normalized `x`.

        Raises
        -------
        AssertionError
            Raises if the scaled output's shape does not match input's shape.
        """
        loc, scale = self.data[name]
        out = (x - loc) / scale
        assert np.shape(out) == np.shape(x), "Scaling altered input shape."
        return out

    def unscale(self, name: str, x: T) -> T:
        """Unscales the value `x` according to the ranges of `name`.

        Parameters
        ----------
        name : str
            Variable group `x` belongs to.
        x : array_like or supports algebraic operations
            Value to be unscaled.

        Returns
        -------
        array_like
            Denormalized `x`.

        Raises
        ------
        AssertionError
            Raises if the unscaled output's shape does not match input's shape.
        """
        loc, scale = self.data[name]
        out = x * scale + loc
        assert np.shape(out) == np.shape(x), "Unscaling altered input shape."
        return out

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {super().__repr__()}>"

    def __str__(self) -> str:
        return f"<{self.__class__.__name__}: {super().__str__()}>"


class MinMaxScaling(Scaling):
    """Class for scaling of quantities. It suffices to register the scale of each the
    variable, and then it can be easily (un-)scaled according to
    ```
            x_scaled = (x - min) / (max - min)
    ```
    Quantities need not be numerical, and can be of any type that supports basic
    algebraic operations.
    """

    def register(self, name: str, min: T = 0, max: T = 1) -> None:
        """Registers a new variable's min and max for normalization, but raises if
        duplicates occur.

        Parameters
        ----------
        name : str
            Variable group to register for normalization.
        min : array_like or support algebraic operations, optional
            Minimum of the variable, by default 0.
        max : array_like or supports algebraic operations, optional
            Maximum of the variable, by default 1.

        Raises
        ------
        KeyError
            Raises if a duplicate name is detected.
        ValueError
        """
        return super().register(name=name, loc=min, scale=max - min)
