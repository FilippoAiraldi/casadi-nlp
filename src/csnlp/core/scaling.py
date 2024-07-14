"""A collection of classes to perform scaling of variables in an :class:`csnlp.Nlp`
instance wrapped with :class:`csnlp.wrappers.NlpScaling`. The classes in this module
inform the wrapper on which variables or parameters to scale and how to scale them."""

from typing import Optional

import numpy as np
import numpy.typing as npt


class Scaler(dict[str, tuple[npt.ArrayLike, npt.ArrayLike]]):
    r"""Class for scaling of NLP variables. It suffices to register the scale of each
    the variable, and then it can be easily (un-)scaled according to

    .. math:: x_{scaled} = \frac{x - loc}{scale}.

    Quantities need not be numerical, and can be of any type that supports basic
    algebraic operations.

    Parameters
    ----------
    d : dict of (str, tuple of 2 array_like), optional
        A possible non-empty dict of variable group names with the corresponding scaling
        parameters ``loc`` and ``scale``.
    """

    def __init__(
        self, d: Optional[dict[str, tuple[npt.ArrayLike, npt.ArrayLike]]] = None
    ) -> None:
        super().__init__()
        if d is not None:
            for k, v in d.items():
                self.register(k, *v)

    def register(
        self, name: str, loc: npt.ArrayLike = 0, scale: npt.ArrayLike = 1
    ) -> None:
        """Registers for the variable ``name`` a new set of ``loc`` and ``scale``
        parameters for its normalization, but raises if duplicates occur.

        Parameters
        ----------
        name : str
            Variable group to register for normalization.
        loc : array_like or supports_algebraic_operations, optional
            Mean value of the variable, by default ``0``.
        scale : array_like or supports_algebraic_operations, optional
            Standard deviation of the variable, by default ``1``.

        Raises
        ------
        KeyError
            Raises if ``name`` is duplicated.
        """
        if name in self:
            raise KeyError(f"'{name}' already registered for normalization.")
        self[name] = (loc, scale)

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
            Whether the variable ``name`` has been registered.
        """
        return name in self

    def scale(self, name: str, x: npt.ArrayLike) -> npt.ArrayLike:
        """Scales the value ``x`` according to the ranges of variable ``name``.

        Parameters
        ----------
        name : str
            Variable group ``x`` belongs to.
        x : array_like or supports_algebraic_operations
            Value to be scaled.

        Returns
        -------
        array_like or supports_algebraic_operations
            Normalized ``x``.

        Raises
        ------
        AssertionError
            Raises if the scaled output's shape does not match input's shape.
        """
        loc, scale = self[name]
        out = (x - loc) / scale  # type: ignore[operator]
        assert np.shape(out) == np.shape(x), "Scaling altered input shape."
        return out

    def unscale(self, name: str, x: npt.ArrayLike) -> npt.ArrayLike:
        """Unscales the value ``x`` according to the ranges of ``name``.

        Parameters
        ----------
        name : str
            Variable group ``x`` belongs to.
        x : array_like or supports_algebraic_operations
            Value to be unscaled.

        Returns
        -------
        array_like
            Denormalized ``x``.

        Raises
        ------
        AssertionError
            Raises if the unscaled output's shape does not match input's shape.
        """
        loc, scale = self[name]
        out = x * scale + loc  # type: ignore[operator]
        assert np.shape(out) == np.shape(x), "Unscaling altered input shape."
        return out

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {super().__repr__()}>"

    def __str__(self) -> str:
        return f"<{self.__class__.__name__}: {super().__str__()}>"


class MinMaxScaler(Scaler):
    r"""Class for scaling of quantities. It suffices to register the scale of each the
    variable, and then it can be easily (un-)scaled according to

    .. math:: x_{scaled} = \frac{x - min}{max - min}.

    Quantities need not be numerical, and can be of any type that supports basic
    algebraic operations.
    """

    def register(
        self, name: str, min: npt.ArrayLike = 0, max: npt.ArrayLike = 1
    ) -> None:
        """Registers for the variable ``name`` a new set of ``min`` and ``max``
        parameters for its normalization, but raises if duplicates occur.

        Parameters
        ----------
        name : str
            Variable group to register for normalization.
        min : array_like or supports_algebraic_operations, optional
            Minimum of the variable, by default ``0``.
        max : array_like or supports_algebraic_operations, optional
            Maximum of the variable, by default ``1``.

        Raises
        ------
        KeyError
            Raises if ``name`` is duplicated.
        """
        return super().register(name, min, max - min)
