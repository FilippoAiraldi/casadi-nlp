from typing import Any, Dict, Generator, NamedTuple, Optional

import numpy as np
import numpy.typing as npt

from csnlp.util.random import np_random


class RandomStartPoint:
    """Class containing all the information to guide the random generation of this
    point."""

    __slots__ = ("method", "args", "kwargs")

    def __init__(self, method: str, *args: Any, **kwargs: Any) -> None:
        """Instantiates a `RandomStartPoint` object.

        Parameters
        ----------
        method : str
            Name of the method of `numpy.random.Generator` that must be used to generate
            random start locations for this point, e.g., `"unform"`, `"normal"`, etc.
        args, kwargs
            Args and kwargs with which to call the above method.
        """
        self.method = method
        self.args = args
        self.kwargs = kwargs


class RandomStartPoints:
    """Class that can be iterated to yield a set of random start points for a multistart
    NLP optimization problem."""

    __slots__ = ("np_random", "multistarts", "points")

    def __init__(
        self,
        points: Dict[str, RandomStartPoint],
        multistarts: int,
        seed: Optional[int] = None,
    ) -> None:
        """Instantiates the generator of random start points for multistarting.

        Parameters
        ----------
        points : dict of (str, RandomStartPoint)
            Dictionary containing the name of each variable, and how to generate random
            starting points for it (in the form of a `RandomStartPoint` object).
        multistarts : int
            The number of multiple start points.
        seed : int, optional
            RNG seed.
        """
        self.points = points
        self.multistarts = multistarts
        self.np_random = np_random(seed)

    def __iter__(self) -> Generator[Dict[str, npt.ArrayLike], None, None]:
        """Iterates over the random start points, yielding each time a different set."""
        points = self.points.items()
        yield from (
            {
                n: getattr(self.np_random, p.method)(*p.args, **p.kwargs)
                for n, p in points
            }
            for _ in range(self.multistarts)
        )


class StructuredStartPoint(NamedTuple):
    """Class containing all the information to guide the structured generation of this
    point."""

    lb: npt.ArrayLike
    ub: npt.ArrayLike


class StructuredStartPoints:
    """Class that can be iterated to yield a set of structured (deterministic) start
    points for a multistart NLP optimization problem. The points are linearly spaced
    between upper- and lower-bounds."""

    __slots__ = ("multistarts", "points")

    def __init__(
        self,
        points: Dict[str, StructuredStartPoint],
        multistarts: int,
    ) -> None:
        """Instantiates the generator of structured start points for multistarting.

        Parameters
        ----------
        points : dict of (str, StructuredStartPoint)
            Dictionary containing the name of each variable, and how to generate
            structured starting points for it (in the form of a `StructuredStartPoint`
            object).
        multistarts : int
            The number of multiple start points.
        """
        self.points = points
        self.multistarts = multistarts

    def __iter__(self) -> Generator[Dict[str, npt.ArrayLike], None, None]:
        """Iterates over the structured start points."""
        data = {
            n: iter(np.linspace(p.lb, p.ub, self.multistarts))  # type: ignore[arg-type]
            for n, p in self.points.items()
        }.items()
        yield from ({n: next(v) for n, v in data} for _ in range(self.multistarts))
