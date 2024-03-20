from collections.abc import Generator, Sequence
from typing import Any, NamedTuple, Optional, Union

import numpy as np
import numpy.typing as npt


class RandomStartPoint:
    """Class containing all the information to guide the random generation of this
    point."""

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

    def __init__(
        self,
        points: dict[str, RandomStartPoint],
        multistarts: int,
        biases: Optional[dict[str, npt.ArrayLike]] = None,
        scales: Optional[dict[str, npt.ArrayLike]] = None,
        seed: Union[
            None,
            int,
            Sequence[int],
            np.random.SeedSequence,
            np.random.BitGenerator,
            np.random.Generator,
        ] = None,
    ) -> None:
        """Instantiates the generator of random start points for multistarting.

        Parameters
        ----------
        points : dict of (str, RandomStartPoint)
            Dictionary containing the name of each variable, and how to generate random
            starting points for it (in the form of a `RandomStartPoint` object).
        multistarts : int
            The number of multiple start points.
        biases : dict of (str, array_like), optional
            Biases to add to the generated random points under the same name. If `None`,
            no bias is added.
        scales : float, or array of floats
            Scales to multiplty the generated random points with, under the same name.
            If `None`, no scale is multiplied.
        seed : None, int, array_like[ints], SeedSequence, BitGenerator, Generator
            RNG seed.
        """
        self.points = points
        self.multistarts = multistarts
        self.biases = biases or {}
        self.scales = scales or {}
        self.np_random = np.random.default_rng(seed)

    def __iter__(self) -> Generator[dict[str, npt.ArrayLike], None, None]:
        """Iterates over the random start points, yielding each time a different set."""
        biases = self.biases
        scales = self.scales
        points = self.points
        out = {}
        for _ in range(self.multistarts):
            out.clear()
            for name, point in points.items():
                val = getattr(self.np_random, point.method)(*point.args, **point.kwargs)
                if name in scales:
                    val = np.multiply(val, scales[name])
                if name in biases:
                    val = np.add(val, biases[name])
                out[name] = val
            yield out.copy()


class StructuredStartPoint(NamedTuple):
    """Class containing all the information to guide the structured generation of this
    point."""

    lb: npt.ArrayLike
    ub: npt.ArrayLike


class StructuredStartPoints:
    """Class that can be iterated to yield a set of structured (deterministic) start
    points for a multistart NLP optimization problem. The points are linearly spaced
    between upper- and lower-bounds."""

    def __init__(
        self,
        points: dict[str, StructuredStartPoint],
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

    def __iter__(self) -> Generator[dict[str, npt.ArrayLike], None, None]:
        """Iterates over the structured start points."""
        data = {
            n: iter(np.linspace(p.lb, p.ub, self.multistarts))  # type: ignore[arg-type]
            for n, p in self.points.items()
        }.items()
        yield from ({n: next(v) for n, v in data} for _ in range(self.multistarts))
