"""A module that provides parallelization of optimization problems for multistarting.

Motivation
==========

Sometimes, NLP problems will converge to poor local minima. This is especially true
when the nonlinearities in the problem are strong and nonconvex and the number of
optimization variables is high, increasing the chances of the gradient-based solver
getting stuck in a local minimum.

One way to mitigate this issue is to use multistart methods, which consist in running
the optimization problem from multiple starting points (i.e., initial guesses for the
primal variables) and selecting the best solution out of the ones found. The benefit is
straightforward: the more starting points are used and the more spread these are, the
higher the chances of finding a better solution (possibly, but difficulty, the global
minimum). The downside is that the computational cost increases with the number of
starting points, in particular when the repeated optimization problems are run serially.
To overcome this, parallelization of the computations is an intuitive yet effective
solution.

Overview
========

To address this problem, the :mod:`csnlp.multistart` module provides classes to run
multistart optimization problems in parallel, with different implementations of the
parallelization. These classes maintain the same interface from :class:`csnlp.Nlp`, so
from the perspective of the user little effort is required to employ multistart. In
particular, the module provides the following classes:

- :class:`MappedMultistartNlp` runs the optimization problems in parallel
  using the :func:`casadi.Function.map` function
- :class:`ParallelMultistartNlp` runs the optimization problems in parallel using the
  :class:`joblib.Parallel` class
- :class:`StackedMultistartNlp` runs the optimization problems in parallel by stacking
  them multiple times in a single large-scale optimization problem.

These are all subclasses of the base class :class:`MultistartNlp`, which provides the
interface for multistarting optimization problems (see
:meth:`MultistartNlp.solve_multi`). As aforementioned, the effectiveness of multistart
methods depends also on the quality and spread of the starting points. This module thus
also provides classes to generate these points in a structured or random way:

- :class:`RandomStartPoint` and :class:`RandomStartPoints` allows to generate random
  starting points and group them, respectively
- :class:`StructuredStartPoint` and :class:`StructuredStartPoints` allows to generate
  structured (i.e., deterministic) starting points and group them, respectively.
"""

__all__ = [
    "MappedMultistartNlp",
    "MultistartNlp",
    "ParallelMultistartNlp",
    "RandomStartPoint",
    "RandomStartPoints",
    "StackedMultistartNlp",
    "StructuredStartPoint",
    "StructuredStartPoints",
]

from .multistart_nlp import (
    MappedMultistartNlp,
    MultistartNlp,
    ParallelMultistartNlp,
    StackedMultistartNlp,
)
from .startpoints import (
    RandomStartPoint,
    RandomStartPoints,
    StructuredStartPoint,
    StructuredStartPoints,
)
