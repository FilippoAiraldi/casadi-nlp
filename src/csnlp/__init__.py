r"""**C**\ a\ **s**\ ADi-**NLP**\  (**csnlp**, for short) is a library that provides
classes and utilities to model, solve and analyse nonlinear (but not only) programmes
(NLPs) in optimization.

While it is similar in functionality (and was inspired by) the :class:`casadi.Opti`
stack (see `this blog post <https://web.casadi.org/blog/opti/>`_ for example), it is
more tailored to research as

1. it is more flexible, since it is written in Python and allows the user to easily
   access all the constituents of the optimization problem (e.g. the objective function,
   constraints, dual variables, bounds, etc.)

2. it is more modular, since it allows the base :class:`csnlp.Nlp` class to be wrapped
    with additional functionality (e.g. sensitivity, Model Predictive Control, etc.),
    and it provides parallel implementations in case of multistarting in the
    :mod:`csnlp.multistart` module.

The library is not meant to be a faster alternative to :class:`casadi.Opti`, but rather
a more flexible and modular one for research purposes.

    ==================== ========================================================
    **Documentation:**       In progress

    **Download:**            https://pypi.python.org/pypi/csnlp/

    **Source code:**         https://github.com/FilippoAiraldi/casadi-nlp/

    **Report issues:**       https://github.com/FilippoAiraldi/casadi-nlp/issues/
    ==================== ========================================================
"""

__version__ = "1.5.10.post2"

__all__ = ["Nlp", "Solution", "multistart", "scaling"]

from . import multistart
from .core import scaling
from .core.solutions import Solution
from .nlps.nlp import Nlp
