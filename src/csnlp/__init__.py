r"""**C**\ a\ **s**\ ADi-**NLP**\  (**csnlp**, for short) is a library that provides
classes and utilities to model, solve and analyse nonlinear (but not only) programmes
(NLPs) for optimization purposes.

==================== =======================================================
**Documentation**        https://casadi-nlp.readthedocs.io/en/latest/

**Download**             https://pypi.python.org/pypi/csnlp/

**Source code**          https://github.com/FilippoAiraldi/casadi-nlp/

**Report issues**        https://github.com/FilippoAiraldi/casadi-nlp/issues
==================== =======================================================
"""

__version__ = "1.6.8rc3"

__all__ = ["Nlp", "Solution", "multistart", "scaling"]

from . import multistart
from .core import scaling
from .core.solutions import Solution
from .nlps.nlp import Nlp
