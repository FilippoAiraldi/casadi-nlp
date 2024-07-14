"""A module to provide wrappers to enhance the NLP class's capabilities.

Motivation
==========

The standard class :class:`csnlp.Mpc` provides a way to solve NLP problems; however, it
lacks many features that are useful in practice in different fields. For instance, it
does not provide a way to scale the problem, in cases where the primal variables have
widely different orders of magnitude and convergence is difficult to numerically
guarantee. It also does not provide a way to compute the sensitivity of the solution
with respect to the problem's parameters, which is useful in differentiating throught
the optimization problems.

To address this, inspired by the approach adopted by the
`gymnasium <https://gymnasium.farama.org/>` package, we provide a way to wrap instances
of the basic :class:`csnlp.Mpc` class with wrapper classes that can add desired
features.

Overview
========

The basic idea is to create a base class :class:`csnlp.Wrapper` that can be subclassed
to implement the desired features. The base class provides the same interface as
:class:`csnlp.Mpc`, so that the user can interact with the wrapped instance in the same
way as with the basic NLP instance. We also provide a
:class:`csnlp.NonRetroactiveWrapper`, which is a special wrapper that can only wrap
instances of :class:`csnlp.Nlp` before any variable, parameters, etc. is defined.

The following wrappers are provided in this module:

- :class:`csnlp.wrappers.NlpScaling`: a wrapper that scales the NLP parameters,
  variables and expressions automatically (:mod:`csnlp.scaling` provides also classes to
  inform this wrapper on how to scale the quantities)
- :class:`csnlp.wrappers.NlpSensitivity`: a wrapper that computes the sensitivity of the
  NLP solution with respect to the parameters [1]_
- :class:`csnlp.wrappers.Mpc`: a wrapper that facilities the creation of MPC
  optimization problems [2]_
- :class:`csnlp.wrappers.ScenarioBasedMpc`: a wrapper that facilities the creation of
  MPC controllers based on the Scenario Approach [3]_.

References
==========

.. [1] Büskens, C. and Maurer, H. (2001). *Sensitivity analysis and real-time
 optimization of parametric nonlinear programming problems*. In M. Grötschel, S.O.
 Krumke, and J. Rambau (eds.), Online Optimization of Large Scale Systems, 3–16.
 Springer, Berlin, Heidelberg

.. [2] Rawlings, J.B., Mayne, D.Q. and Diehl, M., 2017. *Model Predictive Control:
 theory, computation, and design (Vol. 1)*. Madison, WI: Nob Hill Publishing.

.. [3] Schildbach, G., Fagiano, L., Frei, C. and Morari, M., 2014. *The scenario
 approach for stochastic model predictive control with bounds on closed-loop constraint
 violations*. Automatica, 50(12), pp.3009-3018.
"""

__all__ = [
    "Mpc",
    "NlpScaling",
    "NlpSensitivity",
    "NonRetroactiveWrapper",
    "ScenarioBasedMpc",
    "Wrapper",
]

from .mpc.mpc import Mpc
from .mpc.scenario_based_mpc import ScenarioBasedMpc
from .scaling import NlpScaling
from .sensitivity import NlpSensitivity
from .wrapper import NonRetroactiveWrapper, Wrapper
