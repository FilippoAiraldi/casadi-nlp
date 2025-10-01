"""A module to provide wrappers to enhance the NLP class's capabilities.

Motivation
==========

The standard class :class:`csnlp.Nlp` provides a way to solve NLP problems; however, it
lacks many features that are useful in practice in different fields. For instance, it
does not provide a way to scale the problem, in cases where the primal variables have
widely different orders of magnitude and convergence is difficult to numerically
guarantee. It also does not provide a way to compute the sensitivity of the solution
with respect to the problem's parameters, which is useful in differentiating throught
the optimization problems.

To address this, inspired by the approach adopted by the
`gymnasium <https://gymnasium.farama.org/>`_ package, we provide a way to wrap instances
of the basic :class:`csnlp.Nlp` class with wrapper classes that can add desired
features.

Overview
========

The basic idea is to create a base class :class:`csnlp.wrappers.Wrapper` that can be
subclassed to implement the desired features. The base class provides the same interface
as :class:`csnlp.Nlp`, so that the user can interact with the wrapped instance in the
same way as with the basic NLP instance. We also provide a
:class:`csnlp.wrappers.NonRetroactiveWrapper`, which is a special wrapper that can only
wrap instances of :class:`csnlp.Nlp` before any variable, parameters, etc. is defined.

The following wrappers are provided in this module:

- :class:`csnlp.wrappers.NlpScaling`: a wrapper that scales the NLP parameters,
  variables and expressions automatically (:mod:`csnlp.scaling` provides also classes to
  inform this wrapper on how to scale the quantities)
- :class:`csnlp.wrappers.NlpSensitivity`: a wrapper that computes the sensitivity of the
  NLP solution with respect to the parameters :cite:`buskens_sensitivity_2001`
- :class:`csnlp.wrappers.Mpc`: a wrapper that facilities the creation of MPC
  optimization problems :cite:`rawlings_model_2017`
- :class:`csnlp.wrappers.ScenarioBasedMpc`: a wrapper that facilities the creation of
  MPC controllers based on the Scenario Approach :cite:`schildbach_scenario_2014`
- :class:`csnlp.wrappers.MultiScenarioMpc`: a wrapper that generalizes the above one to
  handle MPC controllers that handle multiple dynamics scenarios at once
- :class:`csnlp.wrappers.PwaMpc`: a wrapper that facilities the creation of MPC
  controllers for piecewise affine (PWA) systems :cite:`borrelli_predictive_2017`.
"""

__all__ = [
    "Mpc",
    "MultiScenarioMpc",
    "NlpScaling",
    "NlpSensitivity",
    "NonRetroactiveWrapper",
    "PwaMpc",
    "PwaRegion",
    "ScenarioBasedMpc",
    "Wrapper",
]

from .mpc.mpc import Mpc
from .mpc.multi_scenario_mpc import MultiScenarioMpc
from .mpc.pwa_mpc import PwaMpc, PwaRegion
from .mpc.scenario_based_mpc import ScenarioBasedMpc
from .scaling import NlpScaling
from .sensitivity import NlpSensitivity
from .wrapper import NonRetroactiveWrapper, Wrapper
