"""A module for defining NLPs and its building blocks. From simplest to most complex,
and following the inheritance hierarchy, these are:

- :class:`csnlp.nlps.HasParameters`: a class for the creation and storage of symbolic
  parameters for NLP problems
- :class:`csnlp.nlps.HasVariables`: a class for the creation and storage of symbolic
  variables for NLP problems
- :class:`csnlp.nlps.HasConstraints`: a class for the creation and storage of symbolic
  constraints (dependent on variables and parameters) for NLP problems. The
  constraints can be equality, inequality, lower- or upper-bound.
- :class:`csnlp.nlps.HasObjective`: a class for the assignment of a scalar minimization
  objective function (dependent on variables and parameters).
- :class:`csnlp.Nlp`: a class that combines all the above building blocks into a
  full-fledged NLP problem.
"""

__all__ = ["HasConstraints", "HasObjective", "HasParameters", "HasVariables", "Nlp"]

from .constraints import HasConstraints
from .nlp import Nlp
from .objective import HasObjective
from .parameters import HasParameters
from .variables import HasVariables
