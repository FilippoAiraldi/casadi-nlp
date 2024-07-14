r"""This module contains the core components, aside :class:`csnlp.Nlp` and its base
classes, that are used to build the package.

Overview
========

It contains the following submodules:

- :mod:`csnlp.core.cache`: a collection of methods to handle caching in the package. In
  particular, it offers a decorator :func:`invalidate_cache` that allows to invalidate
  the cache of a given set of other cached properties or methods when the decorated
  method is  invoked, as well as a function :func:`invalidate_caches_of` that allows to
  invalidate the cache of a given object on the fly.
- :mod:`csnlp.core.data`: a collection of functions for manipulating data in CasADi, in
  particular, on how to convert to and from numpy arrays and CasADi symbolic variables,
  and how to find the indices of a symbolic variable in a vector of symbols.
- :mod:`csnlp.core.debug`: scontains classes for storing debug information on the
  parameters, variables and constraints in an instance of the :class:`csnlp.Nlp` class.
- :mod:`csnlp.core.derivatives`: a collection of two methods for computing higher-order
  sensitivities (i.e., Jacobian and Hessian) w.r.t. CasADi symbolic variables. Natively,
  CasADi does not support jacobian or hessian for matrices (or at least, they will be
  flattened). These  "higher-order" functions allows to compute the jacobian and hessian
  of a matrix w.r.t. another matrix.
- :mod:`csnlp.core.scaling`: a collection of classes to perform scaling of variables in
  an :class:`csnlp.Nlp` instance wrapped with :class:`csnlp.wrappers.NlpScaling`. The
  classes in this module inform the wrapper on which variables or parameters to scale
  and how to scale them.
- :mod:`csnlp.core.solutions`: contains classes and methods to store the solution of an
  NLP problem after a call to :meth:`csnlp.Nlp.solve` or
  :meth:`csnlp.multistart.MultistartNlp.solve_multi`.

Submodules
==========

.. autosummary::
   :toctree: generated
   :template: module.rst

   cache
   data
   debug
   derivatives
   scaling
   solutions
"""
