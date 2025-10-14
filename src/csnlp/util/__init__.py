r"""A module with utility functions and classes around and for optimization.

Overview
========

It contains the following submodules:

- :mod:`csnlp.util.docs`: a collection of stand-alone functions to extract information
  from the CasADi documentation via code. In particular, this module offers a way to get
  the solvers that are available in CasADi (i.e., they have an interface) as well as and
  their options. The functions are taken from the
  `MPCTools <https://bitbucket.org/rawlings-group/mpc-tools-casadi/src/master/mpctools/util.py>`_
  repository by the Rawlings' group.
- :mod:`csnlp.util.io`: a collection of utilities for input/output operations. The goals
  of these methods are:

   * compatibility of pickling/deepcopying with CasADi objects and classes that hold
     such objects (since these are often not picklable)
   * saving and loading data to/from files, possibly compressed.

- :mod:`csnlp.util.math`: a collection of stand-alone functions that implement some of
  the basic mathematical operations that are not available in CasADi. The
  implementations are simple and thus not optimized for performance. They are meant to
  be used as a fallback when the CasADi ally does not provide the required
  functionality.

Submodules
==========

.. autosummary::
   :toctree: generated
   :template: module.rst

   docs
   io
   math
"""

__all__ = ["docs", "io", "math"]

from . import docs, io, math
