Features
========

**csnlp** builds on top of the
`CasADi <https://web.casadi.org/>`_ framework :cite:`andersson_casadi_2019` to model
the optimization problems and perform symbolic differentiation, and heavily relies on
the `IPOPT <https://github.com/coin-or/Ipopt>`_ solver
:cite:`wachter_implementation_2006` (though the package allows the user to seamlessly
switch to other solvers supported by CasADi). While it is similar in functionality (and
was inspired by) the :class:`casadi.Opti` stack (see
`this blog post <https://web.casadi.org/blog/opti/>`_ for example), it is more tailored
to research as

1. it is more flexible, since it is written in Python and allows the user to easily
   access all the constituents of the optimization problem (e.g. the objective function,
   constraints, dual variables, bounds, etc.)

2. it is more modular, since it allows the base :class:`csnlp.Nlp` class to be wrapped
   with additional functionality (e.g. sensitivity, Model Predictive Control, etc.),
   and it provides parallel implementations in case of multistarting in the
   :mod:`csnlp.multistart` module.

The package offers also tools for the sensitivity analysis of NLPs, solving them with
multiple initial conditions, as well as for building MPC controllers. The library is not
meant to be a faster alternative to :class:`casadi.Opti`, but rather a more flexible and
modular one for research purposes.
