=============================================
CasADi-NLP: NonLinear Programming with CasADi
=============================================



Introduction
------------

**C**\ a\ **s**\ ADi-**NLP**\  (**csnlp**, for short) is a library that provides classes
and utilities to model, solve and analyse nonlinear (but not only) programmes (NLPs) in
optimization.

While it is similar in functionality (and was inspired by) :class:`casadi.Opti` (see
`this blog post <https://web.casadi.org/blog/opti/>`_ for example), it is more focused
on research purposes as it provides a more flexible and modular way to handle NLPs and
their constituent parts, such as constraints and associated dual variables.

As aforementioned, **csnlp** builds on top of the `CasADi <https://web.casadi.org/>`__
framework [1]_ to model the optimization problems and perform
symbolic differentiation, as well as the `IPOPT <https://github.com/coin-or/Ipopt>`__
solver [2]_ (though the package can be adapted to other solvers pretty easily).
The package offers also tools for the sensitivity analysis of NLPs, solving them with
multiple initial conditions, as well as for building MPC controllers.

|PyPI version| |Source Code License| |Python 3.9|

|Tests| |Downloads| |Maintainability| |Test Coverage| |Code style: black|

.. |PyPI version| image:: https://badge.fury.io/py/csnlp.svg
   :target: https://badge.fury.io/py/csnlp
   :alt: PyPI version
.. |Source Code License| image:: https://img.shields.io/badge/license-MIT-blueviolet
   :target: https://github.com/FilippoAiraldi/casadi-nlp/blob/main/LICENSE
   :alt: MIT License
.. |Python 3.9| image:: https://img.shields.io/badge/python-%3E=3.9-green.svg
   :alt: Python 3.9
.. |Tests| image:: https://github.com/FilippoAiraldi/casadi-nlp/actions/workflows/test-main.yml/badge.svg
   :target: https://github.com/FilippoAiraldi/casadi-nlp/actions/workflows/test-main.yml
   :alt: Tests
.. |Downloads| image:: https://static.pepy.tech/badge/csnlp
   :target: https://www.pepy.tech/projects/csnlp
   :alt: Downloads
.. |Maintainability| image:: https://api.codeclimate.com/v1/badges/d1cf537cff6af1a08508/maintainability
   :target: https://codeclimate.com/github/FilippoAiraldi/casadi-nlp/maintainability
   :alt: Maintainability
.. |Test Coverage| image:: https://api.codeclimate.com/v1/badges/d1cf537cff6af1a08508/test_coverage
   :target: https://codeclimate.com/github/FilippoAiraldi/casadi-nlp/test_coverage
   :alt: Test Coverage
.. |Code style: black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
   :alt: blacks



Installation
------------

You can use `pip` to install **csnlp** with the command

.. code:: bash

   pip install csnlp

**csnlp** has the following dependencies

-  Python 3.9 or higher
-  `NumPy <https://pypi.org/project/numpy/>`__
-  `CasADi <https://pypi.org/project/casadi/>`__
-  `Joblib <https://joblib.readthedocs.io/en/latest/>`__

For playing around with the source code instead, run

.. code:: bash

   git clone https://github.com/FilippoAiraldi/casadi-nlp.git

The `main` branch contains the main releases of the packages (and the occasional post
release). The `experimental` branch is reserved for the implementation and test of new
features and hosts the release candidates.


Motivating example
------------------

Here we provide a compact example on how **csnlp** can be employed to build and solve
an optimization problem. Similar to :class:`casadi.Opti`, we instantiate a class which
represents the NLP and allows us to create its variables and parameters and model its
constraints and objective. For example, suppose we'd like to solve the following problem

.. math::
      \begin{aligned}
         \min_{x,y} \quad & (1 - x)^2 + 0.2(y - x^2)^2 \\
         \text{s.t.} \quad & \left(\frac{p}{2}\right)^2 \le (x + 0.5)^2 + y^2 \le p^2.
      \end{aligned}

We can do so with the following code:

.. code:: python

   from csnlp import Nlp

   nlp = Nlp()
   x = nlp.variable("x")[0]  # create primal variable x
   y = nlp.variable("y")[0]  # create primal variable y
   p = nlp.parameter("p")  # create parameter p

   # define the objective and constraints
   nlp.minimize((1 - x) ** 2 + 0.2 * (y - x**2) ** 2)
   g = (x + 0.5) ** 2 + y**2
   nlp.constraint("c1", (p / 2) ** 2, "<=", g)
   nlp.constraint("c2", g, "<=", p**2)

   nlp.init_solver()  # initializes IPOPT under the hood
   sol = nlp.solve(pars={"p": 1.25})  # solves the NLP for parameter p=1.25

   x_opt = sol.vals["x"]   # optimal values can be retrieved via the dict .vals
   y_opt = sol.value(y)  # or the .value method

However, the package also allows to seamlessly enhance the standard :class:`csnlp.Nlp`
with different capabilities. For instance, when the problem is highly nonlinear and
necessitates to be solved with multiple initial conditions, the :mod:`csnlp.multistart`
module offers various solutions to parallelize the computations (see, e.g.,
:class:`csnlp.multistart.ParallelMultistartNlp`). The :mod:`csnlp.wrappers` module
offers instead a set of wrappers that can be used to augment the NLP with additional
capabilities, without modifying the original NLP instance: as of now, wrappers have been
implemented for

- sensitivity analysis (see :class:`csnlp.wrappers.NlpSensitivity` [3]_)
- Model Predictive Control (see :class:`csnlp.wrappers.Mpc` [4]_ and
  :class:`csnlp.wrappers.ScenarioBasedMpc` [5]_)
- NLP scaling (see :class:`csnlp.wrappers.NlpScaling` and :mod:`csnlp.core.scaling`).

For example, if we'd like to compute the sensitivity
:math:`\frac{\partial y}{\partial p}` of the optimal primal variable :math:`y` with
respect to the parameter :math:`p`, we just need to wrap the :class:`csnlp.Nlp` instance
with the :class:`csnlp.wrappers.NlpSensitivity` wrapper, which is specialized in
differentiating the optimization problem. This in turn allows us to compute the
first-order :math:`\frac{\partial y}{\partial p}` and second sensitivities
:math:`\frac{\partial^2 y}{\partial p^2}` (``dydp`` and ``d2ydp2``, respectively) as
such:

.. code:: python

   from csnlp import wrappers

   nlp = wrappers.NlpSensitivity(nlp)
   dydp, d2ydp2 = nlp.parametric_sensitivity()

In other words, these sensitivities provide the jacobian and hessian
that locally approximate the solution w.r.t. the parameter :math:`p`. As
shown in the corresponding example but not in this quick demonstation, the sensitivity
can be also computed for any generic expression :math:`z(x(p),\lambda(p),p)` that is a
function of the primal :math:`x` and dual :math:`\lambda` variables, and the parameters
:math:`p`. Moreover, the sensitivity computations can be carried out symbolically (more
demanding) or numerically (more stable and reliable).

Similarly, a :class:`csnlp.Nlp` can be wrapped in a :class:`csnlp.wrappers.Mpc` wrapper
that makes it easier to build such finite-horizon optimal controllers for model-based
control applications.



Examples
--------

Our
`examples <https://github.com/FilippoAiraldi/casadi-nlp/tree/main/examples>`__
subdirectory contains example applications of this package in NLP
optimization, sensitivity analysis, scaling of NLPs, and optimal
control.




License
-------

The repository is provided under the MIT License. See the LICENSE file
included with this repository.



Author
------

`Filippo Airaldi <https://www.tudelft.nl/staff/f.airaldi/>`__, PhD
Candidate [f.airaldi@tudelft.nl \| filippoairaldi@gmail.com]

   `Delft Center for Systems and
   Control <https://www.tudelft.nl/en/me/about/departments/delft-center-for-systems-and-control/>`__
   in `Delft University of Technology <https://www.tudelft.nl/en/>`__

Copyright (c) 2024 Filippo Airaldi.

Copyright notice: Technische Universiteit Delft hereby disclaims all
copyright interest in the program “csnlp” (CasADi-NLP: NonLinear Programming with
CasADi) written by the Author(s). Prof. Dr. Ir. Fred van Keulen, Dean of ME.



References
----------

.. [1] Andersson, J.A.E., Gillis, J., Horn, G., Rawlings, J.B., and Diehl, M. (2019).
 *CasADi: a software framework for nonlinear optimization and optimal control*.
 Mathematical Programming Computation, 11(1), 1–36.

.. [2] Wachter, A. and Biegler, L.T. (2006). *On the implementation of an interior-point
 filter line-search algorithm for large-scale nonlinear programming*. Mathematical
 Programming, 106(1), 25–57.

.. [3] Büskens, C. and Maurer, H. (2001). *Sensitivity analysis and real-time
 optimization of parametric nonlinear programming problems*. In M. Grötschel, S.O.
 Krumke, and J. Rambau (eds.), Online Optimization of Large Scale Systems, 3–16.
 Springer, Berlin, Heidelberg

.. [4] Rawlings, J.B., Mayne, D.Q. and Diehl, M., 2017. *Model Predictive Control:
 theory, computation, and design (Vol. 1)*. Madison, WI: Nob Hill Publishing.

.. [5] Schildbach, G., Fagiano, L., Frei, C. and Morari, M., 2014. *The scenario
 approach for stochastic model predictive control with bounds on closed-loop constraint
 violations*. Automatica, 50(12), pp.3009-3018.
