A simple optimization problem
=============================

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


Enhancing the NLP class
=======================

However, the package also allows to seamlessly enhance the standard :class:`csnlp.Nlp`
with different capabilities. For instance, when the problem is highly nonlinear and
necessitates to be solved with multiple initial conditions, the :mod:`csnlp.multistart`
module offers various solutions to parallelize the computations (see, e.g.,
:class:`csnlp.multistart.ParallelMultistartNlp`). The :mod:`csnlp.wrappers` module
offers instead a set of wrappers that can be used to augment the NLP with additional
capabilities, without modifying the original NLP instance: as of now, wrappers have been
implemented for

- sensitivity analysis (see :class:`csnlp.wrappers.NlpSensitivity`
  :cite:`buskens_sensitivity_2001`)
- Model Predictive Control (see :class:`csnlp.wrappers.Mpc` :cite:`rawlings_model_2017`
  and :class:`csnlp.wrappers.ScenarioBasedMpc` :cite:`schildbach_scenario_2014`)
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
