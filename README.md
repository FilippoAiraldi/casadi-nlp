# NonLinear Programming with CasADi

**C**a**s**ADi-**NLP** (**csnlp**, for short) is a library that provides classes
and utilities to model, solve and analyse nonlinear (but not only) programmes (NLPs) for
optimization purposes.

> |   |   |
> |---|---|
> | **Documentation** | <https://casadi-nlp.readthedocs.io/en/latest/>         |
> | **Download**      | <https://pypi.python.org/pypi/csnlp/>                  |
> | **Source code**   | <https://github.com/FilippoAiraldi/casadi-nlp/>        |
> | **Report issues** | <https://github.com/FilippoAiraldi/casadi-nlp/issues/> |

[![PyPI version](https://badge.fury.io/py/csnlp.svg)](https://badge.fury.io/py/csnlp)
[![Source Code License](https://img.shields.io/badge/license-MIT-blueviolet)](https://github.com/FilippoAiraldi/casadi-nlp/blob/experimental/LICENSE)
![Python 3.9](https://img.shields.io/badge/python->=3.9-green.svg)

[![Tests](https://github.com/FilippoAiraldi/casadi-nlp/actions/workflows/tests.yml/badge.svg)](https://github.com/FilippoAiraldi/casadi-nlp/actions/workflows/tests.yml)
[![Docs](https://readthedocs.org/projects/casadi-nlp/badge/?version=latest)](https://casadi-nlp.readthedocs.io/en/latest/?badge=latest)
[![Downloads](https://static.pepy.tech/badge/csnlp)](https://www.pepy.tech/projects/csnlp)
[![Maintainability](https://qlty.sh/gh/FilippoAiraldi/projects/casadi-nlp/maintainability.svg)](https://qlty.sh/gh/FilippoAiraldi/projects/casadi-nlp)
[![Code Coverage](https://qlty.sh/gh/FilippoAiraldi/projects/casadi-nlp/coverage.svg)](https://qlty.sh/gh/FilippoAiraldi/projects/casadi-nlp)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://docs.astral.sh/ruff/)

---

## Features

**csnlp** builds on top of the [CasADi](https://web.casadi.org/)
framework [[1]](#1) to model the optimization problems and perform symbolic
differentiation, and heavily relies on the [IPOPT](https://github.com/coin-or/Ipopt)
solver [[2]](#2) (though the package allows the user to seamlessly switch to other
solvers supported by CasADi). While it is similar in functionality (and was inspired by)
the CasADi's
[Opti Stack](https://web.casadi.org/api/html/dd/dc6/classcasadi_1_1Opti.html) (see
[this blog post](https://web.casadi.org/blog/opti/) for example), it is more tailored to
research as

1. it is more flexible, since it is written in Python and allows the user to easily
   access all the constituents of the optimization problem (e.g. the objective function,
   constraints, dual variables, bounds, etc.)

2. it is more modular, since it allows the base `csnlp.Nlp` class to be wrapped with
   additional functionality (e.g. sensitivity, Model Predictive Control, etc.), and it
   provides parallel implementations in case of multistarting in the `csnlp.multistart`
   module.

The package offers also tools for the sensitivity analysis of NLPs, solving them with
multiple initial conditions, as well as for building MPC controllers. The library is not
meant to be a faster alternative to `casadi.Opti`, but rather a more flexible and
modular one for research purposes.

---

## Installation

### Using `pip`

You can use `pip` to install **csnlp** with the command

```bash
pip install csnlp
```

**csnlp** has the following dependencies

- Python 3.9 or higher
- [NumPy](https://pypi.org/project/numpy/)
- [CasADi](https://pypi.org/project/casadi/)
- [Joblib](https://joblib.readthedocs.io/)

### Using source code

If you'd like to play around with the source code instead, run

```bash
git clone https://github.com/FilippoAiraldi/casadi-nlp.git
```

The `main` branch contains the main releases of the packages (and the occasional post
release). The `experimental` branch is reserved for the implementation and test of new
features and hosts the release candidates. You can then install the package to edit it
as you wish as

```bash
pip install -e /path/to/casadi-nlp
```

---

## Getting started

Here we provide a compact example on how **csnlp** can be employed to build and solve
an optimization problem. Similar to
[Opti](https://web.casadi.org/api/html/dd/dc6/classcasadi_1_1Opti.html), we instantiate
a class which represents the NLP and allows us to create its variables and parameters
and model its constraints and objective. For example, suppose we'd like to solve the
problem

$$
\min_{x,y}{ (1 - x)^2 + 0.2(y - x^2)^2 \text{ s.t. } (p/2)^2 \le (x + 0.5)^2 + y^2 \le p^2 }
$$

We can do so with the following code:

```python
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
```

However, the package also allows to seamlessly enhance the standard `csnlp.Nlp` with
different capabilities. For instance, when the problem is highly nonlinear and
necessitates to be solved with multiple initial conditions, the `csnlp.multistart`
module offers various solutions to parallelize the computations (see, e.g.,
`csnlp.multistart.ParallelMultistartNlp`). The `csnlp.wrappers` module offers instead a
set of wrappers that can be used to augment the NLP with additional capabilities without
modifying the original NLP instance: as of now, wrappers have been implemented for

The package also allows to enhance the NLP with different capabilities with, e.g.,
multistart (see `csnlp.MultistartNlp`) or by wrapping it. As of now, wrappers have been
implemented for

- sensitivity analysis (see `csnlp.wrappers.NlpSensitivity` [[3]](#3))
- Model Predictive Control (see `csnlp.wrappers.Mpc` [[4]](#4) and
  `csnlp.wrappers.ScenarioBasedMpc` [[5]](#5))
- NLP scaling (see `csnlp.wrappers.NlpScaling` and `csnlp.core.scaling`).

For example, if we'd like to compute the sensitivity $\frac{\partial y}{\partial p}$ of
the optimal primal variable $y$ with respect to the parameter $p$, we just need to wrap
the `csnlp.Nlp` instance with the `csnlp.wrappers.NlpSensitivity` wrapper, which is
specialized in differentiating the optimization problem. This in turn allows us to
compute the first-order $\frac{\partial y}{\partial p}$ and second sensitivities
$\frac{\partial^2 y}{\partial p^2}$ (`dydp` and `d2ydp2`, respectively) as such:

```python
from csnlp import wrappers

nlp = wrappers.NlpSensitivity(nlp)
dydp, d2ydp2 = nlp.parametric_sensitivity()
```

In other words, these sensitivities provide the jacobian and hessian
that locally approximate the solution w.r.t. the parameter $p$. As
shown in the corresponding example but not in this quick demonstation, the sensitivity
can be also computed for any generic expression $z(x(p),\lambda(p),p)$ that is a
function of the primal $x$ and dual $\lambda$ variables, and the parameters
$p$. Moreover, the sensitivity computations can be carried out symbolically (more
demanding) or numerically (more stable and reliable).

Similarly, a `csnlp.Nlp` instance can be wrapped in a `csnlp.wrappers.Mpc` wrapper
that makes it easier to build such finite-horizon optimal controllers for model-based
control applications.

---

## Examples

Our [examples](https://github.com/FilippoAiraldi/casadi-nlp/tree/main/examples)
subdirectory contains example applications of this package in NLP optimization,
sensitivity analysis, scaling of NLPs, and optimal control.

---

## License

The repository is provided under the MIT License. See the LICENSE file included with
this repository.

---

## Author

[Filippo Airaldi](https://www.tudelft.nl/staff/f.airaldi/), PhD Candidate
[f.airaldi@tudelft.nl | filippoairaldi@gmail.com]

> [Delft Center for Systems and Control](https://www.tudelft.nl/en/me/about/departments/delft-center-for-systems-and-control/)
in [Delft University of Technology](https://www.tudelft.nl/en/)

Copyright (c) 2024 Filippo Airaldi.

Copyright notice: Technische Universiteit Delft hereby disclaims all copyright interest
in the program “csnlp” (Nonlinear Progamming with CasADi) written by the Author(s).
Prof. Dr. Ir. Fred van Keulen, Dean of ME.

---

## References

<a id="1">[1]</a>
Andersson, J.A.E., Gillis, J., Horn, G., Rawlings, J.B., and Diehl, M. (2019).
[CasADi: a software framework for nonlinear optimization and optimal control](https://link.springer.com/article/10.1007/s12532-018-0139-4).
Mathematical Programming Computation, 11(1), 1–36.

<a id="2">[2]</a>
Wachter, A. and Biegler, L.T. (2006).
[On the implementation of an interior-point filter line-search algorithm for large-scale nonlinear programming](https://link.springer.com/article/10.1007/s10107-004-0559-y).
Mathematical Programming, 106(1), 25–57.

<a id="3">[3]</a>
Büskens, C. and Maurer, H. (2001).
[Sensitivity analysis and real-time optimization of parametric nonlinear programming problems](https://link.springer.com/chapter/10.1007/978-3-662-04331-8_1).
In M. Grötschel, S.O. Krumke, and J. Rambau (eds.), Online Optimization of Large Scale Systems, 3–16. Springer, Berlin, Heidelberg

<a id="4">[4]</a>
Rawlings, J.B., Mayne, D.Q. and Diehl, M., 2017.
[Model Predictive Control: theory, computation, and design (Vol. 2)](https://sites.engineering.ucsb.edu/~jbraw/mpc/).
Madison, WI: Nob Hill Publishing.

<a id="5">[5]</a>
Schildbach, G., Fagiano, L., Frei, C. and Morari, M., 2014.
[The Scenario Approach for stochastic Model Predictive Control with bounds on closed-loop constraint violations](https://www.sciencedirect.com/science/article/pii/S0005109814004166).
Automatica, 50(12), pp.3009-3018.
