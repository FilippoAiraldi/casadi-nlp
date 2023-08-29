# Nonlinear Programming with CasADi

**csnlp** provides classes and utilities to model, solve and analyse nonlinear programmes (NLPs) in optimization.

In particular, it makes use of the [CasADi](https://web.casadi.org/) framework [[1]](#1) to model the optimization problems and perform symbolic differentiation, as well as the [IPOPT](https://github.com/coin-or/Ipopt) solver [[2]](#2) (though the package can be adapted to other solvers pretty easily). The package offers also tools for the sensitivity analysis of NLPs, solving them with multiple initial conditions, as well as for building MPC controllers.

[![PyPI version](https://badge.fury.io/py/csnlp.svg)](https://badge.fury.io/py/csnlp)
[![Source Code License](https://img.shields.io/badge/license-MIT-blueviolet)](https://github.com/FilippoAiraldi/casadi-nlp/blob/dev/LICENSE)
![Python 3.8](https://img.shields.io/badge/python->=3.8-green.svg)

[![Tests](https://github.com/FilippoAiraldi/casadi-nlp/actions/workflows/test-experimental.yml/badge.svg)](https://github.com/FilippoAiraldi/casadi-nlp/actions/workflows/test-experimental.yml)
[![Downloads](https://static.pepy.tech/badge/csnlp)](https://www.pepy.tech/projects/csnlp)
[![Maintainability](https://api.codeclimate.com/v1/badges/d1cf537cff6af1a08508/maintainability)](https://codeclimate.com/github/FilippoAiraldi/casadi-nlp/maintainability)
[![Test Coverage](https://api.codeclimate.com/v1/badges/d1cf537cff6af1a08508/test_coverage)](https://codeclimate.com/github/FilippoAiraldi/casadi-nlp/test_coverage)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## Installation

To install the package, run

```bash
pip install csnlp
```

**csnlp** has the following dependencies

- Python 3.8 or higher
- [NumPy](https://pypi.org/project/numpy/)
- [CasADi](https://pypi.org/project/casadi/)
- [Joblib](https://joblib.readthedocs.io/en/latest/)

For playing around with the source code instead, run

```bash
git clone https://github.com/FilippoAiraldi/casadi-nlp.git
```

---

## Usage

Similar to CasADi Opti, we instantiate a class that represents the NLP and allows to model its variables, parameters, constraints and objective. A simple example is below:

```python
from csnlp import Nlp

nlp = Nlp()
x = nlp.variable("x")[0]
y = nlp.variable("y")[0]
p = nlp.parameter("p")

nlp.minimize((1 - x) ** 2 + 0.2 * (y - x**2) ** 2)

g = (x + 0.5) ** 2 + y**2
nlp.constraint("c1", (p / 2) ** 2, "<=", g)
nlp.constraint("c2", g, "<=", p**2)

nlp.init_solver()  # initializes IPOPT under the hood
sol = nlp.solve(pars={"p": 1.25})  # solves the NLP for parameter p=1.25

x_opt = sol.vals["x"]
y_opt = sol.value(y)
```

The package also allows to enhance the NLP with different capabilities with, e.g., multistart (see `csnlp.MultistartNlp`) or by wrapping it. As of now, wrappers have been implemented for

- sensitivity analysis (see `csnlp.wrappers.NlpSensitivity`; based on [[3]](#3))
- Model Predictive Control (see `csnlp.wrappers.Mpc`; [[4]](#4) for details)
- NLP scaling (see `csnlp.wrappers.NlpScaling`).

For instance, to compute the sensitivity of the optimal primal-dual variables `y` with respect to the parameters `p` of the NLP , first we need to augment the capabilities of the NLP with a wrapper specialized in differentiating the optimization problem, and then compute the first-order and second sensitivities (`dydp` and `d2ydp2`, respectively) as such:

```python
from csnlp import wrappers

nlp = wrappers.NlpSensitivity(nlp)
dydp, d2ydp2 = nlp.parametric_sensitivity()
```

In other words, these sensitivities provide the jacobian and hessian that locally approximate the solution w.r.t. the parameters `p`. As shown in the examples, the sensitivity can be also computed for any generic expression `z` that is a function of the primal-dual variables and the parameters, i.e., `z(x(p),lam(p),p)`, and computations can be carried out symbolically or numerically (more stable).

Similarly, an NLP instance can be wrapped in an MPC wrapper that makes it easier to build such controller.

## Examples

Our [examples](https://github.com/FilippoAiraldi/casadi-nlp/tree/dev/examples) subdirectory contains example applications of this package in NLP optimization, sensitivity analysis, scaling of NLPs, and optimal control.

---

## License

The repository is provided under the MIT License. See the LICENSE file included with this repository.

---

## Author

[Filippo Airaldi](https://www.tudelft.nl/staff/f.airaldi/), PhD Candidate [f.airaldi@tudelft.nl | filippoairaldi@gmail.com]

> [Delft Center for Systems and Control](https://www.tudelft.nl/en/3me/about/departments/delft-center-for-systems-and-control/) in [Delft University of Technology](https://www.tudelft.nl/en/)

Copyright (c) 2023 Filippo Airaldi.

Copyright notice: Technische Universiteit Delft hereby disclaims all copyright interest in the program “csnlp” (Nonlinear Progamming with CasADi) written by the Author(s). Prof. Dr. Ir. Fred van Keulen, Dean of 3mE.

---

## References

<a id="1">[1]</a>
Andersson, J.A.E., Gillis, J., Horn, G., Rawlings, J.B., and Diehl, M. (2019). [CasADi: a software framework for nonlinear optimization and optimal control](https://link.springer.com/article/10.1007/s12532-018-0139-4). Mathematical Programming Computation, 11(1), 1–36.

<a id="2">[2]</a>
Wachter, A. and Biegler, L.T. (2006). [On the implementation of an interior-point filter line-search algorithm for large-scale nonlinear programming](https://link.springer.com/article/10.1007/s10107-004-0559-y). Mathematical Programming, 106(1), 25–57.

<a id="3">[3]</a>
Büskens, C. and Maurer, H. (2001). [Sensitivity analysis and real-time optimization of parametric nonlinear programming problems](https://link.springer.com/chapter/10.1007/978-3-662-04331-8_1). In M. Grötschel, S.O. Krumke, and J. Rambau (eds.), Online Optimization of Large Scale Systems, 3–16. Springer, Berlin, Heidelberg

<a id="4">[4]</a>
Rawlings, J.B., Mayne, D.Q. and Diehl, M., 2017. [Model Predictive Control: theory, computation, and design (Vol. 2)](https://sites.engineering.ucsb.edu/~jbraw/mpc/). Madison, WI: Nob Hill Publishing.
