[![Source Code License](https://img.shields.io/badge/license-MIT-blueviolet)](https://github.com/FilippoAiraldi/casadi-nlp/blob/release/LICENSE)
![Python 3.8](https://img.shields.io/badge/python->=3.8-green.svg)


# Nonlinear Programming with CasADi

**casadi-nlp** provides classes and utilities to model, solve and analyse nonlinear programmes (NLPs) in optimization.

In particular, it makes use of the [CasADi](https://web.casadi.org/) framework [[1]](#1) to model the optimization problems and perform symbolic differentiation, as well as the [IPOPT](https://github.com/coin-or/Ipopt) solver [[2]](#2) (though the package can be adapted to other solvers pretty easily). The package offers also tools for the sensitivity analysis of NLPs.

---
## Installation
To install the package, run
```bash
pip install casadi-nlp
```
**casadi-nlp** has the following dependencies

- Python 3.8
- [NumPy](https://pypi.org/project/numpy/)
- [CasADi](https://pypi.org/project/casadi/)

For playing around with the source code instead, run
```bash
git clone https://github.com/FilippoAiraldi/casadi-nlp.git
``` 


---
## Usage
Similar to CasADi Opti, we instantiate a class that represents the NLP and allows to model its variables, parameters, constraints and objective. A simple example is below
```python
from casadi_nlp import Nlp

nlp = Nlp()
x = nlp.variable('x')[0]
y = nlp.variable('y')[0]
p = nlp.parameter('p')

nlp.minimize((1 - x)**2 + 0.2 * (y - x**2)**2)

g = (x + 0.5)**2 + y**2
nlp.constraint('c1', (p / 2)**2, '<=', g)
nlp.constraint('c2', g, '<=', p**2)

nlp.init_solver()  # initializes IPOPT under the hood
sol = nlp.solve(pars={'p': 1.25})  # solves the NLP for parameter p=1.25

x_opt = sol.vals['x']  
y_opt = sol.value(y)
```
To compute the sensitivity of the optimal primal variables with respect to the 
parameters, first we need to augment the capabilities of the NLP with a wrapper
specialized in differentiating the optimization problem, and then compute the
first-order sensitivities
```python
from casadi_nlp import wrappers

nlp = wrappers.DifferentiableNlp(nlp)

# dxydp is the sensitivity of primal variables x and y w.r.t. parameters p
dxydp = nlp.parametric_sensitivity()[:nlp.nx]
```


## Examples
Our [examples](examples) subdirectory contains other applications of this package in NLP optimization and sensitivity analysis.

---
## License
The repository is provided under the MIT License. See the LICENSE file included with this repository. 


---
## Author
[Filippo Airaldi](https://www.tudelft.nl/staff/f.airaldi/), PhD Candidate [f.airaldi@tudelft.nl | filippoairaldi@gmail.com]

> [Delft Center for Systems and Control](https://www.tudelft.nl/en/3me/about/departments/delft-center-for-systems-and-control/) in [Delft University of Technology](https://www.tudelft.nl/en/)

---
## References
<a id="1">[1]</a> 
Andersson, J.A.E., Gillis, J., Horn, G., Rawlings, J.B., and
Diehl, M. (2019). CasADi: a software framework for nonlinear
optimization and optimal control. Mathematical
Programming Computation, 11(1), 1–36.

<a id="2">[2]</a> 
Wachter, A. and Biegler, L.T. (2006). On the implementation
of an interior-point filter line-search algorithm
for large-scale nonlinear programming. Mathematical
Programming, 106(1), 25–57.
