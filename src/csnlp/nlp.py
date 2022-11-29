import warnings
from functools import partial
from itertools import count
from typing import Any, Dict, Literal, Optional, Sequence, Tuple, Type, Union

import casadi as cs
import numpy as npy

from csnlp.debug import NlpDebug
from csnlp.solutions import DMStruct, Solution, subsevalf
from csnlp.util.data import dict2struct, is_casadi_object, struct_symSX
from csnlp.util.funcs import cache_clearer, cached_property, np_random
from csnlp.util.io import is_pickleable

"""This tuple dictates the order for operations related to dual variables."""
_DUAL_VARIABLES_ORDER = ("g", "h", "h_lbx", "h_ubx")


class Nlp:
    """
    The generic NLP class is a controller that solves a (possibly, nonlinear)
    optimization problem to yield a (possibly, sub-) optimal solution.

    This is a generic class in the sense that it does not solve a particular
    problem, but only offers the generic methods to build one (e.g., variables,
    constraints, objective, solver).
    """

    __ids = count(0)

    def __init__(
        self,
        sym_type: Literal["SX", "MX"] = "SX",
        remove_redundant_x_bounds: bool = True,
        name: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> None:
        """Creates an NLP instance with a given name.

        Parameters
        ----------
        sym_type : 'SX' or 'MX', optional
            The CasADi symbolic variable type to use in the NLP, by default
            'SX'.
        remove_redundant_x_bounds : bool, optional
            If `True`, then redundant entries in `lbx` and `ubx` are removed
            when properties `h_lbx` and `h_ubx` are called. See these two
            properties for more details. By default, `True`.
        name : str, optional
            Name of the NLP scheme. If `None`, it is automatically assigned.
        seed : int, optional
            Random number generator seed.
        """
        self.id = next(self.__ids)
        self.name = f"{self.__class__.__name__}{self.id}" if name is None else name
        self._csXX: Union[Type[cs.SX], Type[cs.MX]] = getattr(cs, sym_type)

        self._vars: Dict[str, Union[cs.SX, cs.MX]] = {}
        self._dual_vars: Dict[str, Union[cs.SX, cs.MX]] = {}
        self._pars: Dict[str, Union[cs.SX, cs.MX]] = {}
        self._cons: Dict[str, Union[cs.SX, cs.MX]] = {}

        self._f: Optional[Union[cs.SX, cs.MX]] = None
        self._p = self._csXX()
        self._x = self._csXX()
        self._lbx, self._ubx = npy.array([]), npy.array([])
        self._lam_lbx, self._lam_ubx = self._csXX(), self._csXX()
        self._g, self._lam_g = self._csXX(), self._csXX()
        self._h, self._lam_h = self._csXX(), self._csXX()
        self._lbg, self._lbh = npy.array([]), npy.array([])

        self._solver: Optional[cs.Function] = None
        self._solver_opts: Dict[str, Any] = {}
        self._failures = 0
        self._debug = NlpDebug()
        self._seed = seed
        self._np_random: Optional[npy.random.Generator] = None
        self._remove_redundant_x_bounds = remove_redundant_x_bounds

    @property
    def unwrapped(self) -> "Nlp":
        """Returns the original NLP of the wrapper."""
        return self

    def is_wrapped(self, *args, **kwargs) -> bool:
        """Gets whether the NLP instance is wrapped or not by the given wrapper type."""
        return False

    @property
    def np_random(self) -> npy.random.Generator:
        """Returns the nlp's random engine that, if not set, will be
        initialised with the nlp's seed."""
        if self._np_random is None:
            self._np_random, _ = np_random(self._seed)
        return self._np_random

    @property
    def sym_type(self) -> Union[Type[cs.SX], Type[cs.MX]]:
        """Gets the CasADi symbolic type used in this NLP scheme."""
        return self._csXX

    @property
    def f(self) -> Union[None, cs.SX, cs.MX]:
        """Gets the objective of the NLP scheme, which is `None` if not set
        previously set via the `minimize` method."""
        return self._f

    @property
    def p(self) -> Union[cs.SX, cs.MX]:
        """Gets the parameters of the NLP scheme."""
        return self._p

    @property
    def x(self) -> Union[cs.SX, cs.MX]:
        """Gets the primary variables of the NLP scheme in vector form."""
        return self._x

    @property
    def lbx(self) -> npy.ndarray:
        """Gets the lower bound constraints of primary variables of the NLP
        scheme in vector form."""
        return self._lbx

    @property
    def ubx(self) -> npy.ndarray:
        """Gets the upper bound constraints of primary variables of the NLP
        scheme in vector form."""
        return self._ubx

    @property
    def lam_lbx(self) -> Union[cs.SX, cs.MX]:
        """Gets the dual variables of the primary variables lower bound
        constraints of the NLP scheme in vector form."""
        return self._lam_lbx

    @property
    def lam_ubx(self) -> Union[cs.SX, cs.MX]:
        """Gets the dual variables of the primary variables upper bound
        constraints of the NLP scheme in vector form."""
        return self._lam_ubx

    @property
    def g(self) -> Union[cs.SX, cs.MX]:
        """Gets the equality constraint expressions of the NLP scheme in vector
        form."""
        return self._g

    @property
    def h(self) -> Union[cs.SX, cs.MX]:
        """Gets the inequality constraint expressions of the NLP scheme in
        vector form."""
        return self._h

    @property
    def lam_g(self) -> Union[cs.SX, cs.MX]:
        """Gets the dual variables of the equality constraints of the NLP
        scheme in vector form."""
        return self._lam_g

    @property
    def lam_h(self) -> Union[cs.SX, cs.MX]:
        """Gets the dual variables of the inequality constraints of the NLP
        scheme in vector form."""
        return self._lam_h

    @property
    def solver(self) -> Optional[cs.Function]:
        """Gets the NLP optimization solver. Can be `None`, if the solver is
        not set with method `init_solver`."""
        return self._solver

    @property
    def solver_opts(self) -> Dict[str, Any]:
        """Gets the NLP optimization solver options. The dict is empty, if the
        solver options are not set with method `init_solver`."""
        return self._solver_opts

    @property
    def failures(self) -> int:
        """Gets the cumulative number of failures of the NLP solver."""
        return self._failures

    @property
    def debug(self) -> NlpDebug:
        """Gets debug information on the NLP scheme."""
        return self._debug

    @property
    def nx(self) -> int:
        """Number of variables in the NLP scheme."""
        return self._x.shape[0]

    @property
    def np(self) -> int:
        """Number of parameters in the NLP scheme."""
        return self._p.shape[0]

    @property
    def ng(self) -> int:
        """Number of equality constraints in the NLP scheme."""
        return self._g.shape[0]

    @property
    def nh(self) -> int:
        """Number of inequality constraints in the NLP scheme."""
        return self._h.shape[0]

    @cached_property
    def parameters(self) -> Union[struct_symSX, Dict[str, cs.MX]]:
        """Gets the parameters of the NLP scheme."""
        return dict2struct(self._pars)

    @cached_property
    def variables(self) -> Union[struct_symSX, Dict[str, cs.MX]]:
        """Gets the primal variables of the NLP scheme."""
        return dict2struct(self._vars)

    @cached_property
    def dual_variables(self) -> Union[struct_symSX, Dict[str, cs.MX]]:
        """Gets the dual variables of the NLP scheme."""
        return dict2struct(self._dual_vars)

    @cached_property
    def constraints(self) -> Union[struct_symSX, Dict[str, cs.MX]]:
        """Gets the constraints of the NLP scheme."""
        return dict2struct(self._cons, entry_type="expr")

    @cached_property
    def lam(self) -> Union[cs.SX, cs.MX]:
        """Gets the dual variables of the NLP scheme in vector form.

        Note: The order of the dual variables can be adjusted via
        `_DUAL_VARIABLES_ORDER`."""
        items = {
            "g": self._lam_g,
            "h": self._lam_h,
            "h_lbx": self.h_lbx[1],
            "h_ubx": self.h_ubx[1],
        }
        dual = cs.vertcat(*(items.pop(v) for v in _DUAL_VARIABLES_ORDER))
        assert not items, "Internal error. _DUAL_VARIABLES_ORDER modified."
        return dual

    @cached_property
    def lam_all(self) -> Union[cs.SX, cs.MX]:
        """Gets all the dual variables of the NLP scheme in vector form,
        irrespective of redundant `lbx` and `ubx` multipliers. If
        `remove_redundant_x_bounds`, then this property is equivalent to
        the `lam` property.

        Note: The order of the dual variables can be adjusted via
        `_DUAL_VARIABLES_ORDER`."""
        items = {
            "g": self._lam_g,
            "h": self._lam_h,
            "h_lbx": self._lam_lbx,
            "h_ubx": self._lam_ubx,
        }
        dual = cs.vertcat(*(items.pop(v) for v in _DUAL_VARIABLES_ORDER))
        assert not items, "Internal error. _DUAL_VARIABLES_ORDER modified."
        return dual

    @cached_property
    def h_lbx(self) -> Union[Tuple[cs.SX, cs.SX], Tuple[cs.MX, cs.MX]]:
        """Gets the inequalities due to `lbx` and their multipliers. If
        `simplify_x_bounds=True`, it removes redundant entries, i.e., where
        `lbx == -inf`; otherwise, returns all lower bound constraints."""
        if not self._remove_redundant_x_bounds:
            return self._lbx[:, None] - self._x, self._lam_lbx
        idx = npy.where(self._lbx != -npy.inf)[0]
        if idx.size == 0:
            return self._csXX(), self._csXX()
        return self._lbx[idx, None] - self._x[idx], self._lam_lbx[idx]

    @cached_property
    def h_ubx(self) -> Union[Tuple[cs.SX, cs.SX], Tuple[cs.MX, cs.MX]]:
        """Gets the inequalities due to `ubx` and their multipliers. If
        `simplify_x_bounds=True`, it removes redundant entries, i.e., where
        `ubx == +inf`; otherwise, returns all upper bound constraints."""
        if not self._remove_redundant_x_bounds:
            return self._x - self._ubx[:, None], self._lam_ubx
        idx = npy.where(self._ubx != npy.inf)[0]
        if idx.size == 0:
            return self._csXX(), self._csXX()
        return self._x[idx] - self._ubx[idx, None], self._lam_ubx[idx]

    def primal_dual_vars(self, all: bool = False) -> Union[cs.SX, cs.MX]:
        """Gets the collection of primal-dual variables (usually, denoted as
        `y`)
        ```
                    y = [x^T, lam^T]^T
        ```
        where `x` are the primal variables, and `lam` the dual variables.

        Parameters
        ----------
        all : bool, optional
            If `True`, all dual variables are included, even the multipliers
            connected to redundant `lbx` or `ubx`. Otherwise, the redundant
            ones are removed. By default, `False`.

        Returns
        ------
        Union[cs.SX, cs.MX]
            The collection of primal-dual variables `y`.

        Note
        ----
        The order of the dual variables can be adjusted via
        `_DUAL_VARIABLES_ORDER`.
        """
        return cs.vertcat(self._x, self.lam_all if all else self.lam)

    @cache_clearer(parameters)
    def parameter(
        self, name: str, shape: Tuple[int, int] = (1, 1)
    ) -> Union[cs.SX, cs.MX]:
        """Adds a parameter to the NLP scheme.

        Parameters
        ----------
        name : str
            Name of the new parameter. Must not be already in use.
        shape : tuple[int, int], optional
            Shape of the new parameter. By default, a scalar.

        Returns
        -------
        casadi.SX or MX
            The symbol for the new parameter.

        Raises
        ------
        ValueError
            Raises if there is already another parameter with the same name.
        """
        if name in self._pars:
            raise ValueError(f"Parameter name '{name}' already exists.")
        par = self._csXX.sym(name, *shape)
        self._pars[name] = par
        self._p = cs.vertcat(self._p, cs.vec(par))
        self._debug.register("p", name, shape)

        if self._solver is not None:
            self.init_solver(self._solver_opts)  # resets solver
        return par

    @cache_clearer(variables, dual_variables, h_lbx, h_ubx, lam, lam_all)
    def variable(
        self,
        name: str,
        shape: Tuple[int, int] = (1, 1),
        lb: Union[npy.ndarray, cs.DM] = -npy.inf,
        ub: Union[npy.ndarray, cs.DM] = +npy.inf,
    ) -> Union[Tuple[cs.SX, cs.SX, cs.SX], Tuple[cs.MX, cs.MX, cs.MX]]:
        """
        Adds a variable to the NLP problem.

        Parameters
        ----------
        name : str
            Name of the new variable. Must not be already in use.
        shape : tuple[int, int], optional
            Shape of the new variable. By default, a scalar.
        lb, ub: array_like, optional
            Lower and upper bounds of the new variable. By default, unbounded.
            If provided, their dimension must be broadcastable.

        Returns
        -------
        var : casadi.SX
            The symbol of the new variable.
        lam_lb : casadi.SX
            The symbol corresponding to the new variable lower bound
            constraint's multipliers.
        lam_ub : casadi.SX
            Same as above, for upper bound.

        Raises
        ------
        ValueError
            Raises if there is already another variable with the same name; or
            if any element of the lower bound is larger than the corresponding
            lower bound element.
        """
        if name in self._vars:
            raise ValueError(f"Variable name '{name}' already exists.")
        lb, ub = npy.broadcast_to(lb, shape), npy.broadcast_to(ub, shape)
        if npy.all(lb > ub):
            raise ValueError("Improper variable bounds.")

        var = self._csXX.sym(name, *shape)
        self._vars[name] = var
        self._x = cs.vertcat(self._x, cs.vec(var))
        self._lbx = npy.concatenate((self._lbx, lb.flatten("F")))
        self._ubx = npy.concatenate((self._ubx, ub.flatten("F")))
        self._debug.register("x", name, shape)

        name_lam = f"lam_lb_{name}"
        lam_lb = self._csXX.sym(name_lam, *shape)
        self._dual_vars[name_lam] = lam_lb
        self._lam_lbx = cs.vertcat(self._lam_lbx, cs.vec(lam_lb))
        name_lam = f"lam_ub_{name}"
        lam_ub = self._csXX.sym(name_lam, *shape)
        self._dual_vars[name_lam] = lam_ub
        self._lam_ubx = cs.vertcat(self._lam_ubx, cs.vec(lam_ub))

        if self._solver is not None:
            self.init_solver(self._solver_opts)  # resets solver
        return var, lam_lb, lam_ub

    @cache_clearer(constraints, dual_variables, lam, lam_all)
    def constraint(
        self,
        name: str,
        lhs: Union[npy.ndarray, cs.DM, cs.SX, cs.MX],
        op: Literal["==", ">=", "<="],
        rhs: Union[npy.ndarray, cs.DM, cs.SX, cs.MX],
        soft: bool = False,
        simplify: bool = True,
    ) -> Union[
        Tuple[cs.SX, cs.SX],
        Tuple[cs.MX, cs.MX],
        Tuple[cs.SX, cs.SX, cs.SX],
        Tuple[cs.MX, cs.MX, cs.MX],
    ]:
        """Adds a constraint to the NLP problem, e.g., `lhs <= rhs`.

        Parameters
        ----------
        name : str
            Name of the new constraint. Must not be already in use.
        lhs : casadi.SX, MX, DM or numerical
            Symbolic expression of the left-hand term of the constraint.
        op: str, {'==' '>=', '<='}
            Operator relating the two terms.
        rhs : casadi.SX, MX, DM or numerical
            Symbolic expression of the right-hand term of the constraint.
        soft : bool, optional
            If `True`, then a slack variable with appropriate size is added to
            the NLP to make the inequality constraint soft, and returned.
            This slack is automatically lower-bounded by 0, but remember to
            penalize its magnitude in the objective. Slacks are not supported
            for equality constraints. Defaults to `False`.
        simplify : bool, optional
            Optionally simplies the constraint expression, but can be disabled.

        Returns
        -------
        expr : casadi.SX, MX
            The constraint expression in canonical form, i.e., `g(x,u) = 0` or
            `h(x,u) <= 0`.
        lam : casadi.SX, MX
            The symbol corresponding to the constraint's multipliers.
        slack : casadi.SX, MX, optional
            The slack variable in case of `soft=True`; otherwise, only a
            2-tuple is returned.

        Raises
        ------
        ValueError
            Raises if there is already another constraint with the same name;
            or if the operator is not recognized.
        NotImplementedError
            Raises if a soft equality constraint is requested.
        TypeError
            Raises if the constraint is not symbolic.
        """
        if name in self._cons:
            raise ValueError(f"Constraint name '{name}' already exists.")
        expr: Union[cs.SX, cs.MX] = lhs - rhs
        if not isinstance(expr, (cs.SX, cs.MX)):
            raise TypeError("Constraint must be symbolic.")
        if simplify:
            expr = cs.simplify(expr)

        shape = expr.shape
        if op == "==":
            is_eq = True
            lb = npy.zeros(npy.prod(shape))
            if soft:
                raise NotImplementedError(
                    "Soft equality constraints are not currently supported."
                )
        elif op == "<=":
            is_eq = False
            lb = npy.full(npy.prod(shape), -npy.inf)
        elif op == ">=":
            is_eq = False
            expr = -expr
            lb = npy.full(npy.prod(shape), -npy.inf)
        else:
            raise ValueError(f"Unrecognized operator {op}.")

        if soft:
            slack = self.variable(f"slack_{name}", expr.shape, lb=0)[0]
            expr -= slack

        self._cons[name] = expr
        group, con, lam = (
            ("_g", "_lbg", "_lam_g") if is_eq else ("_h", "_lbh", "_lam_h")
        )
        self._debug.register(group[1:], name, shape)
        name_lam = f"{lam[1:]}_{name}"
        lam_c = self._csXX.sym(name_lam, *shape)
        self._dual_vars[name_lam] = lam_c

        setattr(self, group, cs.vertcat(getattr(self, group), cs.vec(expr)))
        setattr(self, con, npy.concatenate((getattr(self, con), lb)))
        setattr(self, lam, cs.vertcat(getattr(self, lam), cs.vec(lam_c)))

        if self._solver is not None:
            self.init_solver(self._solver_opts)  # resets solver
        return (expr, lam_c, slack) if soft else (expr, lam_c)

    def minimize(self, objective: Union[cs.SX, cs.MX]) -> None:
        """Sets the objective function to be minimized.

        Parameters
        ----------
        objective : Union[cs.SX, cs.MX]
            A Symbolic variable dependent only on the NLP variables and
            parameters that needs to be minimized.

        Raises
        ------
        ValueError
            Raises if the objective is not scalar.
        """
        if objective.shape != (1, 1):
            raise ValueError("Objective must be scalar.")
        self._f = objective
        if self._solver is not None:
            self.init_solver(self._solver_opts)  # resets solver

    def init_solver(self, opts: Optional[Dict[str, Any]] = None) -> None:
        """Initializes the IPOPT solver for this NLP with the given options.

        Parameters
        ----------
        opts : Dict[str, Any], optional
            Options to be passed to the CasADi interface to the solver.

        Raises
        ------
        RuntimeError
            Raises if the objective has not yet been specified with `minimize`.
        """
        if self._f is None:
            raise RuntimeError("NLP objective not set.")
        opts = {} if opts is None else opts.copy()
        con = cs.vertcat(self._g, self._h)
        nlp = {"x": self._x, "p": self._p, "g": con, "f": self._f}
        self._solver = cs.nlpsol(f"nlpsol_{self.name}", "ipopt", nlp, opts)
        self._solver_opts = opts

    def solve(
        self,
        pars: Union[None, DMStruct, Dict[str, npy.ndarray]] = None,
        vals0: Union[None, DMStruct, Dict[str, npy.ndarray]] = None,
    ) -> Solution:
        """Solves the NLP optimization problem.

        Parameters
        ----------
        pars : DMStruct, dict[str, array_like], optional
            Dictionary or structure containing, for each parameter in the NLP
            scheme, the corresponding numerical value. Can be `None` if no
            parameters are present.
        vals0 : DMStruct, dict[str, array_like], optional
            Dictionary or structure containing, for each variable in the NLP
            scheme, the corresponding initial guess. By default, initial
            guesses are not passed to the solver.

        Returns
        -------
        sol : Solution
            A solution object containing all the information.

        Raises
        ------
        RuntimeError
            Raises if the solver is un-initialized (see `init_solver`); or if
            not all the parameters are not provided with a numerical value.
        """
        if pars is None:
            pars = {}
        if self._solver is None:
            raise RuntimeError("Solver uninitialized.")
        parsdiff = self._pars.keys() - pars.keys()
        if len(parsdiff) != 0:
            raise RuntimeError(
                "Trying to solve the NLP with unspecified parameters: "
                + ", ".join(parsdiff)
                + "."
            )

        p = subsevalf(self._p, self._pars, pars)
        kwargs = {
            "p": p,
            "lbx": self._lbx,
            "ubx": self._ubx,
            "lbg": npy.concatenate((self._lbg, self._lbh)),
            "ubg": 0,
        }
        if vals0 is not None:
            kwargs["x0"] = subsevalf(self._x, self._vars, vals0)
        sol: Dict[str, cs.DM] = self._solver(**kwargs)

        # extract lam_x, lam_g and lam_h
        lam_lbx = -cs.fmin(sol["lam_x"], 0)
        lam_ubx = cs.fmax(sol["lam_x"], 0)
        lam_g = sol["lam_g"][: self.ng, :]
        lam_h = sol["lam_g"][self.ng :, :]

        vars_ = self.variables
        if self._csXX is cs.SX:
            vals = vars_(sol["x"])
        else:
            vals = dict2struct(
                {name: subsevalf(var, self._x, sol["x"]) for name, var in vars_.items()}
            )
        old = cs.vertcat(
            self._p, self._x, self._lam_g, self._lam_h, self._lam_lbx, self._lam_ubx
        )
        new = cs.vertcat(p, sol["x"], lam_g, lam_h, lam_lbx, lam_ubx)
        get_value = partial(subsevalf, old=old, new=new)
        solution = Solution(
            f=float(sol["f"]),
            vars=vars_,
            vals=vals,
            stats=self._solver.stats().copy(),
            _get_value=get_value,
        )
        self._failures += int(not solution.success)
        return solution

    def to_function(
        self,
        name: str,
        ins: Union[Sequence[cs.SX], Sequence[cs.MX]],
        outs: Union[Sequence[cs.SX], Sequence[cs.MX]],
        name_in: Optional[Sequence[str]] = None,
        name_out: Optional[Sequence[str]] = None,
        opts: Optional[Dict[Any, Any]] = None,
    ) -> cs.Function:
        """Converts the optimization problem to an MX symbolic function. If the
        NLP is modelled in SX, the function will still be converted in MX since
        the IPOPT interface cannot expand SX for now.

        Parameters
        ----------
        name : str
            Name of the function.
        ins : Sequence of cs.SX or MX
            Input variables of the function. These must be expressions
            providing the parameters of the NLP and the initial conditions of
            the primal variables `x`.
        outs : Sequence of cs.SX or MX
            Output variables of the function. These must be expressions
            depending on the primal variable `x`, parameters `p`, and dual
            variables `lam_g`, `lam_h`, `lam_lbx`, `lam_ubx` of the NLP.
        name_in : Sequence of str, optional
            Name of the inputs, by default None.
        name_out : Sequence of str, optional
            Name of the outpus, by default None.
        opts : Dict[Any, Any], optional
            Options to be passed to `casadi.Function`, by default None.

        Returns
        -------
        casadi.Function
            The NLP solver as a `casadi.Function`.

        Raises
        ------
        RuntimeError
            Raises if the solver is uninitialized.
        ValueError
            Raises if the input or output expressions have free variables that
            are not provided or cannot be computed by the solver.
        """
        if self._csXX is cs.SX:
            warnings.warn(
                "The IPOPT interface does not support SX expansion, "
                "so the function must be wrapped in MX.",
                RuntimeWarning,
            )
        S = self._solver
        if S is None:
            raise RuntimeError("Solver not yet initialized.")

        # converts inputs/outputs to/from variables and parameters
        n_outs = len(outs)
        Fin = cs.Function("Fin", ins, [self._x, self._p])
        Fout = cs.Function(
            "Fout",
            [self._p, self._x, self._lam_g, self._lam_h, self._lam_lbx, self._lam_ubx],
            outs,
        )
        if Fin.has_free():
            raise ValueError(
                "Input expressions do not provide values for: "
                f'{", ".join(Fin.get_free())}.'
            )
        if Fout.has_free():
            raise ValueError(
                "Output solver cannot provide values for: "
                f'{", ".join(Fout.get_free())}.'
            )

        # call the solver
        if self._csXX is cs.SX:
            Fin = Fin.wrap()
            Fout = Fout.wrap()
            ins = [Fin.mx_in(i) for i in range(len(ins))]
            outs = [Fout.mx_out(i) for i in range(len(outs))]
        x0, p = Fin(*ins)
        sol = S(
            x0=x0,
            p=p,
            lbx=self._lbx,
            ubx=self._ubx,
            lbg=npy.concatenate((self._lbg, self._lbh)),
            ubg=0,
            lam_x0=0,
            lam_g0=0,
        )
        x = sol["x"]
        lam_g = sol["lam_g"][: self.ng, :]
        lam_h = sol["lam_g"][self.ng :, :]
        lam_lbx = -cs.fmin(sol["lam_x"], 0)
        lam_ubx = cs.fmax(sol["lam_x"], 0)
        Fsol = cs.Function("Fsol", ins, [x, lam_g, lam_h, lam_lbx, lam_ubx])

        # build final function
        final_outs = Fout(p, *Fsol(*ins))
        if n_outs == 1:
            final_outs = [final_outs]
        args = [name, ins, final_outs]
        if name_in is not None and name_out is not None:
            args.extend((name_in, name_out))
        if opts is not None:
            args.append(opts)
        return cs.Function(*args)

    def __str__(self) -> str:
        """Returns the NLP name and a short description."""
        msg = "not initialized" if self._solver is None else "initialized"
        C = len(self._cons)
        return (
            f"{type(self).__name__} {{\n"
            f"  name: {self.name}\n"
            f"  #variables: {len(self._vars)} (nx={self.nx})\n"
            f"  #parameters: {len(self._pars)} (np={self.np})\n"
            f"  #constraints: {C} (ng={self.ng}, nh={self.nh})\n"
            f"  CasADi solver {msg}.\n}}"
        )

    def __repr__(self) -> str:
        """Returns the string representation of the NLP instance."""
        return f"{type(self).__name__}: {self.name}"

    def __getstate__(self) -> Dict[str, Any]:
        """Returns the instance's state to be pickled."""
        warnings.warn(
            f"to pickle {self.__class__.__name__} all references to CasADi and "
            "unpickleable objects are removed.",
            RuntimeWarning,
        )
        state = self.__dict__.copy()
        for attr, val in self.__dict__.items():
            if is_casadi_object(val) or not is_pickleable(val):
                state.pop(attr)
        return state
