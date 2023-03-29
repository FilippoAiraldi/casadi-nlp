from typing import Dict, Literal, Tuple, TypeVar, Union

import casadi as cs
import numpy as np
import numpy.typing as npt

from csnlp.core.cache import cached_property, invalidate_cache
from csnlp.nlps.variables import HasVariables

SymType = TypeVar("SymType", cs.SX, cs.MX)


class HasConstraints(HasVariables[SymType]):
    """Class for creating and storing symbolic constraints for an NLP problem."""

    __slots__ = (
        "_dual_vars",
        "_pars",
        "_cons",
        "_lbx",
        "_ubx",
        "_lam_lbx",
        "_lam_ubx",
        "_g",
        "_lam_g",
        "_h",
        "_lam_h",
        "_remove_redundant_x_bounds",
    )

    def __init__(
        self,
        sym_type: Literal["SX", "MX"] = "SX",
        remove_redundant_x_bounds: bool = True,
    ) -> None:
        """Instantiate the class.

        Parameters
        ----------
        sym_type : "SX" or "MX", optional
            The CasADi symbolic variable type to use in the NLP, by default "SX".
        remove_redundant_x_bounds : bool, optional
            If `True`, then redundant entries in `lbx` and `ubx` are removed when
            properties `h_lbx` and `h_ubx` are called. See these two properties for more
            details. By default, `True`.
        """
        super().__init__(sym_type)

        self._dual_vars: Dict[str, SymType] = {}
        self._pars: Dict[str, SymType] = {}
        self._cons: Dict[str, SymType] = {}

        self._lbx, self._ubx = np.asarray([]), np.asarray([])
        self._lam_lbx, self._lam_ubx = self._sym_type(0, 1), self._sym_type(0, 1)
        self._g, self._lam_g = self._sym_type(0, 1), self._sym_type(0, 1)
        self._h, self._lam_h = self._sym_type(0, 1), self._sym_type(0, 1)

        self._remove_redundant_x_bounds = remove_redundant_x_bounds

    @property
    def lbx(self) -> npt.NDArray[np.double]:
        """Gets the lower bound constraints of primary variables of the NLP scheme in
        vector form."""
        return self._lbx

    @property
    def ubx(self) -> npt.NDArray[np.double]:
        """Gets the upper bound constraints of primary variables of the NLP scheme in
        vector form."""
        return self._ubx

    @property
    def lam_lbx(self) -> SymType:
        """Gets the dual variables of the primary variables lower bound constraints of
        the NLP scheme in vector form."""
        return self._lam_lbx

    @property
    def lam_ubx(self) -> SymType:
        """Gets the dual variables of the primary variables upper bound constraints of
        the NLP scheme in vector form."""
        return self._lam_ubx

    @property
    def g(self) -> SymType:
        """Gets the equality constraint expressions of the NLP scheme in vector form."""
        return self._g

    @property
    def h(self) -> SymType:
        """Gets the inequality constraint expressions of the NLP scheme in vector
        form."""
        return self._h

    @property
    def lam_g(self) -> SymType:
        """Gets the dual variables of the equality constraints of the NLP scheme in
        vector form."""
        return self._lam_g

    @property
    def lam_h(self) -> SymType:
        """Gets the dual variables of the inequality constraints of the NLP scheme in
        vector form."""
        return self._lam_h

    @property
    def ng(self) -> int:
        """Number of equality constraints in the NLP scheme."""
        return self._g.shape[0]

    @property
    def nh(self) -> int:
        """Number of inequality constraints in the NLP scheme."""
        return self._h.shape[0]

    @property
    def dual_variables(self) -> Dict[str, SymType]:
        """Gets the dual variables of the NLP scheme."""
        return self._dual_vars

    @property
    def constraints(self) -> Dict[str, SymType]:
        """Gets the constraints of the NLP scheme."""
        return self._cons

    @cached_property
    def h_lbx(self) -> Tuple[SymType, SymType]:
        """Gets the inequalities due to `lbx` and their multipliers. If
        `remove_redundant_x_bounds=True`, it removes redundant entries, i.e., where
        `lbx == -inf`; otherwise, returns all lower bound constraints."""
        if not self._remove_redundant_x_bounds:
            return self._lbx[:, None] - self._x, self._lam_lbx
        idx = np.where(self._lbx != -np.inf)[0]
        if idx.size == 0:
            return self._sym_type(0, 1), self._sym_type(0, 1)
        return self._lbx[idx, None] - self._x[idx], self._lam_lbx[idx]

    @cached_property
    def h_ubx(self) -> Tuple[SymType, SymType]:
        """Gets the inequalities due to `ubx` and their multipliers. If
        `remove_redundant_x_bounds=True`, it removes redundant entries, i.e., where
        `ubx == +inf`; otherwise, returns all upper bound constraints."""
        if not self._remove_redundant_x_bounds:
            return self._x - self._ubx[:, None], self._lam_ubx
        idx = np.where(self._ubx != np.inf)[0]
        if idx.size == 0:
            return self._sym_type(0, 1), self._sym_type(0, 1)
        return self._x[idx] - self._ubx[idx, None], self._lam_ubx[idx]

    @cached_property
    def lam(self) -> SymType:
        """Gets the dual variables of the NLP scheme in vector form.

        Note
        ----
        The dual variables are vertically concatenated in the following order:
        `lam_g, lam_h, lam_lbx, lam_ubx`.
        """
        return cs.vertcat(self._lam_g, self._lam_h, self.h_lbx[1], self.h_ubx[1])

    @cached_property
    def lam_all(self) -> SymType:
        """Gets all the dual variables of the NLP scheme in vector form, irrespective of
        redundant `lbx` and `ubx` multipliers. If `remove_redundant_x_bounds=True`, then
        this property is equivalent to the `lam` property.

        Note
        ----
        The dual variables are vertically concatenated in the following order:
        `lam_g, lam_h, lam_lbx, lam_ubx`.
        """
        return cs.vertcat(self._lam_g, self._lam_h, self._lam_lbx, self._lam_ubx)

    def primal_dual_vars(self, all: bool = False) -> SymType:
        """Gets the collection of primal-dual variables (usually, denoted as `y`)
        ```
                    y = [x^T, lam^T]^T
        ```
        where `x` are the primal variables, and `lam` the dual variables.

        Parameters
        ----------
        all : bool, optional
            If `True`, all dual variables are included, even the multipliers connected
            to redundant `lbx` or `ubx`. Otherwise, the redundant ones are removed. By
            default, `False`.

        Returns
        ------
        casadi SX or MX
            The collection of primal-dual variables `y`.
        """
        return cs.vertcat(self._x, self.lam_all if all else self.lam)

    @invalidate_cache(h_lbx, h_ubx, lam, lam_all)
    def variable(
        self,
        name: str,
        shape: Tuple[int, int] = (1, 1),
        lb: Union[npt.ArrayLike, cs.DM] = -np.inf,
        ub: Union[npt.ArrayLike, cs.DM] = +np.inf,
    ) -> Tuple[SymType, SymType, SymType]:
        """
        Adds a variable to the NLP problem.

        Parameters
        ----------
        name : str
            Name of the new variable. Must not be already in use.
        shape : tuple[int, int], optional
            Shape of the new variable. By default, a scalar.
        lb, ub: array_like, optional
            Lower and upper bounds of the new variable. By default, unbounded. If
            provided, their dimension must be broadcastable.

        Returns
        -------
        var : casadi.SX or MX
            The symbol of the new variable.
        lam_lb : casadi.SX or MX
            The symbol corresponding to the new variable lower bound constraint's
            multipliers.
        lam_ub : casadi.SX or MX
            Same as above, for upper bound.

        Raises
        ------
        ValueError
            Raises if there is already another variable with the same name; or if any
            element of the lower bound is larger than the corresponding lower bound
            element.
        """
        var = super().variable(name, shape)

        lb, ub = np.broadcast_to(lb, shape), np.broadcast_to(ub, shape)
        if np.all(lb > ub):
            raise ValueError("Improper variable bounds.")
        self._lbx = np.concatenate((self._lbx, lb.flatten("F")))
        self._ubx = np.concatenate((self._ubx, ub.flatten("F")))

        name_lam = f"lam_lb_{name}"
        lam_lb = self._sym_type.sym(name_lam, *shape)
        self._dual_vars[name_lam] = lam_lb
        self._lam_lbx = cs.veccat(self._lam_lbx, lam_lb)
        name_lam = f"lam_ub_{name}"
        lam_ub = self._sym_type.sym(name_lam, *shape)
        self._dual_vars[name_lam] = lam_ub
        self._lam_ubx = cs.veccat(self._lam_ubx, lam_ub)
        return var, lam_lb, lam_ub

    @invalidate_cache(lam, lam_all)
    def constraint(
        self,
        name: str,
        lhs: Union[SymType, np.ndarray, cs.DM],
        op: Literal["==", ">=", "<="],
        rhs: Union[SymType, np.ndarray, cs.DM],
        soft: bool = False,
        simplify: bool = True,
    ) -> Tuple[SymType, ...]:
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
            If `True`, then a slack variable with appropriate size is added to the NLP
            to make the inequality constraint soft, and returned. This slack is
            automatically lower-bounded by 0, but remember to manually penalize its
            magnitude in the objective. Slacks are not supported for equality
            constraints. Defaults to `False`.
        simplify : bool, optional
            Optionally simplies the constraint expression, but can be disabled.

        Returns
        -------
        expr : casadi.SX or MX
            The constraint expression in canonical form, i.e., `g(x,u) = 0` or
            `h(x,u) <= 0`.
        lam : casadi.SX or MX
            The symbol corresponding to the constraint's multipliers.
        slack : casadi.SX or MX, optional
            The slack variable in case of `soft=True`; otherwise, only a 2-tuple is
            returned.

        Raises
        ------
        ValueError
            Raises if there is already another constraint with the same name; or if the
            operator is not recognized.
        NotImplementedError
            Raises if a soft equality constraint is requested.
        TypeError
            Raises if the constraint is not symbolic.
        """
        if name in self._cons:
            raise ValueError(f"Constraint name '{name}' already exists.")
        expr = lhs - rhs
        if not isinstance(expr, (cs.SX, cs.MX)):
            raise TypeError("Constraint must be symbolic.")
        if simplify:
            expr = cs.simplify(expr)

        shape = expr.shape
        if op == "==":
            is_eq = True
            if soft:
                raise NotImplementedError(
                    "Soft equality constraints are not currently supported."
                )
        elif op == "<=":
            is_eq = False
        elif op == ">=":
            is_eq = False
            expr = -expr
        else:
            raise ValueError(f"Unrecognized operator {op}.")

        if soft:
            slack = self.variable(f"slack_{name}", expr.shape, lb=0)[0]
            expr -= slack

        self._cons[name] = expr
        group, lam = ("_g", "_lam_g") if is_eq else ("_h", "_lam_h")
        name_lam = f"{lam[1:]}_{name}"
        lam_c = self._sym_type.sym(name_lam, *shape)
        self._dual_vars[name_lam] = lam_c

        setattr(self, group, cs.veccat(getattr(self, group), expr))
        setattr(self, lam, cs.veccat(getattr(self, lam), lam_c))
        return (expr, lam_c, slack) if soft else (expr, lam_c)
