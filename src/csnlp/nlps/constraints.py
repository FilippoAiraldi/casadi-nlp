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
            If `True`, then redundant entries in `lbx` and `ubx` are masked out, e.g.,
            when computing `h_lbx` and `h_ubx`. By default, `True`.
        """
        super().__init__(sym_type)

        self._dual_vars: Dict[str, SymType] = {}
        self._pars: Dict[str, SymType] = {}
        self._cons: Dict[str, SymType] = {}

        self._g, self._lam_g = self._sym_type(0, 1), self._sym_type(0, 1)
        self._h, self._lam_h = self._sym_type(0, 1), self._sym_type(0, 1)
        self._lbx: np.ma.MaskedArray = np.ma.empty(0, fill_value=-np.inf)
        self._ubx: np.ma.MaskedArray = np.ma.empty(0, fill_value=+np.inf)
        self._lam_lbx = self._sym_type(0, 1)
        self._lam_ubx = self._sym_type(0, 1)

        self._remove_redundant_x_bounds = remove_redundant_x_bounds

    @property
    def lbx(self) -> np.ma.MaskedArray:
        """Gets the lower bound constraints of primary variables of the NLP scheme in
        masked vector form."""
        return self._lbx

    @property
    def ubx(self) -> np.ma.MaskedArray:
        """Gets the upper bound constraints of primary variables of the NLP scheme in
        masked vector form."""
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
    def nonmasked_lbx_idx(self) -> Union[slice, npt.NDArray[np.int64]]:
        """Gets the indices of non-masked entries in `lbx` (or the full slice)."""
        return (
            slice(None)
            if np.ma.getmask(self._lbx) is np.ma.nomask
            else np.where(~np.ma.getmaskarray(self._lbx))[0]
        )

    @cached_property
    def nonmasked_ubx_idx(self) -> Union[slice, npt.NDArray[np.int64]]:
        """Gets the indices of non-masked entries in `ubx` (or the full slice)."""
        return (
            slice(None)
            if np.ma.getmask(self._ubx) is np.ma.nomask
            else np.where(~np.ma.getmaskarray(self._ubx))[0]
        )

    @cached_property
    def h_lbx(self) -> SymType:
        """Gets the inequalities due to `lbx`."""
        idx = self.nonmasked_lbx_idx
        return self._lbx.data[idx, None] - self._x[idx, :]

    @cached_property
    def h_ubx(self) -> SymType:
        """Gets the inequalities due to `ubx`."""
        idx = self.nonmasked_ubx_idx
        return self._x[idx, :] - self._ubx.data[idx, None]

    @cached_property
    def lam(self) -> SymType:
        """Gets the dual variables of the NLP scheme in vector form.

        Note
        ----
        The dual variables are vertically concatenated in the following order:
        `lam_g, lam_h, lam_lbx, lam_ubx`.
        """
        return cs.vertcat(self._lam_g, self._lam_h, self._lam_lbx, self._lam_ubx)

    @cached_property
    def primal_dual(self) -> SymType:
        """Gets the collection of primal-dual variables (usually, denoted as `y`)
        ```
                    y = [x^T, lam^T]^T
        ```
        where `x` are the primal variables, and `lam` the dual variables."""
        return cs.vertcat(self._x, self.lam)

    @invalidate_cache(
        nonmasked_lbx_idx, nonmasked_ubx_idx, h_lbx, h_ubx, lam, primal_dual
    )
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
            multipliers. The shape of the multiplier is equal to the number of relevant
            lower bounds (i.e., `!=-np.inf`), so it may differ from the shape of the
            variable itself. This behaviour can be disabled by setting
            `remove_redundant_x_bounds=False`.
        lam_ub : casadi.SX or MX
            Same as above, for upper bound.

        Raises
        ------
        ValueError
            Raises if there is already another variable with the same name; or if any
            element of the lower bound is larger than the corresponding lower bound
            element.
        """
        lb = np.broadcast_to(lb, shape).reshape(-1, order="F")
        ub = np.broadcast_to(ub, shape).reshape(-1, order="F")
        if np.any(lb > ub):
            raise ValueError("Improper variable bounds.")

        var = super().variable(name, shape)

        mlb: np.ma.MaskedArray = np.ma.masked_array(lb, np.ma.nomask)
        mub: np.ma.MaskedArray = np.ma.masked_array(ub, np.ma.nomask)
        if self._remove_redundant_x_bounds:
            mlb.mask = lb == -np.inf
            mub.mask = ub == +np.inf

        self._lbx = np.ma.concatenate((self._lbx, mlb))
        self._ubx = np.ma.concatenate((self._ubx, mub))
        self._lbx.fill_value = -np.inf
        self._ubx.fill_value = +np.inf

        name_lam_lb = f"lam_lb_{name}"
        name_lam_ub = f"lam_ub_{name}"
        lam_lb = self._sym_type.sym(name_lam_lb, (~np.ma.getmaskarray(mlb)).sum())
        lam_ub = self._sym_type.sym(name_lam_ub, (~np.ma.getmaskarray(mub)).sum())
        self._dual_vars[name_lam_lb] = lam_lb
        self._dual_vars[name_lam_ub] = lam_ub
        self._lam_lbx = cs.veccat(self._lam_lbx, lam_lb)
        self._lam_ubx = cs.veccat(self._lam_ubx, lam_ub)
        return var, lam_lb, lam_ub

    @invalidate_cache(lam, primal_dual)
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
            expr = cs.cse(cs.simplify(expr))

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
        lam_c = self._sym_type.sym(name_lam, shape[0] * shape[1])
        self._dual_vars[name_lam] = lam_c

        setattr(self, group, cs.veccat(getattr(self, group), expr))
        setattr(self, lam, cs.veccat(getattr(self, lam), lam_c))
        return (expr, lam_c, slack) if soft else (expr, lam_c)
