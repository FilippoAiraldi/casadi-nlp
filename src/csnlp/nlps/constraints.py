from functools import cached_property
from typing import Literal, TypeVar, Union

import casadi as cs
import numpy as np
import numpy.typing as npt

from ..core.cache import invalidate_cache
from .variables import HasVariables

SymType = TypeVar("SymType", cs.SX, cs.MX)


class HasConstraints(HasVariables[SymType]):
    r"""Class for the creation and storage symbolic constraints for an NLP problem. It
    builds on top of :class:`HasVariables`, which handles both parameters and variables.

    Constraints are stored and managed in the canonical way. Equality constraints are
    in the form :math:`g(x,p) = 0` or :math:`G(x,p) = 0`, whereas inequality constraints
    are in the form :math:`h(x,p) \le 0` or :math:`H(x,p) \le 0`. Separated from the
    latter are the lower and upper bounds of the primary variables, which are also
    inequalities, i.e., :math:`lbx - x \le 0` and :math:`x - ubx \le 0`, but are passed
    differently to the CasADi solver interface. Moreover, the class is equipped with a
    mechanism to automatically remove lower and upper bounds that are redundant, i.e.,
    when the lower bound is :math:`-\infty` and the upper bound is :math:`+\infty`,
    which often create numerical issues if passed to the solver as is.

    Parameters
    ----------
    sym_type : {"SX", "MX"}, optional
        The CasADi symbolic variable type to use in the NLP, by default ``"SX"``.
    remove_redundant_x_bounds : bool, optional
        If ``True``, then redundant entries in :meth:`lbx` and :meth:`ubx` are removed
        when properties :meth:`h_lbx` and :meth:`h_ubx` are called. See these two
        properties for more details. By default, ``True``.
    """

    def __init__(
        self,
        sym_type: Literal["SX", "MX"] = "SX",
        remove_redundant_x_bounds: bool = True,
    ) -> None:
        super().__init__(sym_type)

        self._dual_vars: dict[str, SymType] = {}
        self._pars: dict[str, SymType] = {}
        self._cons: dict[str, SymType] = {}

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
    def dual_variables(self) -> dict[str, SymType]:
        """Gets the dual variables of the NLP scheme."""
        return self._dual_vars

    @property
    def constraints(self) -> dict[str, SymType]:
        """Gets the constraints of the NLP scheme."""
        return self._cons

    @cached_property
    def nonmasked_lbx_idx(self) -> Union[slice, npt.NDArray[np.int64]]:
        """Gets the indices of non-masked entries in :meth:`lbx` (or the full slice)."""
        return (
            slice(None)
            if np.ma.getmask(self._lbx) is np.ma.nomask
            else np.where(~np.ma.getmaskarray(self._lbx))[0]
        )

    @cached_property
    def nonmasked_ubx_idx(self) -> Union[slice, npt.NDArray[np.int64]]:
        """Gets the indices of non-masked entries in :meth:`ubx` (or the full slice)."""
        return (
            slice(None)
            if np.ma.getmask(self._ubx) is np.ma.nomask
            else np.where(~np.ma.getmaskarray(self._ubx))[0]
        )

    @cached_property
    def h_lbx(self) -> SymType:
        """Gets the inequalities cor to :meth:`lbx`, i.e., :math:`lbx - x`."""
        idx = self.nonmasked_lbx_idx
        return self._lbx.data[idx, None] - self._x[idx, :]

    @cached_property
    def h_ubx(self) -> SymType:
        """Gets the inequalities due to :meth:`ubx`, i.e., :math:`x - ubx`."""
        idx = self.nonmasked_ubx_idx
        return self._x[idx, :] - self._ubx.data[idx, None]

    @cached_property
    def lam(self) -> SymType:
        """Gets the dual variables of the NLP scheme in vector form.

        Notes
        -----
        The dual variables are vertically concatenated in the following order:
        :meth:`lam_g`, :meth:`lam_h`, :meth:`lam_lbx`, :meth:`lam_ubx`.
        """
        return cs.vertcat(self._lam_g, self._lam_h, self._lam_lbx, self._lam_ubx)

    @cached_property
    def primal_dual(self) -> SymType:
        r"""Gets the collection of primal-dual variables (usually, denoted as ``y``)

        .. math:: y = \begin{bmatrix} x \\ \lambda \end{bmatrix},

        where :math:`x` are the primal variables, and :math:`\lambda` the dual
        variables (see :meth:`x` and :meth:`lam`, respectively)."""
        return cs.vertcat(self._x, self.lam)

    @invalidate_cache(
        nonmasked_lbx_idx, nonmasked_ubx_idx, h_lbx, h_ubx, lam, primal_dual
    )
    def variable(
        self,
        name: str,
        shape: tuple[int, int] = (1, 1),
        discrete: bool = False,
        lb: Union[npt.ArrayLike, cs.DM] = -np.inf,
        ub: Union[npt.ArrayLike, cs.DM] = +np.inf,
    ) -> tuple[SymType, SymType, SymType]:
        r"""Adds a variable to the NLP problem.

        Parameters
        ----------
        name : str
            Name of the new variable. Must not be already in use.
        shape : tuple of 2 ints, optional
            Shape of the new variable. By default, a scalar.
        discrete : bool, optional
            Flag indicating if the variable is discrete. Defaults to ``False``.
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
            lower bounds (i.e., :math:`\neq \pm \infty`), so it may differ from the
            shape of the variable itself. This behaviour can be disabled by setting
            ``remove_redundant_x_bounds=False``.
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

        var = super().variable(name, shape, discrete)

        mlb: np.ma.MaskedArray = np.ma.masked_array(lb, np.ma.nomask)
        mub: np.ma.MaskedArray = np.ma.masked_array(ub, np.ma.nomask)
        if self._remove_redundant_x_bounds:
            mlb.mask = np.isneginf(lb)
            mub.mask = np.isposinf(ub)

        self._lbx = np.ma.concatenate((self._lbx, mlb))
        self._ubx = np.ma.concatenate((self._ubx, mub))
        np.ma.set_fill_value(self._lbx, -np.inf)
        np.ma.set_fill_value(self._ubx, +np.inf)

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
    ) -> tuple[SymType, ...]:
        r"""Adds a constraint to the NLP problem, e.g., :math:`lhs \le rhs`.

        Parameters
        ----------
        name : str
            Name of the new constraint. Must not be already in use.
        lhs : casadi.SX, MX, DM or numerical
            Symbolic expression of the left-hand term of the constraint.
        op: {"==", ">=", "<="}
            Operator relating the two terms.
        rhs : casadi.SX, MX, DM or numerical
            Symbolic expression of the right-hand term of the constraint.
        soft : bool, optional
            If ``True``, then a slack variable with appropriate size is added to the NLP
            to make the inequality constraint soft, and returned. This slack is
            automatically lower-bounded by 0, but remember to manually penalize its
            magnitude in the objective. Slacks are not supported for equality
            constraints. Defaults to ``False``.
        simplify : bool, optional
            Optionally simplies the constraint expression, but can be disabled.

        Returns
        -------
        expr : casadi.SX or MX
            The constraint expression in canonical form, i.e., :math:`g(x,p) = 0` or
            :math:`h(x,p) \le 0`.
        lam : casadi.SX or MX
            The symbol corresponding to the constraint's multipliers.
        slack : casadi.SX or MX, optional
            The slack variable in case of ``soft=True``; otherwise, only a 2-tuple is
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

    @invalidate_cache(
        nonmasked_lbx_idx, nonmasked_ubx_idx, h_lbx, h_ubx, lam, primal_dual
    )
    def remove_variable_bounds(
        self,
        name: str,
        direction: Literal["lb", "ub", "both"],
        idx: Union[None, tuple[int, int], list[tuple[int, int]]] = None,
    ) -> None:
        """Removes one or more lower and/or upper bounds from the given variable

        Parameters
        ----------
        name : str
            Name of the variable whose bounds must be modified
        direction : {"lb", "ub", "both"}
            Which bound to modify.
        idx : tuple[int, int] or a list of, optional
            A 2D index, or a list of 2D indices, of the variable entries whose
            corresponding lower/upper bounds must be removed, i.e., set to ``-/+ inf``.
            If not provided, then all the bounds for that variable are removed.

        Notes
        -----
        This is a somewhat costly operation, so it is preferable to avoid creating
        in the first place constraints that will need to be eliminated. Moreover, this
        operation may compromise the results already obtained in, e.g., sensitivity
        analysis, because it changes the underlying NLP problem and there is no way to
        invalidate any user-arbitrary result obtained previously.
        """
        n_rows, n_cols = self._vars[name].shape
        size = n_rows * n_cols
        if idx is None:
            idx_ = np.arange(size, dtype=int)
        else:
            # transform 2D indices to 1D (casadi column-wise)
            if isinstance(idx, tuple):
                idx = (idx,)
            idx_ = np.asarray([i[0] + i[1] * n_rows for i in idx], int)

        # add offset to skip variable created prior to the current
        offset = 0
        for n, other_var in self._vars.items():
            if n == name:
                break
            offset += other_var.shape[0] * other_var.shape[1]
        idx_ += offset

        # set lbx and ubx to -/+ inf
        if direction == "both" or direction == "lb":
            self._lbx[idx_] = -np.inf
        if direction == "both" or direction == "ub":
            self._ubx[idx_] = +np.inf

        if self._remove_redundant_x_bounds:
            # update masks
            lbx_mask = np.ma.getmaskarray(self._lbx)
            ubx_mask = np.ma.getmaskarray(self._ubx)
            lbx_mask[idx_] = ubx_mask[idx_] = True
            self._lbx.mask = lbx_mask
            self._ubx.mask = ubx_mask

            # remove obsolete multipliers
            directions = ("lb", "ub") if direction == "both" else (direction,)
            for lb_or_ub in directions:
                name_lam = f"lam_{lb_or_ub}_{name}"
                bounds = getattr(self, f"_{lb_or_ub}x")[offset : offset + size]
                mask = np.ma.getmaskarray(bounds)
                new_lam = self._sym_type.sym(name_lam, (~mask).sum())

                # replace in dict and re-create vector of lbx/ubx multipliers
                self._dual_vars[name_lam] = new_lam
                all_lams = [
                    lam
                    for n, lam in self._dual_vars.items()
                    if n.startswith(f"lam_{lb_or_ub}")
                ]
                setattr(self, f"_lam_{lb_or_ub}x", cs.vvcat(all_lams))

    @invalidate_cache(lam, primal_dual)
    def remove_constraints(
        self,
        name: str,
        idx: Union[None, tuple[int, int], list[tuple[int, int]]] = None,
    ) -> None:
        """Removes one or more (equality or inequality) constraints from the problem.

        Parameters
        ----------
        name : str
            Name of the constraint to be removed. The name will be used to identify if
            the constraint is an inequality or an equality constraint.
        idx : tuple of 2 ints or a list of, optional
            A 2D index, or a list of 2D indices, of the constraint entries that
            must be removed. If not provided, then the constraint is removed entirely.

        Notes
        -----
        This is a somewhat costly operation, so it is preferable to avoid creating
        in the first place constraints that will need to be eliminated. Moreover, this
        operation may compromise the results already obtained in, e.g., sensitivity
        analysis, because it changes the underlying NLP problem and there is no way to
        invalidate any user-arbitrary result obtained previously.
        """
        old_con = self._cons.pop(name)
        group = "g" if f"lam_g_{name}" in self._dual_vars else "h"
        this_name_lam = f"lam_{group}_{name}"
        self._dual_vars.pop(this_name_lam)
        if idx is not None:
            # transform 2D indices to 1D (casadi column-wise) and keep only the
            # remaining indices
            n_rows = old_con.size1()
            if isinstance(idx, tuple):
                idx = (idx,)
            idx_to_remove = {i[0] + i[1] * n_rows for i in idx}

            # remove constraints and re-create corresponding multipliers
            old_con = cs.vec(old_con)  # flatten the constraint - cannot do otherwise
            idx_to_keep = [i for i in range(old_con.size1()) if i not in idx_to_remove]
            if idx_to_keep:
                new_con = old_con[idx_to_keep]
                self._cons[name] = new_con
                self._dual_vars[this_name_lam] = self._sym_type.sym(
                    this_name_lam, new_con.size1()
                )

        # re-create constraints and multipliers vectors, and refresh the solver
        new_cons = []
        new_lams = []
        for n, con in self._cons.items():
            name_lam = f"lam_{group}_{n}"
            if name_lam in self._dual_vars:
                new_cons.append(con)
                new_lams.append(self._dual_vars[name_lam])
        setattr(self, f"_{group}", cs.vvcat(new_cons))
        setattr(self, f"_lam_{group}", cs.vcat(new_lams))
        if hasattr(self, "refresh_solver"):
            self.refresh_solver()
