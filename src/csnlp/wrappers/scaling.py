from collections.abc import Iterable
from typing import Literal, Optional, TypeVar, Union
from warnings import warn

import casadi as cs
import numpy as np
import numpy.typing as npt

from ..core.scaling import Scaler
from ..core.solutions import Solution, subsevalf
from .wrapper import Nlp, NonRetroactiveWrapper

SymType = TypeVar("SymType", cs.SX, cs.MX)


def _scale_dict(
    d: dict[str, npt.ArrayLike], scaler: Scaler
) -> dict[str, npt.ArrayLike]:
    """Internal utility for scaling dicts."""
    return {
        name: scaler.scale(name, d[name]) if scaler.can_scale(name) else d[name]
        for name in d
    }


class NlpScaling(NonRetroactiveWrapper[SymType]):
    """Wraps an instance of :class:`csnlp.Nlp` to facilitate the scaling of its
    parameters and/or variables as well as the automatic scaling of expressions (e.g.,
    objective and constraints).

    Parameters
    ----------
    nlp : Nlp[T]
        The NLP problem to be wrapped.
    scaler : Scaler
        A class for scaling the NLP's quantities. See :mod:`csnlp.core.scaling` for a
        collection of these.
    warns : bool, optional
        If ``True``, warns each time a variable or parameter is created which has not
        been registered to the scaler and thus cannot be scaled; otherwise, it will not
        raise warnings.
    """

    def __init__(self, nlp: Nlp[SymType], scaler: Scaler, warns: bool = True) -> None:
        super().__init__(nlp)
        self.scaler = scaler
        self.warns = warns
        self._svars: dict[str, SymType] = {}
        self._spars: dict[str, SymType] = {}
        self._uvars: dict[str, SymType] = {}
        self._upars: dict[str, SymType] = {}

    @property
    def scaled_variables(self) -> dict[str, SymType]:
        """Gets the scaled variables of the NLP scheme."""
        return self._svars

    @property
    def scaled_parameters(self) -> dict[str, SymType]:
        """Gets the scaled parameters of the NLP scheme."""
        return self._spars

    @property
    def unscaled_variables(self) -> dict[str, SymType]:
        """Gets the unscaled variables of the NLP scheme."""
        return self._uvars

    @property
    def unscaled_parameters(self) -> dict[str, SymType]:
        """Gets the unscaled parameters of the NLP scheme."""
        return self._upars

    def scale(self, expr: SymType) -> SymType:
        """Scales an expression with the MPC's scaled variables and parameters.

        Parameters
        ----------
        expr : casadi.SX or MX
            The expression to be scaled.

        Returns
        -------
        casadi.SX or MX
            The scaled expression.
        """
        expr = subsevalf(expr, self.nlp.variables, self._svars, eval=False)
        return subsevalf(expr, self.nlp.parameters, self._spars, eval=False)

    def unscale(self, expr: SymType) -> SymType:
        """Unscales an expression with the MPC's unscaled variables and parameters.

        Parameters
        ----------
        expr : casadi.SX or MX
            The expression to be unscaled.

        Returns
        -------
        casadi.SX or MX
            The unscaled expression.
        """
        expr = subsevalf(expr, self.nlp.variables, self._uvars, eval=False)
        return subsevalf(expr, self.nlp.parameters, self._upars, eval=False)

    def variable(
        self,
        name: str,
        shape: tuple[int, int] = (1, 1),
        discrete: bool = False,
        lb: Union[npt.ArrayLike, cs.DM] = -np.inf,
        ub: Union[npt.ArrayLike, cs.DM] = +np.inf,
    ) -> tuple[SymType, SymType, SymType]:
        """See :meth:`csnlp.Nlp.variable`."""
        can_scale = name in self.scaler
        if can_scale:
            lb, ub = np.broadcast_to(lb, shape), np.broadcast_to(ub, shape)
            lb = self.scaler.scale(name, lb)
            ub = self.scaler.scale(name, ub)
        var, lam_lb, lam_ub = self.nlp.variable(name, shape, discrete, lb, ub)
        if can_scale:
            svar = self.scaler.scale(name, var)
            uvar = self.scaler.unscale(name, var)
        else:
            if self.warns:
                warn(f"Scaling for variable {name} not found.", RuntimeWarning, 2)
            svar = uvar = var
        self._svars[name] = svar
        self._uvars[name] = uvar
        return var, lam_lb, lam_ub

    def parameter(self, name: str, shape: tuple[int, int] = (1, 1)) -> SymType:
        """See :meth:`csnlp.Nlp.parameter`."""
        par = self.nlp.parameter(name, shape)
        if name in self.scaler:
            spar = self.scaler.scale(name, par)
            upar = self.scaler.unscale(name, par)
        else:
            if self.warns:
                warn(f"Scaling for parameter {name} not found.", RuntimeWarning, 2)
            spar = upar = par
        self._spars[name] = spar
        self._upars[name] = upar
        return par

    def constraint(
        self,
        name: str,
        lhs: Union[SymType, np.ndarray, cs.DM],
        op: Literal["==", ">=", "<="],
        rhs: Union[SymType, np.ndarray, cs.DM],
        soft: bool = False,
        simplify: bool = True,
    ) -> tuple[SymType, ...]:
        """See :meth:`csnlp.Nlp.constraint`."""
        return self.nlp.constraint(name, self.unscale(lhs - rhs), op, 0, soft, simplify)

    def minimize(self, objective: SymType) -> None:
        """See :meth:`csnlp.Nlp.minimize`."""
        return self.nlp.minimize(self.unscale(objective))

    def solve(
        self,
        pars: Optional[dict[str, npt.ArrayLike]] = None,
        vals0: Optional[dict[str, npt.ArrayLike]] = None,
    ) -> Solution[SymType]:
        """See :meth:`csnlp.Nlp.solve`."""
        scaler = self.scaler
        if pars is not None:
            pars = _scale_dict(pars, scaler)
        if vals0 is not None:
            vals0 = _scale_dict(vals0, scaler)
        return self.nlp.solve(pars, vals0)

    def solve_multi(
        self,
        pars: Union[
            None, dict[str, npt.ArrayLike], Iterable[dict[str, npt.ArrayLike]]
        ] = None,
        vals0: Union[
            None, dict[str, npt.ArrayLike], Iterable[dict[str, npt.ArrayLike]]
        ] = None,
        return_all_sols: bool = False,
        return_stacked_sol: bool = False,
    ) -> Union[Solution[SymType], list[Solution[SymType]]]:
        """See :meth:`csnlp.MultistartNlp.solve_multi`."""
        assert self.nlp.is_multi and hasattr(self.nlp, "solve_multi"), (
            "`solve_multi` called on an nlp instance that is not `MultistartNlp`."
        )
        scaler = self.scaler
        if pars is not None:
            pars = (
                _scale_dict(pars, scaler)
                if isinstance(pars, dict)
                else (_scale_dict(pars_i, scaler) for pars_i in pars)
            )
        if vals0 is not None:
            vals0 = (
                _scale_dict(vals0, scaler)
                if isinstance(vals0, dict)
                else (_scale_dict(vals0_i, scaler) for vals0_i in vals0)
            )
        return self.nlp.solve_multi(
            pars,
            vals0,
            return_all_sols=return_all_sols,
            return_stacked_sol=return_stacked_sol,
        )
