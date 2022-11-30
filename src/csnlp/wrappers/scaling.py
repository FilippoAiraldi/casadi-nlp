from typing import Dict, Tuple, Union
from warnings import warn

import casadi as cs
import numpy as np

from csnlp.solutions import DMStruct, Solution
from csnlp.util.data import dict2struct, struct_symSX
from csnlp.util.funcs import cache_clearer, cached_property
from csnlp.util.scaling import Scaler
from csnlp.wrappers.wrapper import Nlp, NonRetroactiveWrapper


class NlpScaling(NonRetroactiveWrapper):
    def __init__(self, nlp: Nlp, scaler: Scaler, warns: bool = True) -> None:
        super().__init__(nlp)
        self.scaler = scaler
        self.warns = warns
        self.scaled_vars: Dict[str, Union[cs.SX, cs.MX]] = {}
        self.scaled_pars: Dict[str, Union[cs.SX, cs.MX]] = {}

    @cached_property
    def variables(self) -> Union[struct_symSX, Dict[str, cs.MX]]:
        """Gets the usncaled primal variables of the NLP scheme."""
        return dict2struct(self.nlp.unwrapped._vars, entry_type="expr")

    @cached_property
    def parameters(self) -> Union[struct_symSX, Dict[str, cs.MX]]:
        """Gets the unscaled parameters of the NLP scheme."""
        return dict2struct(self.nlp.unwrapped._pars, entry_type="expr")

    @cache_clearer(variables)
    def variable(
        self,
        name: str,
        shape: Tuple[int, int] = (1, 1),
        lb: Union[np.ndarray, cs.DM] = -np.inf,
        ub: Union[np.ndarray, cs.DM] = +np.inf,
    ) -> Union[Tuple[cs.SX, cs.SX, cs.SX], Tuple[cs.MX, cs.MX, cs.MX]]:
        """See `Nlp.variable` method."""
        if name not in self.scaler:
            if self.warns:
                warn(f"Scaling for variable {name} not found.", RuntimeWarning)
            return self.nlp.variable(name, shape, lb, ub)
        # scale bounds
        lb, ub = np.broadcast_to(lb, shape), np.broadcast_to(ub, shape)
        lb = self.scaler.scale(name, lb)
        ub = self.scaler.scale(name, ub)

        var, lam_lb, lam_ub = self.nlp.variable(name, shape, lb, ub)

        # unscale var and return it
        uvar = self.scaler.unscale(name, var)
        self.nlp.unwrapped._vars[name] = uvar
        self.scaled_vars[name] = var
        return uvar, lam_lb, lam_ub

    @cache_clearer(parameters)
    def parameter(
        self, name: str, shape: Tuple[int, int] = (1, 1)
    ) -> Union[cs.SX, cs.MX]:
        """See `Nlp.parameter` method."""
        par = self.nlp.parameter(name, shape)
        if name not in self.scaler:
            if self.warns:
                warn(f"Scaling for parameter {name} not found.", RuntimeWarning)
            return par

        # unscale par and return it
        upar = self.scaler.unscale(name, par)
        self.nlp.unwrapped._pars[name] = upar
        self.scaled_pars[name] = par
        return upar

    def solve(
        self,
        pars: Union[None, DMStruct, Dict[str, np.ndarray]] = None,
        vals0: Union[None, DMStruct, Dict[str, np.ndarray]] = None,
    ) -> Solution:
        """See `Nlp.solve` method."""

        # NOTE: swapping the unscaled expressions with the scaled variables is required
        # because in Nlp.solve we cannot assign parameter/variable values from
        # expressions (e.g., 100 * x = 10), but only from assignments (e.g., x = 0.1).
        # In turn, this requires to scale the input pars and vals0.

        # swap vars (expr) with scaled vars (variables)
        self.nlp.unwrapped._vars, self.scaled_vars = (
            self.scaled_vars,
            self.nlp.unwrapped._vars,
        )
        self.nlp.unwrapped._pars, self.scaled_pars = (
            self.scaled_pars,
            self.nlp.unwrapped._pars,
        )

        # scale parameters and initial conditions and solve
        scaler = self.scaler
        pars = {
            name: scaler.scale(name, pars[name])
            if scaler.can_scale(name)
            else pars[name]
            for name in pars.keys()
        }
        vals0 = {
            name: scaler.scale(name, vals0[name])
            if scaler.can_scale(name)
            else vals0[name]
            for name in vals0.keys()
        }
        sol = self.nlp.solve(pars, vals0)

        # swap back to vars (expr) with scaled vars (variables)
        self.nlp.unwrapped._vars, self.scaled_vars = (
            self.scaled_vars,
            self.nlp.unwrapped._vars,
        )
        self.nlp.unwrapped._pars, self.scaled_pars = (
            self.scaled_pars,
            self.nlp.unwrapped._pars,
        )
        return sol
