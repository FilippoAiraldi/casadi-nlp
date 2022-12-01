from typing import Dict, Literal, Tuple, Union
from warnings import warn

import casadi as cs
import numpy as np

from csnlp.solutions import DMStruct, Solution, subsevalf
from csnlp.util.data import dict2struct, struct_symSX
from csnlp.util.funcs import cache_clearer, cached_property
from csnlp.util.scaling import Scaler
from csnlp.wrappers.wrapper import Nlp, NonRetroactiveWrapper


class NlpScaling(NonRetroactiveWrapper):
    def __init__(self, nlp: Nlp, scaler: Scaler, warns: bool = True) -> None:
        super().__init__(nlp)
        self.scaler = scaler
        self.warns = warns
        self._unscaled_vars: Dict[str, Union[cs.SX, cs.MX]] = {}
        self._unscaled_pars: Dict[str, Union[cs.SX, cs.MX]] = {}

    @cached_property
    def unscaled_variables(self) -> Union[struct_symSX, Dict[str, cs.MX]]:
        """Gets the unscaled variables of the NLP scheme."""
        return dict2struct(self._unscaled_vars, entry_type="expr")

    @cached_property
    def unscaled_parameters(self) -> Union[struct_symSX, Dict[str, cs.MX]]:
        """Gets the unscaled parameters of the NLP scheme."""
        return dict2struct(self._unscaled_pars, entry_type="expr")

    @cache_clearer(unscaled_variables)
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

        # unscale var and return
        uvar = self.scaler.unscale(name, var)
        self._unscaled_vars[name] = uvar
        return var, lam_lb, lam_ub

    @cache_clearer(unscaled_parameters)
    def parameter(
        self, name: str, shape: Tuple[int, int] = (1, 1)
    ) -> Union[cs.SX, cs.MX]:
        """See `Nlp.parameter` method."""
        par = self.nlp.parameter(name, shape)
        if name not in self.scaler:
            if self.warns:
                warn(f"Scaling for parameter {name} not found.", RuntimeWarning)
            return par

        # unscale par and return
        upar = self.scaler.unscale(name, par)
        self._unscaled_pars[name] = upar
        return par

    def constraint(
        self,
        name: str,
        lhs: Union[np.ndarray, cs.DM, cs.SX, cs.MX],
        op: Literal["==", ">=", "<="],
        rhs: Union[np.ndarray, cs.DM, cs.SX, cs.MX],
        soft: bool = False,
        simplify: bool = True,
    ) -> Union[
        Tuple[cs.SX, cs.SX],
        Tuple[cs.MX, cs.MX],
        Tuple[cs.SX, cs.SX, cs.SX],
        Tuple[cs.MX, cs.MX, cs.MX],
    ]:
        """See `Nlp.constraint` method."""
        e = lhs - rhs
        e = subsevalf(e, self.nlp.variables, self.unscaled_variables, eval=False)
        e = subsevalf(e, self.nlp.parameters, self.unscaled_parameters, eval=False)
        return self.nlp.constraint(name, e, op, 0, soft=soft, simplify=simplify)

    def minimize(self, objective: Union[cs.SX, cs.MX]) -> None:
        """See `Nlp.minimize` method."""
        o = objective
        o = subsevalf(o, self.nlp.variables, self.unscaled_variables, eval=False)
        o = subsevalf(o, self.nlp.parameters, self.unscaled_parameters, eval=False)
        return self.nlp.minimize(o)

    def solve(
        self,
        pars: Union[None, DMStruct, Dict[str, np.ndarray]] = None,
        vals0: Union[None, DMStruct, Dict[str, np.ndarray]] = None,
    ) -> Solution:
        """See `Nlp.solve` method."""
        scaler = self.scaler
        scaled_pars = {
            name: scaler.scale(name, pars[name])
            if scaler.can_scale(name)
            else pars[name]
            for name in pars.keys()
        }
        scaled_vals0 = {
            name: scaler.scale(name, vals0[name])
            if scaler.can_scale(name)
            else vals0[name]
            for name in vals0.keys()
        }
        return self.nlp.solve(scaled_pars, scaled_vals0)
