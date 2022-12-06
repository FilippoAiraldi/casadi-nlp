import warnings
from itertools import count
from typing import Any, Dict, Literal, Optional, Sequence, Tuple, Type, TypeVar, Union

import casadi as cs
import numpy as np
import numpy.typing as npt

from csnlp.nlp.core.objective import HasObjective
from csnlp.nlp.debug import NlpDebug
from csnlp.util.io import SupportsDeepcopyAndPickle

T = TypeVar("T", cs.SX, cs.MX)


class Nlp(HasObjective[T], SupportsDeepcopyAndPickle):
    """
    The generic NLP class is a controller that solves a (possibly, nonlinear)
    optimization problem to yield a (possibly, sub-) optimal solution.

    This is a generic class in the sense that it does not solve a particular
    problem, but only offers the generic methods to build one (e.g., variables,
    constraints, objective, solver).
    """

    __ids = count(0)
    is_multi: bool = False

    def __init__(
        self,
        sym_type: Literal["SX", "MX"] = "SX",
        remove_redundant_x_bounds: bool = True,
        name: Optional[str] = None,
        debug: bool = False,
    ) -> None:
        """Creates an NLP instance.

        Parameters
        ----------
        sym_type : "SX" or "MX", optional
            The CasADi symbolic variable type to use in the NLP, by default "SX".
        remove_redundant_x_bounds : bool, optional
            If `True`, then redundant entries in `lbx` and `ubx` are removed when
            properties `h_lbx` and `h_ubx` are called. See these two properties for more
            details. By default, `True`.
        name : str, optional
            Name of the NLP scheme. If `None`, it is automatically assigned.
        debug : bool, optional
            If `True`, the NLP logs in the `debug` property information regarding the
            creation of parameters, variables and constraints. By default, `False`.

        Raises
        ------
        AttributeError
            Raises if the specified CasADi's symbolic type is neither "SX" nor "MX".
        """
        super().__init__(
            sym_type=sym_type, remove_redundant_x_bounds=remove_redundant_x_bounds
        )
        self.id = next(self.__ids)
        self.name = f"{self.__class__.__name__}{self.id}" if name is None else name
        self._debug = NlpDebug() if debug else None

    @property
    def sym_type(self) -> Type[T]:
        """Gets the CasADi symbolic type used in this NLP scheme."""
        return self._csXX

    @property
    def debug(self) -> Optional[NlpDebug]:
        """Gets debug information on the NLP scheme."""
        return self._debug

    @property
    def unwrapped(self) -> "Nlp":
        """Returns the original NLP of the wrapper."""
        return self

    def is_wrapped(self, *args: Any, **kwargs: Any) -> bool:
        """Gets whether the NLP instance is wrapped or not by the given wrapper type."""
        return False

    def parameter(self, name: str, shape: Tuple[int, int] = (1, 1)) -> T:
        out = super().parameter(name, shape)
        if self._debug is not None:
            self._debug.register("p", name, shape)
        return out

    def variable(  # type: ignore
        self,
        name: str,
        shape: Tuple[int, int] = (1, 1),
        lb: Union[npt.ArrayLike, cs.DM] = -np.inf,
        ub: Union[npt.ArrayLike, cs.DM] = +np.inf,
    ) -> Tuple[T, T, T]:
        out = super().variable(name, shape, lb, ub)
        if self._debug is not None:
            self._debug.register("x", name, shape)
        return out

    def constraint(
        self,
        name: str,
        lhs: Union[T, np.ndarray, cs.DM],
        op: Literal["==", ">=", "<="],
        rhs: Union[T, np.ndarray, cs.DM],
        soft: bool = False,
        simplify: bool = True,
    ) -> Tuple[T, ...]:
        out = super().constraint(name, lhs, op, rhs, soft, simplify)
        if self._debug is not None:
            self._debug.register("g" if op == "==" else "h", name, out[0].shape)
        return out

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
            lbg=np.concatenate((self._lbg, self._lbh)),
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

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return super().solve(*args, **kwargs)

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
