from collections.abc import Iterator, Sequence
from itertools import count
from typing import Any, ClassVar, Literal, Optional, TypeVar, Union

import casadi as cs
import numpy as np
import numpy.typing as npt
from joblib import Memory

from ..core.debug import NlpDebug
from .objective import HasObjective

SymType = TypeVar("SymType", cs.SX, cs.MX)


class Nlp(HasObjective[SymType]):
    r"""The generic NLP class is a controller that solves a (possibly, nonlinear)
    optimization problem to yield a (possibly, sub-) optimal solution. This is a generic
    implementation in the sense that it does not solve a particular problem, but only
    offers the generic methods to build one (e.g., variables, constraints, objective,
    solver).

    Parameters
    ----------
    sym_type : {"SX", "MX"}, optional
        The CasADi symbolic variable type to use in the NLP, by default ``"SX"``.
    remove_redundant_x_bounds : bool, optional
        If ``True``, then redundant entries in :meth:`lbx` and :meth:`ubx` are removed
        when properties :meth:`h_lbx` and :meth:`h_ubx` are called. See these two
        properties for more details. By default, ``True``.
    cache : joblib.Memory, optional
        Optional cache to avoid computing the same exact NLP more than once. By default,
        no caching occurs.
    name : str, optional
        Name of the NLP scheme. If `None`, it is automatically assigned.
    debug : bool, optional
        If ``True``, the NLP logs in the :meth:`debug` property information regarding
        the creation of parameters, variables and constraints. By default, ``False``.

    Raises
    ------
    AttributeError
        Raises if the specified CasADi's symbolic type is neither ``"SX"`` nor ``"MX"``.

    Notes
    -----
    Constraints are internally handled in their canonical form, i.e., :math:`g(x,p) = 0`
    and :math:`h(x,p) \leq 0`. The objective :math:`f(x,p)` is always a scalar function
    to be minimized.
    """

    __ids: ClassVar[Iterator[int]] = count(0)
    is_multi: ClassVar[bool] = False
    """Flag to indicate that this is not a multistart NLP."""

    def __init__(
        self,
        sym_type: Literal["SX", "MX"] = "SX",
        remove_redundant_x_bounds: bool = True,
        cache: Memory = None,
        name: Optional[str] = None,
        debug: bool = False,
    ) -> None:
        id = next(self.__ids)
        name = f"{self.__class__.__name__}{id}" if name is None else name
        super().__init__(sym_type, remove_redundant_x_bounds, cache, name)
        self.id = id
        self._debug = NlpDebug() if debug else None

    @property
    def sym_type(self) -> type[SymType]:
        """Gets the CasADi symbolic type used in this NLP scheme."""
        return self._sym_type

    @property
    def debug(self) -> Optional[NlpDebug]:
        """Gets debug information on the NLP scheme."""
        return self._debug

    @property
    def unwrapped(self) -> "Nlp[SymType]":
        """Returns the original NLP of the wrapper."""
        return self

    def is_wrapped(self, *_: Any, **__: Any) -> bool:
        """Gets whether the NLP instance is wrapped or not by the given wrapper type."""
        return False

    def parameter(self, name: str, shape: tuple[int, int] = (1, 1)) -> SymType:
        out = super().parameter(name, shape)
        if self._debug is not None:
            self._debug.register("p", name, shape)
        return out

    def variable(
        self,
        name: str,
        shape: tuple[int, int] = (1, 1),
        discrete: bool = False,
        lb: Union[npt.ArrayLike, cs.DM] = -np.inf,
        ub: Union[npt.ArrayLike, cs.DM] = +np.inf,
    ) -> tuple[SymType, SymType, SymType]:
        out = super().variable(name, shape, discrete, lb, ub)
        if self._debug is not None:
            self._debug.register("x", name, shape)
        return out

    def constraint(
        self,
        name: str,
        lhs: Union[SymType, np.ndarray, cs.DM],
        op: Literal["==", ">=", "<="],
        rhs: Union[SymType, np.ndarray, cs.DM],
        soft: bool = False,
        simplify: bool = True,
    ) -> tuple[SymType, ...]:
        out = super().constraint(name, lhs, op, rhs, soft, simplify)
        if self._debug is not None:
            self._debug.register("g" if op == "==" else "h", name, out[0].shape)
        return out

    def to_function(
        self,
        name: str,
        ins: Sequence[SymType],
        outs: Sequence[SymType],
        name_in: Optional[Sequence[str]] = None,
        name_out: Optional[Sequence[str]] = None,
        opts: Optional[dict[Any, Any]] = None,
        mx_prewrap: bool = False,
    ) -> cs.Function:
        """Converts the optimization problem to an ``SX`` or ``MX`` symbolic
        :class:`casadi.Function`. If the NLP is modelled in ``SX``, the function can be
        pre-wrapped in ``MX``.

        Parameters
        ----------
        name : str
            Name of the function.
        ins : sequence of casadi.SX or MX
            Input variables of the function. These must be expressions providing the
            parameters of the NLP and the initial conditions of the primal variables
            ``x``.
        outs : sequence of casadi.SX or MX
            Output variables of the function. These must be expressions depending on the
            primal variable ``x``, parameters ``p``, and dual variables ``lam_g``,
            ``lam_h``, ``lam_lbx``, ``lam_ubx`` of the NLP.
        name_in : sequence of str, optional
            Name of the inputs, by default ``None``.
        name_out : sequence of str, optional
            Name of the outpus, by default ``None``.
        opts : dict[Any, Any], optional
            Options to be passed to :class:`casadi.Function`, by default ``None``.
        mx_prewrap : bool, optional
            If ``True``, wraps the CasADi interface in an ``MX`` wrapper prior to
            turning it into the function. This is useful when the NLP is defined in
            ``SX`` but the interface is only supported in ``MX``. By default, ``False``.

        Returns
        -------
        casadi.Function
            The NLP solver as an instance of :class:`casadi.Function`.

        Raises
        ------
        RuntimeError
            Raises if the solver is uninitialized, or if the input or output expressions
            have free variables that are not provided or cannot be computed by the
            solver.
        """
        S = self._solver
        if S is None:
            raise RuntimeError("Solver not yet initialized.")
        n_ins = len(ins)
        n_outs = len(outs)

        # converts inputs/outputs to/from variables and parameters
        Fin = cs.Function("Fin", ins, (self._x, self._p))
        Fout = cs.Function(
            "Fout",
            (self._p, self._x, self._lam_g, self._lam_h, self._lam_lbx, self._lam_ubx),
            outs,
        )

        # call the solver
        if mx_prewrap:
            Fin = Fin.wrap()
            Fout = Fout.wrap()
            ins = [Fin.mx_in(i) for i in range(n_ins)]
        x0, p = Fin(*ins)
        sol = S(
            x0=x0,
            p=p,
            lbx=self._lbx.data,
            ubx=self._ubx.data,
            lbg=np.concatenate((np.zeros(self.ng), np.full(self.nh, -np.inf))),
            ubg=0,
            lam_x0=0,
            lam_g0=0,
        )
        x = sol["x"]
        lam_g = sol["lam_g"][: self.ng, :]
        lam_h = sol["lam_g"][self.ng :, :]
        lam_lbx = -cs.fmin(sol["lam_x"], 0)[self.nonmasked_lbx_idx, :]
        lam_ubx = cs.fmax(sol["lam_x"], 0)[self.nonmasked_ubx_idx, :]

        # build final function
        final_outs = Fout(p, x, lam_g, lam_h, lam_lbx, lam_ubx)
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
