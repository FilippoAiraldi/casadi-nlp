"""A collection of stand-alone functions for plotting purposes."""

from typing import Any, Union

import casadi as cs
import numpy as np
import numpy.typing as npt

MATLAB_COLORS = [
    "#0072BD",
    "#D95319",
    "#EDB120",
    "#7E2F8E",
    "#77AC30",
    "#4DBEEE",
    "#A2142F",
]


def save2tikz(*figs) -> None:
    """Saves the figure to a tikz file (``.tex`` extension). See
    `tikzplotlib <https://pypi.org/project/tikzplotlib/>`_ for more details.

    Parameters
    ----------
    figs : :class:`matplotlib.figure.Figure`
        One or more matplotlib figures to be converted to tikz files. These
        files will be named based on the number of the corresponding figure.
    """
    import matplotlib as mpl
    import tikzplotlib

    # monkey patching to fix some issues with tikzplotlib
    mpl.lines.Line2D._us_dashSeq = property(lambda self: self._dash_pattern[1])
    mpl.lines.Line2D._us_dashOffset = property(lambda self: self._dash_pattern[0])
    mpl.legend.Legend._ncol = property(lambda self: self._ncols)
    for fig in figs:
        tikzplotlib.save(
            f"figure_{fig.number}.tex",
            figure=fig,
            extra_axis_parameters={r"tick scale binop=\times"},
        )


def spy(
    H: Union[cs.SX, cs.MX, cs.DM, cs.Sparsity, npt.ArrayLike],
    ax=None,
    **spy_kwargs: Any,
):
    """Implementation equivalent to :func:`matplotlib.pyplot.spy` that works also with
    CasADi symbolic matrices.

    Parameters
    ----------
    H : casadi SX, MX, DM, Sparsity or array_like
        The matrix (or its sparsity pattern) to spy.
    ax : :class:`matplotlib.axes.Axes`, optional
        The axis to draw the result on. If ``None``, creates a new axis.
    spy_kwargs
        Other arguments passed directly to :func:`matplotlib.pyplot.spy`.

    Returns
    -------
    :class:`matplotlib.image.AxesImage` or :class:`matplotlib.lines.Line2D`
        Same return types of :func:`matplotlib.pyplot.spy`.
    """
    import matplotlib.pyplot as plt

    if hasattr(H, "sparsity") and callable(H.sparsity):
        H = H.sparsity()
    if hasattr(H, "spy") and callable(H.spy):
        from contextlib import redirect_stdout
        from io import StringIO

        f = StringIO()
        with redirect_stdout(f):
            H.spy()
        out = f.getvalue()
        H = np.asarray(
            [
                list(line)
                for line in out.replace(".", "0").replace("*", "1").splitlines()
            ],
            dtype=int,
        )
    else:
        H = np.asarray(H)

    if ax is None:
        _, ax = plt.subplots(1, 1)
    o = ax.spy(H, **spy_kwargs)
    nz = np.count_nonzero(H)
    ax.set_xlabel(f"nz = {nz} ({nz / H.size * 100:.2f}%)")
    return o


def set_mpl_defaults(
    mpl_style: str = "bmh",
    linewidth: float = 1.5,
    markersize: float = 2,
    savefig_dpi: int = 600,
    np_print_precision: int = 4,
    matlab_colors: bool = False,
) -> None:
    """Sets some default parameters for :mod:`numpy` and :mod:`matplotlib` for printing
    and plotting.

    Parameters
    ----------
    mpl_style : str, optional
        :mod:`matplotlib` plotting style, by default ``"bmh"``.
    linewidth : float, optional
        :mod:`matplotlib` default linewidth, by default ``1.5``.
    markersize : float, optional
        :mod:`matplotlib` default markersize, by default ``2``.
    savefig_dpi : int, optional
        :mod:`matplotlib` savefig dpi, by default ``600``.
    np_print_precision : int, optional
        :mod:`numpy` printing precision, by default ``4``.
    matlab_colors : bool, optional
        Whether :mod:`matplotlib` should use Matlab colors for plotting, by default
        ``False``.
    """
    import matplotlib as mpl

    np.set_printoptions(precision=np_print_precision)
    mpl.style.use(mpl_style)  # 'seaborn-darkgrid'
    mpl.rcParams["lines.solid_capstyle"] = "round"
    mpl.rcParams["lines.linewidth"] = linewidth
    mpl.rcParams["lines.markersize"] = markersize
    mpl.rcParams["savefig.dpi"] = savefig_dpi
    if matlab_colors:
        from cycler import cycler

        mpl.rcParams["axes.prop_cycle"] = cycler("color", MATLAB_COLORS)
