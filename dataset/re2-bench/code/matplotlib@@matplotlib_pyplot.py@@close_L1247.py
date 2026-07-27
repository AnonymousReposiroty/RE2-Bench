from matplotlib import _api
from matplotlib import _pylab_helpers
from matplotlib.figure import Figure, FigureBase, figaspect
from typing import Any, BinaryIO, Literal, TypeVar

def close(fig: None | int | str | Figure | Literal["all"] = None) -> None:
    """
    Close a figure window, and unregister it from pyplot.

    Parameters
    ----------
    fig : None or int or str or `.Figure`
        The figure to close. There are a number of ways to specify this:

        - *None*: the current figure
        - `.Figure`: the given `.Figure` instance
        - ``int``: a figure number
        - ``str``: a figure name
        - 'all': all figures

    Notes
    -----
    pyplot maintains a reference to figures created with `figure()`. When
    work on the figure is completed, it should be closed, i.e. deregistered
    from pyplot, to free its memory (see also :rc:figure.max_open_warning).
    Closing a figure window created by `show()` automatically deregisters the
    figure. For all other use cases, most prominently `savefig()` without
    `show()`, the figure must be deregistered explicitly using `close()`.
    """
    if fig is None:
        manager = _pylab_helpers.Gcf.get_active()
        if manager is None:
            return
        else:
            _pylab_helpers.Gcf.destroy(manager)
    elif fig == 'all':
        _pylab_helpers.Gcf.destroy_all()
    elif isinstance(fig, int):
        _pylab_helpers.Gcf.destroy(fig)
    elif hasattr(fig, 'int'):  # UUIDs get converted to ints by figure().
        _pylab_helpers.Gcf.destroy(fig.int)
    elif isinstance(fig, str):
        all_labels = get_figlabels()
        if fig in all_labels:
            num = get_fignums()[all_labels.index(fig)]
            _pylab_helpers.Gcf.destroy(num)
    elif isinstance(fig, Figure):
        _pylab_helpers.Gcf.destroy_fig(fig)
    else:
        _api.check_isinstance(  # type: ignore[unreachable]
            (Figure, int, str, None), fig=fig)
