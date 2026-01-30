import inspect
from typing import IO, TYPE_CHECKING, cast, overload
import matplotlib
import matplotlib.image
from matplotlib import get_backend as get_backend, rcParams as rcParams
from matplotlib import _pylab_helpers
from matplotlib import cbook
from matplotlib.backend_bases import (
    FigureCanvasBase, FigureManagerBase, MouseButton)
from matplotlib.figure import Figure, FigureBase, figaspect
from matplotlib import rcsetup, rcParamsDefault, rcParamsOrig
from matplotlib.backends import BackendFilter, backend_registry
import matplotlib.axes
import matplotlib.artist
import matplotlib.backend_bases
import matplotlib.backends
import importlib.metadata as im
from matplotlib import _parse_to_version_info  # type: ignore[attr-defined]

def switch_backend(newbackend: str) -> None:
    """
    Set the pyplot backend.

    Switching to an interactive backend is possible only if no event loop for
    another interactive backend has started.  Switching to and from
    non-interactive backends is always possible.

    If the new backend is different than the current backend then all open
    Figures will be closed via ``plt.close('all')``.

    Parameters
    ----------
    newbackend : str
        The case-insensitive name of the backend to use.

    """
    global _backend_mod
    # make sure the init is pulled up so we can assign to it later
    import matplotlib.backends

    if newbackend is rcsetup._auto_backend_sentinel:
        current_framework = cbook._get_running_interactive_framework()

        if (current_framework and
                (backend := backend_registry.backend_for_gui_framework(
                    current_framework))):
            candidates = [backend]
        else:
            candidates = []
        candidates += [
            "macosx", "qtagg", "gtk4agg", "gtk3agg", "tkagg", "wxagg"]

        # Don't try to fallback on the cairo-based backends as they each have
        # an additional dependency (pycairo) over the agg-based backend, and
        # are of worse quality.
        for candidate in candidates:
            try:
                switch_backend(candidate)
            except ImportError:
                continue
            else:
                rcParamsOrig['backend'] = candidate
                return
        else:
            # Switching to Agg should always succeed; if it doesn't, let the
            # exception propagate out.
            switch_backend("agg")
            rcParamsOrig["backend"] = "agg"
            return
    old_backend = rcParams._get('backend')  # get without triggering backend resolution

    module = backend_registry.load_backend_module(newbackend)
    canvas_class = module.FigureCanvas

    required_framework = canvas_class.required_interactive_framework
    if required_framework is not None:
        current_framework = cbook._get_running_interactive_framework()
        if (current_framework and required_framework
                and current_framework != required_framework):
            raise ImportError(
                "Cannot load backend {!r} which requires the {!r} interactive "
                "framework, as {!r} is currently running".format(
                    newbackend, required_framework, current_framework))

    # Load the new_figure_manager() and show() functions from the backend.

    # Classically, backends can directly export these functions.  This should
    # keep working for backcompat.
    new_figure_manager = getattr(module, "new_figure_manager", None)
    show = getattr(module, "show", None)

    # In that classical approach, backends are implemented as modules, but
    # "inherit" default method implementations from backend_bases._Backend.
    # This is achieved by creating a "class" that inherits from
    # backend_bases._Backend and whose body is filled with the module globals.
    class backend_mod(matplotlib.backend_bases._Backend):
        locals().update(vars(module))

    # However, the newer approach for defining new_figure_manager and
    # show is to derive them from canvas methods.  In that case, also
    # update backend_mod accordingly; also, per-backend customization of
    # draw_if_interactive is disabled.
    if new_figure_manager is None:

        def new_figure_manager_given_figure(num, figure):
            return canvas_class.new_manager(figure, num)

        def new_figure_manager(num, *args, FigureClass=Figure, **kwargs):
            fig = FigureClass(*args, **kwargs)
            return new_figure_manager_given_figure(num, fig)

        def draw_if_interactive() -> None:
            if matplotlib.is_interactive():
                manager = _pylab_helpers.Gcf.get_active()
                if manager:
                    manager.canvas.draw_idle()

        backend_mod.new_figure_manager_given_figure = (  # type: ignore[method-assign]
            new_figure_manager_given_figure)
        backend_mod.new_figure_manager = (  # type: ignore[method-assign]
            new_figure_manager)
        backend_mod.draw_if_interactive = (  # type: ignore[method-assign]
            draw_if_interactive)

    # If the manager explicitly overrides pyplot_show, use it even if a global
    # show is already present, as the latter may be here for backcompat.
    manager_class = getattr(canvas_class, "manager_class", None)
    # We can't compare directly manager_class.pyplot_show and FMB.pyplot_show because
    # pyplot_show is a classmethod so the above constructs are bound classmethods, and
    # thus always different (being bound to different classes).  We also have to use
    # getattr_static instead of vars as manager_class could have no __dict__.
    manager_pyplot_show = inspect.getattr_static(manager_class, "pyplot_show", None)
    base_pyplot_show = inspect.getattr_static(FigureManagerBase, "pyplot_show", None)
    if (show is None
            or (manager_pyplot_show is not None
                and manager_pyplot_show != base_pyplot_show)):
        if not manager_pyplot_show:
            raise ValueError(
                f"Backend {newbackend} defines neither FigureCanvas.manager_class nor "
                f"a toplevel show function")
        _pyplot_show = cast('Any', manager_class).pyplot_show
        backend_mod.show = _pyplot_show  # type: ignore[method-assign]

    _log.debug("Loaded backend %s version %s.",
               newbackend, backend_mod.backend_version)

    if newbackend in ("ipympl", "widget"):
        # ipympl < 0.9.4 expects rcParams["backend"] to be the fully-qualified backend
        # name "module://ipympl.backend_nbagg" not short names "ipympl" or "widget".
        import importlib.metadata as im
        from matplotlib import _parse_to_version_info  # type: ignore[attr-defined]
        try:
            module_version = im.version("ipympl")
            if _parse_to_version_info(module_version) < (0, 9, 4):
                newbackend = "module://ipympl.backend_nbagg"
        except im.PackageNotFoundError:
            pass

    rcParams['backend'] = rcParamsDefault['backend'] = newbackend
    _backend_mod = backend_mod
    for func_name in ["new_figure_manager", "draw_if_interactive", "show"]:
        globals()[func_name].__signature__ = inspect.signature(
            getattr(backend_mod, func_name))

    # Need to keep a global reference to the backend for compatibility reasons.
    # See https://github.com/matplotlib/matplotlib/issues/6092
    matplotlib.backends.backend = newbackend  # type: ignore[attr-defined]

    # Make sure the repl display hook is installed in case we become interactive.
    try:
        install_repl_displayhook()
    except NotImplementedError as err:
        _log.warning("Fallback to a different backend")
        raise ImportError from err
