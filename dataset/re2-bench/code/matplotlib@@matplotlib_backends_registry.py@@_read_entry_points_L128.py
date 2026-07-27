import importlib.metadata as im
from matplotlib import _parse_to_version_info

class BackendRegistry:
    """
    Registry of backends available within Matplotlib.

    This is the single source of truth for available backends.

    All use of ``BackendRegistry`` should be via the singleton instance
    ``backend_registry`` which can be imported from ``matplotlib.backends``.

    Each backend has a name, a module name containing the backend code, and an
    optional GUI framework that must be running if the backend is interactive.
    There are three sources of backends: built-in (source code is within the
    Matplotlib repository), explicit ``module://some.backend`` syntax (backend is
    obtained by loading the module), or via an entry point (self-registering
    backend in an external package).

    .. versionadded:: 3.9
    """
    _BUILTIN_BACKEND_TO_GUI_FRAMEWORK = {'gtk3agg': 'gtk3', 'gtk3cairo': 'gtk3', 'gtk4agg': 'gtk4', 'gtk4cairo': 'gtk4', 'macosx': 'macosx', 'nbagg': 'nbagg', 'notebook': 'nbagg', 'qtagg': 'qt', 'qtcairo': 'qt', 'qt5agg': 'qt5', 'qt5cairo': 'qt5', 'tkagg': 'tk', 'tkcairo': 'tk', 'webagg': 'webagg', 'wx': 'wx', 'wxagg': 'wx', 'wxcairo': 'wx', 'agg': 'headless', 'cairo': 'headless', 'pdf': 'headless', 'pgf': 'headless', 'ps': 'headless', 'svg': 'headless', 'template': 'headless'}
    _GUI_FRAMEWORK_TO_BACKEND = {'gtk3': 'gtk3agg', 'gtk4': 'gtk4agg', 'headless': 'agg', 'macosx': 'macosx', 'qt': 'qtagg', 'qt5': 'qt5agg', 'qt6': 'qtagg', 'tk': 'tkagg', 'wx': 'wxagg'}

    def __init__(self):
        self._loaded_entry_points = False
        self._backend_to_gui_framework = {}
        self._name_to_module = {'notebook': 'nbagg'}

    def _read_entry_points(self):
        import importlib.metadata as im
        entry_points = im.entry_points(group='matplotlib.backend')
        entries = [(entry.name, entry.value) for entry in entry_points]

        def backward_compatible_entry_points(entries, module_name, threshold_version, names, target):
            from matplotlib import _parse_to_version_info
            try:
                module_version = im.version(module_name)
                if _parse_to_version_info(module_version) < threshold_version:
                    for name in names:
                        entries.append((name, target))
            except im.PackageNotFoundError:
                pass
        names = [entry[0] for entry in entries]
        if 'inline' not in names:
            backward_compatible_entry_points(entries, 'matplotlib_inline', (0, 1, 7), ['inline'], 'matplotlib_inline.backend_inline')
        if 'ipympl' not in names:
            backward_compatible_entry_points(entries, 'ipympl', (0, 9, 4), ['ipympl', 'widget'], 'ipympl.backend_nbagg')
        return entries
