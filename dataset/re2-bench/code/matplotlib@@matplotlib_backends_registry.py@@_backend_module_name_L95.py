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

    def _backend_module_name(self, backend):
        if backend.startswith('module://'):
            return backend[9:]
        backend = backend.lower()
        backend = self._name_to_module.get(backend, backend)
        return backend[9:] if backend.startswith('module://') else f'matplotlib.backends.backend_{backend}'
