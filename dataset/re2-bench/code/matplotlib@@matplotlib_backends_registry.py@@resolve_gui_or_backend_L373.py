import importlib
import importlib.metadata as im
from matplotlib import _parse_to_version_info
from matplotlib import get_backend

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

    def _ensure_entry_points_loaded(self):
        if not self._loaded_entry_points:
            entries = self._read_entry_points()
            self._validate_and_store_entry_points(entries)
            self._loaded_entry_points = True

    def _get_gui_framework_by_loading(self, backend):
        module = self.load_backend_module(backend)
        canvas_class = module.FigureCanvas
        return canvas_class.required_interactive_framework or 'headless'

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

    def _validate_and_store_entry_points(self, entries):
        for name, module in set(entries):
            name = name.lower()
            if name.startswith('module://'):
                raise RuntimeError(f"Entry point name '{name}' cannot start with 'module://'")
            if name in self._BUILTIN_BACKEND_TO_GUI_FRAMEWORK:
                raise RuntimeError(f"Entry point name '{name}' is a built-in backend")
            if name in self._backend_to_gui_framework:
                raise RuntimeError(f"Entry point name '{name}' duplicated")
            self._name_to_module[name] = 'module://' + module
            self._backend_to_gui_framework[name] = 'unknown'

    def backend_for_gui_framework(self, framework):
        """
        Return the name of the backend corresponding to the specified GUI framework.

        Parameters
        ----------
        framework : str
            GUI framework such as "qt".

        Returns
        -------
        str or None
            Backend name or None if GUI framework not recognised.
        """
        return self._GUI_FRAMEWORK_TO_BACKEND.get(framework.lower())

    def load_backend_module(self, backend):
        """
        Load and return the module containing the specified backend.

        Parameters
        ----------
        backend : str
            Name of backend to load.

        Returns
        -------
        Module
            Module containing backend.
        """
        module_name = self._backend_module_name(backend)
        return importlib.import_module(module_name)

    def resolve_backend(self, backend):
        """
        Return the backend and GUI framework for the specified backend name.

        If the GUI framework is not yet known then it will be determined by loading the
        backend module and checking the ``FigureCanvas.required_interactive_framework``
        attribute.

        This function only loads entry points if they have not already been loaded and
        the backend is not built-in and not of ``module://some.backend`` format.

        Parameters
        ----------
        backend : str or None
            Name of backend, or None to use the default backend.

        Returns
        -------
        backend : str
            The backend name.
        framework : str or None
            The GUI framework, which will be None for a backend that is non-interactive.
        """
        if isinstance(backend, str):
            if not backend.startswith('module://'):
                backend = backend.lower()
        else:
            from matplotlib import get_backend
            backend = get_backend()
        gui = self._BUILTIN_BACKEND_TO_GUI_FRAMEWORK.get(backend) or self._backend_to_gui_framework.get(backend)
        if gui is None and isinstance(backend, str) and backend.startswith('module://'):
            gui = 'unknown'
        if gui is None and (not self._loaded_entry_points):
            self._ensure_entry_points_loaded()
            gui = self._backend_to_gui_framework.get(backend)
        if gui == 'unknown':
            gui = self._get_gui_framework_by_loading(backend)
            self._backend_to_gui_framework[backend] = gui
        if gui is None:
            raise RuntimeError(f"'{backend}' is not a recognised backend name")
        return (backend, gui if gui != 'headless' else None)

    def resolve_gui_or_backend(self, gui_or_backend):
        """
        Return the backend and GUI framework for the specified string that may be
        either a GUI framework or a backend name, tested in that order.

        This is for use with the IPython %matplotlib magic command which may be a GUI
        framework such as ``%matplotlib qt`` or a backend name such as
        ``%matplotlib qtagg``.

        This function only loads entry points if they have not already been loaded and
        the backend is not built-in and not of ``module://some.backend`` format.

        Parameters
        ----------
        gui_or_backend : str or None
            Name of GUI framework or backend, or None to use the default backend.

        Returns
        -------
        backend : str
            The backend name.
        framework : str or None
            The GUI framework, which will be None for a backend that is non-interactive.
        """
        if not gui_or_backend.startswith('module://'):
            gui_or_backend = gui_or_backend.lower()
        backend = self.backend_for_gui_framework(gui_or_backend)
        if backend is not None:
            return (backend, gui_or_backend if gui_or_backend != 'headless' else None)
        try:
            return self.resolve_backend(gui_or_backend)
        except Exception:
            raise RuntimeError(f"'{gui_or_backend}' is not a recognised GUI loop or backend name")
