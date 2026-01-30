import inspect

def _get_pyplot_commands() -> list[str]:
    # This works by searching for all functions in this module and removing
    # a few hard-coded exclusions, as well as all of the colormap-setting
    # functions, and anything marked as private with a preceding underscore.
    exclude = {'colormaps', 'colors', 'get_plot_commands', *colormaps}
    this_module = inspect.getmodule(get_plot_commands)
    return sorted(
        name for name, obj in globals().items()
        if not name.startswith('_') and name not in exclude
           and inspect.isfunction(obj)
           and inspect.getmodule(obj) is this_module)
