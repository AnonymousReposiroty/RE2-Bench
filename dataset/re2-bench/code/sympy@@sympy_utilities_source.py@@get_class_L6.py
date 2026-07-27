def get_class(lookup_view):
    """
    Convert a string version of a class name to the object.

    For example, get_class('sympy.core.Basic') will return
    class Basic located in module sympy.core
    """
    if isinstance(lookup_view, str):
        mod_name, func_name = get_mod_func(lookup_view)
        if func_name != '':
            lookup_view = getattr(
                __import__(mod_name, {}, {}, ['*']), func_name)
            if not callable(lookup_view):
                raise AttributeError(
                    "'%s.%s' is not a callable." % (mod_name, func_name))
    return lookup_view
