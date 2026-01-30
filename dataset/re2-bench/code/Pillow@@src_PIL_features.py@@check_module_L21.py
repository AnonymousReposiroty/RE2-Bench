import warnings

def check_module(feature):
    """
    Checks if a module is available.

    :param feature: The module to check for.
    :returns: ``True`` if available, ``False`` otherwise.
    :raises ValueError: If the module is not defined in this version of Pillow.
    """
    if feature not in modules:
        msg = f"Unknown module {feature}"
        raise ValueError(msg)

    module, ver = modules[feature]

    try:
        __import__(module)
        return True
    except ModuleNotFoundError:
        return False
    except ImportError as ex:
        warnings.warn(str(ex))
        return False
