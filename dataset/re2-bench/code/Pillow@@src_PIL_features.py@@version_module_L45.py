def version_module(feature):
    """
    :param feature: The module to check for.
    :returns:
        The loaded version number as a string, or ``None`` if unknown or not available.
    :raises ValueError: If the module is not defined in this version of Pillow.
    """
    if not check_module(feature):
        return None

    module, ver = modules[feature]

    if ver is None:
        return None

    return getattr(__import__(module, fromlist=[ver]), ver)
