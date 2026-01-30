def to_rgb(c):
    """
    Convert the :mpltype:`color` *c* to an RGB color tuple.

    If c has an alpha channel value specified, that is silently dropped.
    """
    return to_rgba(c)[:3]
