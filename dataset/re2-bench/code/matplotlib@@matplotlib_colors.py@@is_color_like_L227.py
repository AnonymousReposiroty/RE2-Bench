def is_color_like(c):
    """Return whether *c* as a valid Matplotlib :mpltype:`color` specifier."""
    # Special-case nth color syntax because it cannot be parsed during setup.
    if _is_nth_color(c):
        return True
    try:
        to_rgba(c)
    except (TypeError, ValueError):
        return False
    else:
        return True
