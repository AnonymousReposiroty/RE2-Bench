from matplotlib import _api, cbook

def _validate_minor_tick_ndivs(n):
    """
    Validate ndiv parameter related to the minor ticks.
    It controls the number of minor ticks to be placed between
    two major ticks.
    """

    if cbook._str_lower_equal(n, 'auto'):
        return n
    try:
        n = _validate_int_greaterequal0(n)
        return n
    except (RuntimeError, ValueError):
        pass

    raise ValueError("'tick.minor.ndivs' must be 'auto' or non-negative int")
