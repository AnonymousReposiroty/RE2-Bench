def interval_contains_open(interval, val):
    """
    Check, excluding endpoints, whether an interval includes a given value.

    Parameters
    ----------
    interval : (float, float)
        The endpoints of the interval.
    val : float
        Value to check is within interval.

    Returns
    -------
    bool
        Whether *val* is within the *interval*.
    """
    a, b = interval
    return a < val < b or a > val > b
