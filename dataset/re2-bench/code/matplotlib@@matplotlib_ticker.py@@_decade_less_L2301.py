def _decade_less(x, base):
    """
    Return the largest integer power of *base* that's less than *x*.

    If *x* is negative, the exponent will be *greater*.
    """
    if x < 0:
        return -_decade_greater(-x, base)
    less = _decade_less_equal(x, base)
    if less == x:
        less /= base
    return less
