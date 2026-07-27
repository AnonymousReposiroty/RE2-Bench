def _decade_greater(x, base):
    """
    Return the smallest integer power of *base* that's greater than *x*.

    If *x* is negative, the exponent will be *smaller*.
    """
    if x < 0:
        return -_decade_less(-x, base)
    greater = _decade_greater_equal(x, base)
    if greater == x:
        greater *= base
    return greater
