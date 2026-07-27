def has_variety(seq):
    """Return True if there are any different elements in ``seq``.

    Examples
    ========

    >>> from sympy import has_variety

    >>> has_variety((1, 2, 1))
    True
    >>> has_variety((1, 1, 1))
    False
    """
    for i, s in enumerate(seq):
        if i == 0:
            sentinel = s
        else:
            if s != sentinel:
                return True
    return False
