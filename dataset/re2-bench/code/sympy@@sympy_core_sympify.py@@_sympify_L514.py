def _sympify(a):
    """
    Short version of :func:`~.sympify` for internal usage for ``__add__`` and
    ``__eq__`` methods where it is ok to allow some things (like Python
    integers and floats) in the expression. This excludes things (like strings)
    that are unwise to allow into such an expression.

    >>> from sympy import Integer
    >>> Integer(1) == 1
    True

    >>> Integer(1) == '1'
    False

    >>> from sympy.abc import x
    >>> x + 1
    x + 1

    >>> x + '1'
    Traceback (most recent call last):
    ...
    TypeError: unsupported operand type(s) for +: 'Symbol' and 'str'

    see: sympify

    """
    return sympify(a, strict=True)
