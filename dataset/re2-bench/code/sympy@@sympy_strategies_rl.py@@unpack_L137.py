def unpack(expr):
    """ Rule to unpack singleton args

    >>> from sympy.strategies import unpack
    >>> from sympy import Basic, S
    >>> unpack(Basic(S(2)))
    2
    """
    if len(expr.args) == 1:
        return expr.args[0]
    else:
        return expr
