from sympy.assumptions import AppliedPredicate

def is_literal(expr):
    """
    Returns True if expr is a literal, else False.

    Examples
    ========

    >>> from sympy import Or, Q
    >>> from sympy.abc import A, B
    >>> from sympy.logic.boolalg import is_literal
    >>> is_literal(A)
    True
    >>> is_literal(~A)
    True
    >>> is_literal(Q.zero(A))
    True
    >>> is_literal(A + B)
    True
    >>> is_literal(Or(A, B))
    False

    """
    from sympy.assumptions import AppliedPredicate

    if isinstance(expr, Not):
        return is_literal(expr.args[0])
    elif expr in (True, False) or isinstance(expr, AppliedPredicate) or expr.is_Atom:
        return True
    elif not isinstance(expr, BooleanFunction) and all(
            (isinstance(expr, AppliedPredicate) or a.is_Atom) for a in expr.args):
        return True
    return False
