from sympy.core.sympify import _sympy_converter, _sympify, sympify

def is_nnf(expr, simplified=True):
    """
    Checks if ``expr`` is in Negation Normal Form (NNF).

    A logical expression is in NNF if it
    contains only :py:class:`~.And`, :py:class:`~.Or` and :py:class:`~.Not`,
    and :py:class:`~.Not` is applied only to literals.
    If ``simplified`` is ``True``, checks if result contains no redundant clauses.

    Examples
    ========

    >>> from sympy.abc import A, B, C
    >>> from sympy.logic.boolalg import Not, is_nnf
    >>> is_nnf(A & B | ~C)
    True
    >>> is_nnf((A | ~A) & (B | C))
    False
    >>> is_nnf((A | ~A) & (B | C), False)
    True
    >>> is_nnf(Not(A & B) | C)
    False
    >>> is_nnf((A >> B) & (B >> A))
    False

    """

    expr = sympify(expr)
    if is_literal(expr):
        return True

    stack = [expr]

    while stack:
        expr = stack.pop()
        if expr.func in (And, Or):
            if simplified:
                args = expr.args
                for arg in args:
                    if Not(arg) in args:
                        return False
            stack.extend(expr.args)

        elif not is_literal(expr):
            return False

    return True
