def eliminate_implications(expr, form=None):
    """
    Change :py:class:`~.Implies` and :py:class:`~.Equivalent` into
    :py:class:`~.And`, :py:class:`~.Or`, and :py:class:`~.Not`.
    That is, return an expression that is equivalent to ``expr``, but has only
    ``&``, ``|``, and ``~`` as logical
    operators.

    Parameters
    ==========

    expr : boolean expression
        The expression to eliminate implications from.
    form : str, optional
        Target form hint: 'cnf' or 'dnf'. Passed to to_nnf for optimization.

    Examples
    ========

    >>> from sympy.logic.boolalg import Implies, Equivalent, \
         eliminate_implications
    >>> from sympy.abc import A, B, C
    >>> eliminate_implications(Implies(A, B))
    B | ~A
    >>> eliminate_implications(Equivalent(A, B))
    (A | ~B) & (B | ~A)
    >>> eliminate_implications(Equivalent(A, B, C))
    (A | ~C) & (B | ~A) & (C | ~B)

    """
    return to_nnf(expr, simplify=False, form=form)
