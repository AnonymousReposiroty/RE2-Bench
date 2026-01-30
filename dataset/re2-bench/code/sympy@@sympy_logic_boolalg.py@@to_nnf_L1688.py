from sympy.simplify.simplify import simplify
from sympy.simplify.simplify import simplify

def to_nnf(expr, simplify=True, form=None):
    """
    Converts ``expr`` to Negation Normal Form (NNF).

    A logical expression is in NNF if it
    contains only :py:class:`~.And`, :py:class:`~.Or` and :py:class:`~.Not`,
    and :py:class:`~.Not` is applied only to literals.
    If ``simplify`` is ``True``, the result contains no redundant clauses.

    Parameters
    ==========

    expr : boolean expression
        The expression to convert to NNF.
    simplify : bool, optional
        If True, simplify the result. Default is True.
    form : str, optional
        Target form hint: 'cnf' for conjunctive normal form bias,
        'dnf' for disjunctive normal form bias, or None (default).
        This hint optimizes XOR conversions.

    Examples
    ========

    >>> from sympy.abc import A, B, C, D
    >>> from sympy.logic.boolalg import Not, Equivalent, to_nnf
    >>> to_nnf(Not((~A & ~B) | (C & D)))
    (A | B) & (~C | ~D)
    >>> to_nnf(Equivalent(A >> B, B >> A))
    (A | ~B | (A & ~B)) & (B | ~A | (B & ~A))

    """
    if is_nnf(expr, simplify):
        return expr
    return expr.to_nnf(simplify, form=form)
