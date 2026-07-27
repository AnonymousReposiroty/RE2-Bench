from sympy.logic.boolalg import Or, And, Not, Xnor

class Literal:
    """
    The smallest element of a CNF object.

    Parameters
    ==========

    lit : Boolean expression

    is_Not : bool

    Examples
    ========

    >>> from sympy import Q
    >>> from sympy.assumptions.cnf import Literal
    >>> from sympy.abc import x
    >>> Literal(Q.even(x))
    Literal(Q.even(x), False)
    >>> Literal(~Q.even(x))
    Literal(Q.even(x), True)
    """

    def __new__(cls, lit, is_Not=False):
        if isinstance(lit, Not):
            lit = lit.args[0]
            is_Not = True
        elif isinstance(lit, (AND, OR, Literal)):
            return ~lit if is_Not else lit
        obj = super().__new__(cls)
        obj.lit = lit
        obj.is_Not = is_Not
        return obj
    __repr__ = __str__
