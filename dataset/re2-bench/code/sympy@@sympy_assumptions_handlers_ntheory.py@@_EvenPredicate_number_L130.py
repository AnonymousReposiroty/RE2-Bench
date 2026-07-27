from sympy.core import Add, Basic, Expr, Float, Mul, Pow, S
from sympy.core.numbers import (ImaginaryUnit, Infinity, Integer, NaN,
    NegativeInfinity, NumberSymbol, Rational, int_valued)

def _EvenPredicate_number(expr, assumptions):
    # helper method
    if isinstance(expr, (float, Float)):
        if int_valued(expr):
            return None
        return False
    try:
        i = int(expr.round())
    except TypeError:
        return False
    if not (expr - i).equals(0):
        return False
    return i % 2 == 0
