from sympy.core import Add, Basic, Expr, Float, Mul, Pow, S
from sympy.ntheory import isprime

def _PrimePredicate_number(expr, assumptions):
    # helper method
    exact = not expr.atoms(Float)
    try:
        i = int(expr.round())
        if (expr - i).equals(0) is False:
            raise TypeError
    except TypeError:
        return False
    if exact:
        return isprime(i)
