from .mul import Mul, _keep_coeff
from .power import Pow
from .expr import Expr
from .numbers import Rational, Integer, Number, I, equal_valued
from .singleton import S

def decompose_power(expr: Expr) -> tuple[Expr, int]:
    """
    Decompose power into symbolic base and integer exponent.

    Examples
    ========

    >>> from sympy.core.exprtools import decompose_power
    >>> from sympy.abc import x, y
    >>> from sympy import exp

    >>> decompose_power(x)
    (x, 1)
    >>> decompose_power(x**2)
    (x, 2)
    >>> decompose_power(exp(2*y/3))
    (exp(y/3), 2)

    """
    base, exp = expr.as_base_exp()

    if exp.is_Number:
        if exp.is_Rational:
            if not exp.is_Integer:
                base = Pow(base, Rational(1, exp.q))  # type: ignore
            e = exp.p  # type: ignore
        else:
            base, e = expr, 1
    else:
        exp, tail = exp.as_coeff_Mul(rational=True)

        if exp is S.NegativeOne:
            base, e = Pow(base, tail), -1
        elif exp is not S.One:
            tail = _keep_coeff(Rational(1, exp.q), tail)
            base, e = Pow(base, tail), exp.p
        else:
            base, e = expr, 1

    return base, e
