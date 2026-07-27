from .singleton import S
from sympy.core.expr import Expr
from .numbers import Float, Rational, Integer, AlgebraicNumber, Number
from .numbers import Float, Number

def pure_complex(v: Expr, or_real=False) -> tuple[Number, Number] | None:
    """Return a and b if v matches a + I*b where b is not zero and
    a and b are Numbers, else None. If `or_real` is True then 0 will
    be returned for `b` if `v` is a real number.

    Examples
    ========

    >>> from sympy.core.evalf import pure_complex
    >>> from sympy import sqrt, I, S
    >>> a, b, surd = S(2), S(3), sqrt(2)
    >>> pure_complex(a)
    >>> pure_complex(a, or_real=True)
    (2, 0)
    >>> pure_complex(surd)
    >>> pure_complex(a + b*I)
    (2, 3)
    >>> pure_complex(I)
    (0, 1)
    """
    h, t = v.as_coeff_Add()
    if t:
        c, i = t.as_coeff_Mul()
        if i is S.ImaginaryUnit:
            return h, c
    elif or_real:
        return h, S.Zero
    return None
