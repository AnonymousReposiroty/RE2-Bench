from sympy.core import Add, Basic, Expr, Mul, Pow, S

def _NegativePredicate_number(expr, assumptions):
    r, i = expr.as_real_imag()

    if r == S.NaN or i == S.NaN:
        return None

    # If the imaginary part can symbolically be shown to be zero then
    # we just evaluate the real part; otherwise we evaluate the imaginary
    # part to see if it actually evaluates to zero and if it does then
    # we make the comparison between the real part and zero.
    if not i:
        r = r.evalf(2)
        if r._prec != 1:
            return r < 0
    else:
        i = i.evalf(2)
        if i._prec != 1:
            if i != 0:
                return False
            r = r.evalf(2)
            if r._prec != 1:
                return r < 0
