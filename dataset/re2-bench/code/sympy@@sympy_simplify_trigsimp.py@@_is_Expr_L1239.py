from sympy.core import (sympify, Basic, S, Expr, factor_terms,
                        Mul, Add, bottom_up)
from sympy.core.function import (count_ops, _mexpand, FunctionClass, expand,
                                 expand_mul, _coeff_isneg, Derivative)

def _is_Expr(e):
    """_eapply helper to tell whether ``e`` and all its args
    are Exprs."""
    if isinstance(e, Derivative):
        return _is_Expr(e.expr)
    if not isinstance(e, Expr):
        return False
    return all(_is_Expr(i) for i in e.args)
