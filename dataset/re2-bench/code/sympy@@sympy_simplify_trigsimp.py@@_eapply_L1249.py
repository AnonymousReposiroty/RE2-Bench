from sympy.core import (sympify, Basic, S, Expr, factor_terms,
                        Mul, Add, bottom_up)

def _eapply(func, e, cond=None):
    """Apply ``func`` to ``e`` if all args are Exprs else only
    apply it to those args that *are* Exprs."""
    if not isinstance(e, Expr):
        return e
    if _is_Expr(e) or not e.args:
        return func(e)
    return e.func(*[
        _eapply(func, ei) if (cond is None or cond(ei)) else ei
        for ei in e.args])
