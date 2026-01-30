from sympy.core import (
    S, Expr, Add, Tuple
)
from sympy.core.mul import Mul, _keep_coeff
from sympy.polys.rationaltools import together

def _symbolic_factor(expr, opt, method):
    """Helper function for :func:`_factor`. """
    if isinstance(expr, Expr):
        if hasattr(expr,'_eval_factor'):
            return expr._eval_factor()
        coeff, factors = _symbolic_factor_list(together(expr, fraction=opt['fraction']), opt, method)
        return _keep_coeff(coeff, _factors_product(factors))
    elif hasattr(expr, 'args'):
        return expr.func(*[_symbolic_factor(arg, opt, method) for arg in expr.args])
    elif hasattr(expr, '__iter__'):
        return expr.__class__([_symbolic_factor(arg, opt, method) for arg in expr])
    else:
        return expr
