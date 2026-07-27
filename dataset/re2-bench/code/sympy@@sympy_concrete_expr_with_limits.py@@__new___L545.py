from sympy.core.expr import Expr
from sympy.concrete.summations import Sum

class AddWithLimits(ExprWithLimits):
    """Represents unevaluated oriented additions.
        Parent class for Integral and Sum.
    """
    __slots__ = ()

    def __new__(cls, function, *symbols, **assumptions):
        from sympy.concrete.summations import Sum
        pre = _common_new(cls, function, *symbols, discrete=issubclass(cls, Sum), **assumptions)
        if isinstance(pre, tuple):
            function, limits, orientation = pre
        else:
            return pre
        obj = Expr.__new__(cls, **assumptions)
        arglist = [orientation * function]
        arglist.extend(limits)
        obj._args = tuple(arglist)
        obj.is_commutative = function.is_commutative
        return obj
