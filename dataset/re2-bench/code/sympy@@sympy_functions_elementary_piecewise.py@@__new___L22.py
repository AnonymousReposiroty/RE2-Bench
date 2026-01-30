from sympy.core import S, diff, Tuple, Dummy, Mul
from sympy.core.basic import Basic, as_Basic
from sympy.logic.boolalg import (And, Boolean, distribute_and_over_or, Not,
    true, false, Or, ITE, simplify_logic, to_cnf, distribute_or_over_and)
from sympy.utilities.misc import filldedent, func_name

class ExprCondPair(Tuple):
    """Represents an expression, condition pair."""

    def __new__(cls, expr, cond):
        expr = as_Basic(expr)
        if cond == True:
            return Tuple.__new__(cls, expr, true)
        elif cond == False:
            return Tuple.__new__(cls, expr, false)
        elif isinstance(cond, Basic) and cond.has(Piecewise):
            cond = piecewise_fold(cond)
            if isinstance(cond, Piecewise):
                cond = cond.rewrite(ITE)
        if not isinstance(cond, Boolean):
            raise TypeError(filldedent('\n                Second argument must be a Boolean,\n                not `%s`' % func_name(cond)))
        return Tuple.__new__(cls, expr, cond)
