from sympy.concrete.expr_with_limits import AddWithLimits
from sympy.core.containers import Tuple
from sympy.core.expr import Expr

class Integral(AddWithLimits):
    """Represents unevaluated integral."""
    __slots__ = ()
    args: tuple[Expr, Tuple]

    def _eval_is_zero(self):
        if self.function.is_zero:
            return True
        got_none = False
        for l in self.limits:
            if len(l) == 3:
                z = l[1] == l[2] or (l[1] - l[2]).is_zero
                if z:
                    return True
                elif z is None:
                    got_none = True
        free = self.function.free_symbols
        for xab in self.limits:
            if len(xab) == 1:
                free.add(xab[0])
                continue
            if len(xab) == 2 and xab[0] not in free:
                if xab[1].is_zero:
                    return True
                elif xab[1].is_zero is None:
                    got_none = True
            free.discard(xab[0])
            for i in xab[1:]:
                free.update(i.free_symbols)
        if self.function.is_zero is False and got_none is False:
            return False
