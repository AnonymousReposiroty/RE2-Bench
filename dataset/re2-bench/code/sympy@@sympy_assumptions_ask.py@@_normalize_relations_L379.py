from sympy.core.relational import Eq, Ne, Gt, Lt, Ge, Le

def _normalize_relations(expr):
    def _filter(e):
        return isinstance(e, (Gt, Ge, Lt, Le, Eq, Ne))

    def _replace(e):
        a, b = e.args
        if isinstance(e, Gt):
            return Q.lt(b, a)
        elif isinstance(e, Ge):
            return Q.le(b, a)
        elif isinstance(e, Lt):
            return Q.lt(a, b)
        elif isinstance(e, Le):
            return Q.le(a, b)
        elif isinstance(e, Eq):
            return Q.eq(a, b)
        elif isinstance(e, Ne):
            return Q.ne(a, b)
        return e

    return expr.replace(_filter, _replace)
