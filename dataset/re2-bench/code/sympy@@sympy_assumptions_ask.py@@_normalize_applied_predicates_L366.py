def _normalize_applied_predicates(expr):
    # Replace Q.gt(a,b) with Q.lt(b,a)
    expr = expr.replace(
        lambda e: hasattr(e, 'function') and e.function == Q.gt,
        lambda e: Q.lt(e.arguments[1], e.arguments[0])
    )
    # Replace Q.ge(a,b) with Q.le(b,a)
    expr = expr.replace(
        lambda e: hasattr(e, 'function') and e.function == Q.ge,
        lambda e: Q.le(e.arguments[1], e.arguments[0])
    )
    return expr
