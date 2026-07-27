from sympy.core.sympify import _sympy_converter, _sympify, sympify

def _is_form(expr, function1, function2):
    """
    Test whether or not an expression is of the required form.

    """
    expr = sympify(expr)

    vals = function1.make_args(expr) if isinstance(expr, function1) else [expr]
    for lit in vals:
        if isinstance(lit, function2):
            vals2 = function2.make_args(lit) if isinstance(lit, function2) else [lit]
            for l in vals2:
                if is_literal(l) is False:
                    return False
        elif is_literal(lit) is False:
            return False

    return True
