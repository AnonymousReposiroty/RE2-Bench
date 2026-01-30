from sympy.core.function import Function

def precedence_Mul(item):
    from sympy.core.function import Function
    if any(hasattr(arg, 'precedence') and isinstance(arg, Function) and
           arg.precedence < PRECEDENCE["Mul"] for arg in item.args):
        return PRECEDENCE["Mul"]

    if item.could_extract_minus_sign():
        return PRECEDENCE["Add"]
    return PRECEDENCE["Mul"]
