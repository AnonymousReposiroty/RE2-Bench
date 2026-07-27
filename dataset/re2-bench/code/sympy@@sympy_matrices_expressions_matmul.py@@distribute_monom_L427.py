from sympy.core.mul import mul, Mul
from .matadd import MatAdd

def distribute_monom(mul):
    """
    Simplify MatMul expressions but distributing
    rational term to MatMul.

    e.g. 2*(A+B) -> 2*A + 2*B
    """
    args = mul.args
    if len(args) == 2:
        from .matadd import MatAdd
        if args[0].is_MatAdd and args[1].is_Rational:
            return MatAdd(*[MatMul(mat, args[1]).doit() for mat in args[0].args])
        if args[1].is_MatAdd and args[0].is_Rational:
            return MatAdd(*[MatMul(args[0], mat).doit() for mat in args[1].args])
    return mul
