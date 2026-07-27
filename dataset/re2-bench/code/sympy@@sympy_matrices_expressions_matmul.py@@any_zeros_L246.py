from sympy.core.mul import mul, Mul
from .special import ZeroMatrix, Identity, GenericIdentity, OneMatrix

def any_zeros(mul):
    if any(arg.is_zero or (arg.is_Matrix and arg.is_ZeroMatrix)
                       for arg in mul.args):
        matrices = [arg for arg in mul.args if arg.is_Matrix]
        return ZeroMatrix(matrices[0].rows, matrices[-1].cols)
    return mul
