from sympy.core.mul import mul, Mul
from .special import ZeroMatrix, Identity, GenericIdentity, OneMatrix

def combine_one_matrices(mul):
    """
    Combine products of OneMatrix

    e.g. OneMatrix(2, 3) * OneMatrix(3, 4) -> 3 * OneMatrix(2, 4)
    """
    factor, args = mul.as_coeff_matrices()
    new_args = [args[0]]

    for B in args[1:]:
        A = new_args[-1]
        if not isinstance(A, OneMatrix) or not isinstance(B, OneMatrix):
            new_args.append(B)
            continue
        new_args.pop()
        new_args.append(OneMatrix(A.shape[0], B.shape[1]))
        factor *= A.shape[1]

    return newmul(factor, *new_args)
