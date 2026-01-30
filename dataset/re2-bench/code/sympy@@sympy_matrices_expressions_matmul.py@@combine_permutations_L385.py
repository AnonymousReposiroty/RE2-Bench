from sympy.core.mul import mul, Mul
from .permutation import PermutationMatrix

def combine_permutations(mul):
    """Refine products of permutation matrices as the products of cycles.
    """
    args = mul.args
    l = len(args)
    if l < 2:
        return mul

    result = [args[0]]
    for i in range(1, l):
        A = result[-1]
        B = args[i]
        if isinstance(A, PermutationMatrix) and \
            isinstance(B, PermutationMatrix):
            cycle_1 = A.args[0]
            cycle_2 = B.args[0]
            result[-1] = PermutationMatrix(cycle_1 * cycle_2)
        else:
            result.append(B)

    return MatMul(*result)
