from sympy.core.mul import mul, Mul
from .matexpr import MatrixExpr
from .special import ZeroMatrix, Identity, GenericIdentity, OneMatrix

class MatMul(MatrixExpr, Mul):
    """
    A product of matrix expressions

    Examples
    ========

    >>> from sympy import MatMul, MatrixSymbol
    >>> A = MatrixSymbol('A', 5, 4)
    >>> B = MatrixSymbol('B', 4, 3)
    >>> C = MatrixSymbol('C', 3, 6)
    >>> MatMul(A, B, C)
    A*B*C
    """
    is_MatMul = True
    identity = GenericIdentity()

    def args_cnc(self, cset=False, warn=True, **kwargs):
        coeff_c = [x for x in self.args if x.is_commutative]
        coeff_nc = [x for x in self.args if not x.is_commutative]
        if cset:
            clen = len(coeff_c)
            coeff_c = set(coeff_c)
            if clen and warn and (len(coeff_c) != clen):
                raise ValueError('repeated commutative arguments: %s' % [ci for ci in coeff_c if list(self.args).count(ci) > 1])
        return [coeff_c, coeff_nc]
