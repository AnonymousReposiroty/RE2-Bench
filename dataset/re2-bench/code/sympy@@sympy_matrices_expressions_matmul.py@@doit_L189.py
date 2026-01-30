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

    def doit(self, **hints):
        deep = hints.get('deep', True)
        if deep:
            args = tuple((arg.doit(**hints) for arg in self.args))
        else:
            args = self.args
        expr = canonicalize(MatMul(*args))
        return expr
