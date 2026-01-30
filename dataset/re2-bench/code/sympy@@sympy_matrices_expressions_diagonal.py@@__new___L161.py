from sympy.core.sympify import _sympify
from sympy.matrices.expressions import MatrixExpr

class DiagMatrix(MatrixExpr):
    """
    Turn a vector into a diagonal matrix.
    """

    def __new__(cls, vector):
        vector = _sympify(vector)
        obj = MatrixExpr.__new__(cls, vector)
        shape = vector.shape
        dim = shape[1] if shape[0] == 1 else shape[0]
        if vector.shape[0] != 1:
            obj._iscolumn = True
        else:
            obj._iscolumn = False
        obj._shape = (dim, dim)
        obj._vector = vector
        return obj
