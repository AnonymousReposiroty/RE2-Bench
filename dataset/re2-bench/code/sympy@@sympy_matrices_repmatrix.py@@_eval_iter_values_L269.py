from sympy.polys.matrices import DomainMatrix
from .matrixbase import classof, MatrixBase

class RepMatrix(MatrixBase):
    """Matrix implementation based on DomainMatrix as an internal representation.

    The RepMatrix class is a superclass for Matrix, ImmutableMatrix,
    SparseMatrix and ImmutableSparseMatrix which are the main usable matrix
    classes in SymPy. Most methods on this class are simply forwarded to
    DomainMatrix.
    """
    _rep: DomainMatrix

    def _eval_iter_values(self):
        rep = self._rep
        K = rep.domain
        values = rep.iter_values()
        if not K.is_EXRAW:
            values = map(K.to_sympy, values)
        return values
