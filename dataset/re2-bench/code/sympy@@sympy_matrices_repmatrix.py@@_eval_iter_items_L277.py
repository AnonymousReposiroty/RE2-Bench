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

    def _eval_iter_items(self):
        rep = self._rep
        K = rep.domain
        to_sympy = K.to_sympy
        items = rep.iter_items()
        if not K.is_EXRAW:
            items = ((i, to_sympy(v)) for i, v in items)
        return items
