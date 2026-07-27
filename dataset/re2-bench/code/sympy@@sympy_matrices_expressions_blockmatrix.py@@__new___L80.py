from sympy.core import Basic, Add, Mul, S
from sympy.utilities.iterables import is_sequence, sift
from sympy.utilities.misc import filldedent
from sympy.matrices.expressions.matexpr import MatrixExpr, MatrixElement
from sympy.matrices.immutable import ImmutableDenseMatrix
from sympy.matrices.immutable import ImmutableDenseMatrix

class BlockMatrix(MatrixExpr):
    """A BlockMatrix is a Matrix comprised of other matrices.

    The submatrices are stored in a SymPy Matrix object but accessed as part of
    a Matrix Expression

    >>> from sympy import (MatrixSymbol, BlockMatrix, symbols,
    ...     Identity, ZeroMatrix, block_collapse)
    >>> n,m,l = symbols('n m l')
    >>> X = MatrixSymbol('X', n, n)
    >>> Y = MatrixSymbol('Y', m, m)
    >>> Z = MatrixSymbol('Z', n, m)
    >>> B = BlockMatrix([[X, Z], [ZeroMatrix(m,n), Y]])
    >>> print(B)
    Matrix([
    [X, Z],
    [0, Y]])

    >>> C = BlockMatrix([[Identity(n), Z]])
    >>> print(C)
    Matrix([[I, Z]])

    >>> print(block_collapse(C*B))
    Matrix([[X, Z + Z*Y]])

    Some matrices might be comprised of rows of blocks with
    the matrices in each row having the same height and the
    rows all having the same total number of columns but
    not having the same number of columns for each matrix
    in each row. In this case, the matrix is not a block
    matrix and should be instantiated by Matrix.

    >>> from sympy import ones, Matrix
    >>> dat = [
    ... [ones(3,2), ones(3,3)*2],
    ... [ones(2,3)*3, ones(2,2)*4]]
    ...
    >>> BlockMatrix(dat)
    Traceback (most recent call last):
    ...
    ValueError:
    Although this matrix is comprised of blocks, the blocks do not fill
    the matrix in a size-symmetric fashion. To create a full matrix from
    these arguments, pass them directly to Matrix.
    >>> Matrix(dat)
    Matrix([
    [1, 1, 2, 2, 2],
    [1, 1, 2, 2, 2],
    [1, 1, 2, 2, 2],
    [3, 3, 3, 4, 4],
    [3, 3, 3, 4, 4]])

    See Also
    ========
    sympy.matrices.matrixbase.MatrixBase.irregular
    """

    def __new__(cls, *args, **kwargs):
        from sympy.matrices.immutable import ImmutableDenseMatrix
        isMat = lambda i: getattr(i, 'is_Matrix', False)
        if len(args) != 1 or not is_sequence(args[0]) or len({isMat(r) for r in args[0]}) != 1:
            raise ValueError(filldedent('\n                expecting a sequence of 1 or more rows\n                containing Matrices.'))
        rows = args[0] if args else []
        if not isMat(rows):
            if rows and isMat(rows[0]):
                rows = [rows]
            blocky = ok = len({len(r) for r in rows}) == 1
            if ok:
                for r in rows:
                    ok = len({i.rows for i in r}) == 1
                    if not ok:
                        break
                blocky = ok
                if ok:
                    for c in range(len(rows[0])):
                        ok = len({rows[i][c].cols for i in range(len(rows))}) == 1
                        if not ok:
                            break
            if not ok:
                ok = len({sum((i.cols for i in r)) for r in rows}) == 1
                if blocky and ok:
                    raise ValueError(filldedent('\n                        Although this matrix is comprised of blocks,\n                        the blocks do not fill the matrix in a\n                        size-symmetric fashion. To create a full matrix\n                        from these arguments, pass them directly to\n                        Matrix.'))
                raise ValueError(filldedent("\n                    When there are not the same number of rows in each\n                    row's matrices or there are not the same number of\n                    total columns in each row, the matrix is not a\n                    block matrix. If this matrix is known to consist of\n                    blocks fully filling a 2-D space then see\n                    Matrix.irregular."))
        mat = ImmutableDenseMatrix(rows, evaluate=False)
        obj = Basic.__new__(cls, mat)
        return obj
