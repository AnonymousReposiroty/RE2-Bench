from typing import TYPE_CHECKING, overload
from sympy.core.basic import Atom, Basic
from sympy.core.sympify import sympify, _sympify
from sympy.core.singleton import S
from sympy.printing.defaults import Printable
import mpmath as mp
from collections.abc import Callable
from sympy.utilities.iterables import reshape
from sympy.core.expr import Expr
from sympy.polys.polytools import Poly
from sympy.utilities.iterables import flatten, is_sequence
from sympy.utilities.misc import as_int, filldedent
from .determinant import (
    _find_reasonable_pivot, _find_reasonable_pivot_naive,
    _adjugate, _charpoly, _cofactor, _cofactor_matrix, _per,
    _det, _det_bareiss, _det_berkowitz, _det_bird, _det_laplace, _det_LU,
    _minor, _minor_submatrix)
from .reductions import _is_echelon, _echelon_form, _rank, _rref
from .solvers import (
    _diagonal_solve, _lower_triangular_solve, _upper_triangular_solve,
    _cholesky_solve, _LDLsolve, _LUsolve, _QRsolve, _gauss_jordan_solve,
    _pinv_solve, _cramer_solve, _solve, _solve_least_squares)
from .inverse import (
    _pinv, _inv_ADJ, _inv_GE, _inv_LU, _inv_CH, _inv_LDL, _inv_QR,
    _inv, _inv_block)
from .subspaces import _columnspace, _nullspace, _rowspace, _orthogonalize
from .eigen import (
    _eigenvals, _eigenvects,
    _bidiagonalize, _bidiagonal_decomposition,
    _is_diagonalizable, _diagonalize,
    _is_positive_definite, _is_positive_semidefinite,
    _is_negative_definite, _is_negative_semidefinite, _is_indefinite,
    _jordan_form, _left_eigenvects, _singular_values)
from .decompositions import (
    _rank_decomposition, _cholesky, _LDLdecomposition,
    _LUdecomposition, _LUdecomposition_Simple, _LUdecompositionFF,
    _singular_value_decomposition, _QRdecomposition, _upper_hessenberg_decomposition)
from .graph import (
    _connected_components, _connected_components_decomposition,
    _strongly_connected_components, _strongly_connected_components_decomposition)
from typing import Literal, TypeVar, Iterator, Mapping, Any
from sympy.matrices.expressions.blockmatrix import BlockDiagMatrix, BlockMatrix
from sympy.matrices.matrixbase import MatrixBase
from sympy.matrices.matrixbase import MatrixBase
from sympy.matrices import SparseMatrix
from sympy.matrices import SparseMatrix
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.matrices.expressions.blockmatrix import BlockMatrix

class MatrixBase(Printable):
    """All common matrix operations including basic arithmetic, shaping,
    and special matrices like `zeros`, and `eye`."""
    _op_priority = 10.01
    __array_priority__ = 11
    is_Matrix = True
    _class_priority = 3
    zero = S.Zero
    one = S.One
    _diff_wrt: bool = True
    _simplify = None
    if TYPE_CHECKING:

        @property
        def rows(self) -> int:
            ...

        @property
        def cols(self) -> int:
            ...

    @overload
    @classmethod
    def _sympify(cls, expr: SExpr, /) -> Expr:
        ...

    @overload
    @classmethod
    def _sympify(cls, expr: Poly, /) -> Poly:
        ...

    @classmethod
    def _sympify(cls, expr: SExpr | Poly, /) -> Expr | Poly:
        if isinstance(expr, Poly):
            return expr
        return sympify(expr)
    _find_reasonable_pivot.__doc__ = _find_reasonable_pivot.__doc__
    _find_reasonable_pivot_naive.__doc__ = _find_reasonable_pivot_naive.__doc__
    _eval_det_bareiss.__doc__ = _det_bareiss.__doc__
    _eval_det_berkowitz.__doc__ = _det_berkowitz.__doc__
    _eval_det_bird.__doc__ = _det_bird.__doc__
    _eval_det_laplace.__doc__ = _det_laplace.__doc__
    _eval_det_lu.__doc__ = _det_LU.__doc__
    _eval_determinant.__doc__ = _det.__doc__
    adjugate.__doc__ = _adjugate.__doc__
    charpoly.__doc__ = _charpoly.__doc__
    cofactor.__doc__ = _cofactor.__doc__
    cofactor_matrix.__doc__ = _cofactor_matrix.__doc__
    det.__doc__ = _det.__doc__
    per.__doc__ = _per.__doc__
    minor.__doc__ = _minor.__doc__
    minor_submatrix.__doc__ = _minor_submatrix.__doc__
    echelon_form.__doc__ = _echelon_form.__doc__
    is_echelon.__doc__ = _is_echelon.__doc__
    rank.__doc__ = _rank.__doc__
    rref.__doc__ = _rref.__doc__
    columnspace.__doc__ = _columnspace.__doc__
    nullspace.__doc__ = _nullspace.__doc__
    rowspace.__doc__ = _rowspace.__doc__
    orthogonalize.__doc__ = _orthogonalize.__doc__
    orthogonalize = classmethod(orthogonalize)
    eigenvals.__doc__ = _eigenvals.__doc__
    eigenvects.__doc__ = _eigenvects.__doc__
    is_diagonalizable.__doc__ = _is_diagonalizable.__doc__
    diagonalize.__doc__ = _diagonalize.__doc__
    is_positive_definite.__doc__ = _is_positive_definite.__doc__
    is_positive_semidefinite.__doc__ = _is_positive_semidefinite.__doc__
    is_negative_definite.__doc__ = _is_negative_definite.__doc__
    is_negative_semidefinite.__doc__ = _is_negative_semidefinite.__doc__
    is_indefinite.__doc__ = _is_indefinite.__doc__
    jordan_form.__doc__ = _jordan_form.__doc__
    left_eigenvects.__doc__ = _left_eigenvects.__doc__
    singular_values.__doc__ = _singular_values.__doc__
    bidiagonalize.__doc__ = _bidiagonalize.__doc__
    bidiagonal_decomposition.__doc__ = _bidiagonal_decomposition.__doc__

    @classmethod
    def _handle_ndarray(cls, arg: Any) -> tuple[int, int, list[Expr]]:
        arr = arg.__array__()
        if len(arr.shape) == 2:
            rows, cols = (arr.shape[0], arr.shape[1])
            flat_list = [cls._sympify(i) for i in arr.ravel()]
            return (rows, cols, flat_list)
        elif len(arr.shape) == 1:
            flat_list = [cls._sympify(i) for i in arr]
            return (arr.shape[0], 1, flat_list)
        else:
            raise NotImplementedError('SymPy supports just 1D and 2D matrices')

    @classmethod
    def _handle_creation_inputs(cls, *args, **kwargs):
        """Return the number of rows, cols and flat matrix elements.

        Examples
        ========

        >>> from sympy import Matrix, I

        Matrix can be constructed as follows:

        * from a nested list of iterables

        >>> Matrix( ((1, 2+I), (3, 4)) )
        Matrix([
        [1, 2 + I],
        [3,     4]])

        * from un-nested iterable (interpreted as a column)

        >>> Matrix( [1, 2] )
        Matrix([
        [1],
        [2]])

        * from un-nested iterable with dimensions

        >>> Matrix(1, 2, [1, 2] )
        Matrix([[1, 2]])

        * from no arguments (a 0 x 0 matrix)

        >>> Matrix()
        Matrix(0, 0, [])

        * from a rule

        >>> Matrix(2, 2, lambda i, j: i/(j + 1) )
        Matrix([
        [0,   0],
        [1, 1/2]])

        See Also
        ========
        irregular - filling a matrix with irregular blocks
        """
        from sympy.matrices import SparseMatrix
        from sympy.matrices.expressions.matexpr import MatrixSymbol
        from sympy.matrices.expressions.blockmatrix import BlockMatrix
        flat_list = None
        if len(args) == 1:
            [arg1] = args
            if isinstance(arg1, SparseMatrix):
                return (arg1.rows, arg1.cols, flatten(arg1.tolist()))
            elif isinstance(arg1, MatrixBase):
                return (arg1.rows, arg1.cols, arg1.flat())
            elif isinstance(arg1, Basic) and arg1.is_Matrix:
                return (arg1.rows, arg1.cols, arg1.as_explicit().flat())
            elif isinstance(args[0], mp.matrix):
                M = args[0]
                flat_list = [cls._sympify(x) for x in M]
                return (M.rows, M.cols, flat_list)
            elif hasattr(args[0], '__array__'):
                return cls._handle_ndarray(args[0])
            elif is_sequence(args[0]) and (not isinstance(args[0], DeferredVector)):
                dat = list(args[0])
                ismat = lambda i: isinstance(i, MatrixBase) and (evaluate or isinstance(i, (BlockMatrix, MatrixSymbol)))
                raw = lambda i: is_sequence(i) and (not ismat(i))
                evaluate = kwargs.get('evaluate', True)
                if evaluate:

                    def make_explicit(x):
                        """make Block and Symbol explicit"""
                        if isinstance(x, BlockMatrix):
                            return x.as_explicit()
                        elif isinstance(x, MatrixSymbol) and all((_.is_Integer for _ in x.shape)):
                            return x.as_explicit()
                        else:
                            return x

                    def make_explicit_row(row):
                        if isinstance(row, (list, tuple)):
                            return [make_explicit(x) for x in row]
                        else:
                            return make_explicit(row)
                    if isinstance(dat, (list, tuple)):
                        dat = [make_explicit_row(row) for row in dat]
                if len(dat) == 0:
                    rows = cols = 0
                    flat_list = []
                elif all((raw(i) for i in dat)) and len(dat[0]) == 0:
                    if not all((len(i) == 0 for i in dat)):
                        raise ValueError('mismatched dimensions')
                    rows = len(dat)
                    cols = 0
                    flat_list = []
                elif not any((raw(i) or ismat(i) for i in dat)):
                    flat_list = [cls._sympify(i) for i in dat]
                    rows = len(flat_list)
                    cols = 1 if rows else 0
                elif evaluate and all((ismat(i) for i in dat)):
                    ncol = {i.cols for i in dat if any(i.shape)}
                    if ncol:
                        if len(ncol) != 1:
                            raise ValueError('mismatched dimensions')
                        flat_list = [_ for i in dat for r in i.tolist() for _ in r]
                        cols = ncol.pop()
                        rows = len(flat_list) // cols
                    else:
                        rows = cols = 0
                        flat_list = []
                elif evaluate and any((ismat(i) for i in dat)):
                    ncol = set()
                    flat_list = []
                    for i in dat:
                        if ismat(i):
                            flat_list.extend([k for j in i.tolist() for k in j])
                            if any(i.shape):
                                ncol.add(i.cols)
                        elif raw(i):
                            if i:
                                ncol.add(len(i))
                                flat_list.extend([cls._sympify(ij) for ij in i])
                        else:
                            ncol.add(1)
                            flat_list.append(i)
                        if len(ncol) > 1:
                            raise ValueError('mismatched dimensions')
                    cols = ncol.pop()
                    rows = len(flat_list) // cols
                else:
                    flat_list = []
                    ncol = set()
                    rows = cols = 0
                    for row in dat:
                        if not is_sequence(row) and (not getattr(row, 'is_Matrix', False)):
                            raise ValueError('expecting list of lists')
                        if hasattr(row, '__array__'):
                            if 0 in row.shape:
                                continue
                        if evaluate and all((ismat(i) for i in row)):
                            r, c, flatT = cls._handle_creation_inputs([i.T for i in row])
                            T = reshape(flatT, [c])
                            flat = [T[i][j] for j in range(c) for i in range(r)]
                            r, c = (c, r)
                        else:
                            r = 1
                            if getattr(row, 'is_Matrix', False):
                                c = 1
                                flat = [row]
                            else:
                                c = len(row)
                                flat = [cls._sympify(i) for i in row]
                        ncol.add(c)
                        if len(ncol) > 1:
                            raise ValueError('mismatched dimensions')
                        flat_list.extend(flat)
                        rows += r
                    cols = ncol.pop() if ncol else 0
        elif len(args) == 3:
            rows = as_int(args[0])
            cols = as_int(args[1])
            if rows < 0 or cols < 0:
                raise ValueError('Cannot create a {} x {} matrix. Both dimensions must be positive'.format(rows, cols))
            if len(args) == 3 and isinstance(args[2], Callable):
                op = args[2]
                flat_list = []
                for i in range(rows):
                    flat_list.extend([cls._sympify(op(cls._sympify(i), cls._sympify(j))) for j in range(cols)])
            elif len(args) == 3 and is_sequence(args[2]):
                flat_list = args[2]
                if len(flat_list) != rows * cols:
                    raise ValueError('List length should be equal to rows*columns')
                flat_list = [cls._sympify(i) for i in flat_list]
        elif len(args) == 0:
            rows = cols = 0
            flat_list = []
        else:
            raise TypeError('Need 0, 1 or 3 arguments')
        if flat_list is None:
            raise TypeError(filldedent('\n                Data type not understood; expecting list of lists\n                or lists of values.'))
        return (rows, cols, flat_list)
    _sage_ = Basic._sage_
    rank_decomposition.__doc__ = _rank_decomposition.__doc__
    cholesky.__doc__ = _cholesky.__doc__
    LDLdecomposition.__doc__ = _LDLdecomposition.__doc__
    LUdecomposition.__doc__ = _LUdecomposition.__doc__
    LUdecomposition_Simple.__doc__ = _LUdecomposition_Simple.__doc__
    LUdecompositionFF.__doc__ = _LUdecompositionFF.__doc__
    singular_value_decomposition.__doc__ = _singular_value_decomposition.__doc__
    QRdecomposition.__doc__ = _QRdecomposition.__doc__
    upper_hessenberg_decomposition.__doc__ = _upper_hessenberg_decomposition.__doc__
    diagonal_solve.__doc__ = _diagonal_solve.__doc__
    lower_triangular_solve.__doc__ = _lower_triangular_solve.__doc__
    upper_triangular_solve.__doc__ = _upper_triangular_solve.__doc__
    cholesky_solve.__doc__ = _cholesky_solve.__doc__
    LDLsolve.__doc__ = _LDLsolve.__doc__
    LUsolve.__doc__ = _LUsolve.__doc__
    QRsolve.__doc__ = _QRsolve.__doc__
    gauss_jordan_solve.__doc__ = _gauss_jordan_solve.__doc__
    pinv_solve.__doc__ = _pinv_solve.__doc__
    cramer_solve.__doc__ = _cramer_solve.__doc__
    solve.__doc__ = _solve.__doc__
    solve_least_squares.__doc__ = _solve_least_squares.__doc__
    pinv.__doc__ = _pinv.__doc__
    inverse_ADJ.__doc__ = _inv_ADJ.__doc__
    inverse_GE.__doc__ = _inv_GE.__doc__
    inverse_LU.__doc__ = _inv_LU.__doc__
    inverse_CH.__doc__ = _inv_CH.__doc__
    inverse_LDL.__doc__ = _inv_LDL.__doc__
    inverse_QR.__doc__ = _inv_QR.__doc__
    inverse_BLOCK.__doc__ = _inv_block.__doc__
    inv.__doc__ = _inv.__doc__
    connected_components.__doc__ = _connected_components.__doc__
    connected_components_decomposition.__doc__ = _connected_components_decomposition.__doc__
    strongly_connected_components.__doc__ = _strongly_connected_components.__doc__
    strongly_connected_components_decomposition.__doc__ = _strongly_connected_components_decomposition.__doc__
