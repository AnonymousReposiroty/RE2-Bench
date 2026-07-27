from typing import TYPE_CHECKING, overload
from collections.abc import Iterable, Sequence
from functools import reduce
from sympy.core import SympifyError, Add
from sympy.core.basic import Atom, Basic
from sympy.core.singleton import S
from sympy.printing.defaults import Printable
from collections.abc import Callable
from sympy.core.expr import Expr
from .utilities import _dotprodsimp, _simplify as _utilities_simplify
from .utilities import _get_intermediate_simp_bool
from .exceptions import (
    MatrixError, ShapeError, NonSquareMatrixError, NonInvertibleMatrixError,
)
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
from abc import ABCMeta, abstractmethod
from abc import abstractmethod
from typing_extensions import Self
from sympy.matrices.matrixbase import MatrixBase
from sympy.matrices.matrixbase import MatrixBase

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
    def _new(cls, rows: int, cols: int, mat: Sequence[SExpr], /, copy: bool=False) -> Self:
        ...

    @overload
    @classmethod
    def _new(cls, rows: int, cols: int, func: Callable[[int, int], SExpr], /) -> Self:
        ...

    @overload
    @classmethod
    def _new(cls, mat: Sequence[Sequence[SExpr]] | Self, /) -> Self:
        ...

    @overload
    @classmethod
    def _new(cls, /) -> Self:
        ...

    @overload
    @classmethod
    def _new(cls, elements: Sequence[SExpr], /) -> Self:
        ...

    @classmethod
    @abstractmethod
    def _new(cls, *args, **kwargs) -> Self:
        """`_new` must, at minimum, be callable as
        `_new(rows, cols, mat) where mat is a flat list of the
        elements of the matrix."""
        raise NotImplementedError('Subclasses must implement this.')

    @property
    def shape(self) -> tuple[int, int]:
        """The shape (dimensions) of the matrix as the 2-tuple (rows, cols).

        Examples
        ========

        >>> from sympy import zeros
        >>> M = zeros(2, 3)
        >>> M.shape
        (2, 3)
        >>> M.rows
        2
        >>> M.cols
        3
        """
        return (self.rows, self.cols)

    def _eval_matrix_mul(self, other: MatrixBase) -> Self:

        def entry(i, j):
            vec = [self[i, k] * other[k, j] for k in range(self.cols)]
            try:
                return Add(*vec)
            except (TypeError, SympifyError):
                return reduce(lambda a, b: a + b, vec)
        return self._new(self.rows, other.cols, entry)

    def _eval_scalar_mul(self, other: SExpr) -> Self:
        return self._new(self.rows, self.cols, lambda i, j: self[i, j] * other)

    @overload
    def multiply(self, other: Self, dotprodsimp: bool | None=None) -> Self:
        ...

    @overload
    def multiply(self, other: MatrixBase, dotprodsimp: bool | None=None) -> MatrixBase:
        ...

    @overload
    def multiply(self, other: Expr, dotprodsimp: bool | None=None) -> Expr:
        ...

    def multiply(self, other: MatrixBase | Expr, dotprodsimp: bool | None=None) -> MatrixBase | Expr:
        """Same as __mul__() but with optional simplification.

        Parameters
        ==========

        dotprodsimp : bool, optional
            Specifies whether intermediate term algebraic simplification is used
            during matrix multiplications to control expression blowup and thus
            speed up calculation. Default is off.
        """
        isimpbool = _get_intermediate_simp_bool(False, dotprodsimp)
        self, other, T = _unify_with_other(self, other)
        if isinstance(other, MatrixBase):
            if self.shape[1] != other.shape[0]:
                raise ShapeError(f'Matrix size mismatch: {self.shape} * {other.shape}.')
            m = self._eval_matrix_mul(other)
            if isimpbool:
                m = m._new(m.rows, m.cols, [_dotprodsimp(e) for e in m.flat()])
            return m
        elif T == 'possible_scalar':
            try:
                return self._eval_scalar_mul(other)
            except TypeError:
                return NotImplemented
        else:
            return NotImplemented
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
