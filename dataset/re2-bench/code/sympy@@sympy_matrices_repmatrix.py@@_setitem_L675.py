from collections import defaultdict
from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.utilities.iterables import is_sequence
from sympy.utilities.misc import filldedent, as_int
from .exceptions import ShapeError, NonSquareMatrixError, NonInvertibleMatrixError
from .matrixbase import classof, MatrixBase

class MutableRepMatrix(RepMatrix):
    """Mutable matrix based on DomainMatrix as the internal representation"""
    is_zero = False

    @classmethod
    def _new(cls, *args, copy=True, **kwargs):
        if copy is False:
            if len(args) != 3:
                raise TypeError("'copy=False' requires a matrix be initialized as rows,cols,[list]")
            rows, cols, flat_list = args
        else:
            rows, cols, flat_list = cls._handle_creation_inputs(*args, **kwargs)
            flat_list = list(flat_list)
        rep = cls._flat_list_to_DomainMatrix(rows, cols, flat_list)
        return cls._fromrep(rep)

    @classmethod
    def _fromrep(cls, rep):
        obj = super().__new__(cls)
        obj.rows, obj.cols = rep.shape
        obj._rep = rep
        return obj

    def _setitem(self, key, value):
        """Helper to set value at location given by key.

        Examples
        ========

        >>> from sympy import Matrix, I, zeros, ones
        >>> m = Matrix(((1, 2+I), (3, 4)))
        >>> m
        Matrix([
        [1, 2 + I],
        [3,     4]])
        >>> m[1, 0] = 9
        >>> m
        Matrix([
        [1, 2 + I],
        [9,     4]])
        >>> m[1, 0] = [[0, 1]]

        To replace row r you assign to position r*m where m
        is the number of columns:

        >>> M = zeros(4)
        >>> m = M.cols
        >>> M[3*m] = ones(1, m)*2; M
        Matrix([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [2, 2, 2, 2]])

        And to replace column c you can assign to position c:

        >>> M[2] = ones(m, 1)*4; M
        Matrix([
        [0, 0, 4, 0],
        [0, 0, 4, 0],
        [0, 0, 4, 0],
        [2, 2, 4, 2]])
        """
        is_slice = isinstance(key, slice)
        i, j = key = self.key2ij(key)
        is_mat = isinstance(value, MatrixBase)
        if isinstance(i, slice) or isinstance(j, slice):
            if isinstance(value, MatrixBase):
                self.copyin_matrix(key, value)
                return
            if not isinstance(value, Expr) and is_sequence(value):
                self.copyin_list(key, value)
                return
            raise ValueError('unexpected value: %s' % value)
        else:
            if not is_mat and (not isinstance(value, Basic)) and is_sequence(value):
                value = self._new(value)
                is_mat = True
            if isinstance(value, MatrixBase):
                if is_slice:
                    key = (slice(*divmod(i, self.cols)), slice(*divmod(j, self.cols)))
                else:
                    key = (slice(i, i + value.rows), slice(j, j + value.cols))
                self.copyin_matrix(key, value)
            else:
                return (i, j, self._sympify(value))
            return

    def copyin_list(self, key, value):
        """Copy in elements from a list.

        Parameters
        ==========

        key : slice
            The section of this matrix to replace.
        value : iterable
            The iterable to copy values from.

        Examples
        ========

        >>> from sympy import eye
        >>> I = eye(3)
        >>> I[:2, 0] = [1, 2] # col
        >>> I
        Matrix([
        [1, 0, 0],
        [2, 1, 0],
        [0, 0, 1]])
        >>> I[1, :2] = [[3, 4]]
        >>> I
        Matrix([
        [1, 0, 0],
        [3, 4, 0],
        [0, 0, 1]])

        See Also
        ========

        copyin_matrix
        """
        if not is_sequence(value):
            raise TypeError('`value` must be an ordered iterable, not %s.' % type(value))
        return self.copyin_matrix(key, type(self)(value))

    def copyin_matrix(self, key, value):
        """Copy in values from a matrix into the given bounds.

        Parameters
        ==========

        key : slice
            The section of this matrix to replace.
        value : Matrix
            The matrix to copy values from.

        Examples
        ========

        >>> from sympy import Matrix, eye
        >>> M = Matrix([[0, 1], [2, 3], [4, 5]])
        >>> I = eye(3)
        >>> I[:3, :2] = M
        >>> I
        Matrix([
        [0, 1, 0],
        [2, 3, 0],
        [4, 5, 1]])
        >>> I[0, 1] = M
        >>> I
        Matrix([
        [0, 0, 1],
        [2, 2, 3],
        [4, 4, 5]])

        See Also
        ========

        copyin_list
        """
        rlo, rhi, clo, chi = self.key2bounds(key)
        shape = value.shape
        dr, dc = (rhi - rlo, chi - clo)
        if shape != (dr, dc):
            raise ShapeError(filldedent("The Matrix `value` doesn't have the same dimensions as the in sub-Matrix given by `key`."))
        for i in range(value.rows):
            for j in range(value.cols):
                self[i + rlo, j + clo] = value[i, j]

    @classmethod
    def _flat_list_to_DomainMatrix(cls, rows, cols, flat_list):
        elements_dod = defaultdict(dict)
        for n, element in enumerate(flat_list):
            if element != 0:
                i, j = divmod(n, cols)
                elements_dod[i][j] = element
        types = set(map(type, flat_list))
        rep = cls._dod_to_DomainMatrix(rows, cols, elements_dod, types)
        return rep
