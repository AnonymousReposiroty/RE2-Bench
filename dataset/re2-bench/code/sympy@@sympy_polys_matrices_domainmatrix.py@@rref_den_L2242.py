from ..domains import Domain
from .ddm import DDM
from .sdm import SDM
from .dfm import DFM
from .rref import _dm_rref, _dm_rref_den

class DomainMatrix:
    """
    Associate Matrix with :py:class:`~.Domain`

    Explanation
    ===========

    DomainMatrix uses :py:class:`~.Domain` for its internal representation
    which makes it faster than the SymPy Matrix class (currently) for many
    common operations, but this advantage makes it not entirely compatible
    with Matrix. DomainMatrix are analogous to numpy arrays with "dtype".
    In the DomainMatrix, each element has a domain such as :ref:`ZZ`
    or  :ref:`QQ(a)`.


    Examples
    ========

    Creating a DomainMatrix from the existing Matrix class:

    >>> from sympy import Matrix
    >>> from sympy.polys.matrices import DomainMatrix
    >>> Matrix1 = Matrix([
    ...    [1, 2],
    ...    [3, 4]])
    >>> A = DomainMatrix.from_Matrix(Matrix1)
    >>> A
    DomainMatrix({0: {0: 1, 1: 2}, 1: {0: 3, 1: 4}}, (2, 2), ZZ)

    Directly forming a DomainMatrix:

    >>> from sympy import ZZ
    >>> from sympy.polys.matrices import DomainMatrix
    >>> A = DomainMatrix([
    ...    [ZZ(1), ZZ(2)],
    ...    [ZZ(3), ZZ(4)]], (2, 2), ZZ)
    >>> A
    DomainMatrix([[1, 2], [3, 4]], (2, 2), ZZ)

    See Also
    ========

    DDM
    SDM
    Domain
    Poly

    """
    rep: SDM | DDM | DFM
    shape: tuple[int, int]
    domain: Domain

    def rref_den(self, *, method='auto', keep_domain=True):
        """
        Returns reduced-row echelon form with denominator and list of pivots.

        Requires exact division in the ground domain (``exquo``).

        Examples
        ========

        >>> from sympy import ZZ, QQ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix([
        ...     [ZZ(2), ZZ(-1), ZZ(0)],
        ...     [ZZ(-1), ZZ(2), ZZ(-1)],
        ...     [ZZ(0), ZZ(0), ZZ(2)]], (3, 3), ZZ)

        >>> A_rref, denom, pivots = A.rref_den()
        >>> A_rref
        DomainMatrix([[6, 0, 0], [0, 6, 0], [0, 0, 6]], (3, 3), ZZ)
        >>> denom
        6
        >>> pivots
        (0, 1, 2)
        >>> A_rref.to_field() / denom
        DomainMatrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]], (3, 3), QQ)
        >>> A_rref.to_field() / denom == A.convert_to(QQ).rref()[0]
        True

        Parameters
        ==========

        method : str, optional (default: 'auto')
            The method to use to compute the RREF. The default is ``'auto'``,
            which will attempt to choose the fastest method. The other options
            are:

            - ``A.rref(method='FF')`` uses fraction-free Gauss-Jordan
              elimination. Elimination is performed using exact division
              (``exquo``) to control the growth of the coefficients. In this
              case the current domain is always used for elimination and the
              result is always returned as a matrix over the current domain.
              This is most efficient for dense matrices or for matrices with
              simple denominators.

            - ``A.rref(method='CD')`` clears denominators before using
              fraction-free Gauss-Jordan elimination in the associated ring.
              The result will be converted back to the original domain unless
              ``keep_domain=False`` is passed in which case the result will be
              over the ring used for elimination. This is most efficient for
              dense matrices with very simple denominators.

            - ``A.rref(method='GJ')`` uses Gauss-Jordan elimination with
              division. If the domain is not a field then it will be converted
              to a field with :meth:`to_field` first and RREF will be computed
              by inverting the pivot elements in each row. The result is
              converted back to the original domain by clearing denominators
              unless ``keep_domain=False`` is passed in which case the result
              will be over the field used for elimination. This is most
              efficient for very sparse matrices or for matrices whose elements
              have complex denominators.

            - ``A.rref(method='GJ_dense')``, ``A.rref(method='FF_dense')``, and
              ``A.rref(method='CD_dense')`` are the same as the above methods
              except that the dense implementations of the algorithms are used.
              By default ``A.rref(method='auto')`` will usually choose the
              sparse implementations for RREF.

            Regardless of which algorithm is used the returned matrix will
            always have the same format (sparse or dense) as the input and if
            ``keep_domain=True`` its domain will always be the same as the
            input.

        keep_domain : bool, optional
            If True (the default), the domain of the returned matrix and
            denominator are the same as the domain of the input matrix. If
            False, the domain of the returned matrix might be changed to an
            associated ring or field if the algorithm used a different domain.
            This is useful for efficiency if the caller does not need the
            result to be in the original domain e.g. it avoids clearing
            denominators in the case of ``A.rref(method='GJ')``.

        Returns
        =======

        (DomainMatrix, scalar, list)
            Reduced-row echelon form, denominator and list of pivot indices.

        See Also
        ========

        rref
            RREF without denominator for field domains.
        sympy.polys.matrices.sdm.sdm_irref
            Sparse implementation of ``method='GJ'``.
        sympy.polys.matrices.sdm.sdm_rref_den
            Sparse implementation of ``method='FF'`` and ``method='CD'``.
        sympy.polys.matrices.dense.ddm_irref
            Dense implementation of ``method='GJ'``.
        sympy.polys.matrices.dense.ddm_irref_den
            Dense implementation of ``method='FF'`` and ``method='CD'``.
        clear_denoms
            Clear denominators from a matrix, used by ``method='CD'``.

        """
        return _dm_rref_den(self, method=method, keep_domain=keep_domain)
