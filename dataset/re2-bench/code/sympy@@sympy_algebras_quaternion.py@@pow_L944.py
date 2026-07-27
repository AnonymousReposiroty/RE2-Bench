from typing import TYPE_CHECKING, overload
from sympy.core.expr import Expr
from sympy.utilities.misc import as_int

class Quaternion(Expr):
    """Provides basic quaternion operations.
    Quaternion objects can be instantiated as ``Quaternion(a, b, c, d)``
    as in $q = a + bi + cj + dk$.

    Parameters
    ==========

    norm : None or number
        Pre-defined quaternion norm. If a value is given, Quaternion.norm
        returns this pre-defined value instead of calculating the norm

    Examples
    ========

    >>> from sympy import Quaternion
    >>> q = Quaternion(1, 2, 3, 4)
    >>> q
    1 + 2*i + 3*j + 4*k

    Quaternions over complex fields can be defined as:

    >>> from sympy import Quaternion
    >>> from sympy import symbols, I
    >>> x = symbols('x')
    >>> q1 = Quaternion(x, x**3, x, x**2, real_field = False)
    >>> q2 = Quaternion(3 + 4*I, 2 + 5*I, 0, 7 + 8*I, real_field = False)
    >>> q1
    x + x**3*i + x*j + x**2*k
    >>> q2
    (3 + 4*I) + (2 + 5*I)*i + 0*j + (7 + 8*I)*k

    Defining symbolic unit quaternions:

    >>> from sympy import Quaternion
    >>> from sympy.abc import w, x, y, z
    >>> q = Quaternion(w, x, y, z, norm=1)
    >>> q
    w + x*i + y*j + z*k
    >>> q.norm()
    1

    References
    ==========

    .. [1] https://www.euclideanspace.com/maths/algebra/realNormedAlgebra/quaternions/
    .. [2] https://en.wikipedia.org/wiki/Quaternion

    """
    _op_priority = 11.0
    is_commutative = False
    if TYPE_CHECKING:

        @property
        def args(self) -> tuple[Expr, Expr, Expr, Expr]:
            ...
        _real_field: bool
        _norm: Expr | None

    def pow(self, p: int) -> Quaternion:
        """Finds the pth power of the quaternion.

        Parameters
        ==========

        p : int
            Power to be applied on quaternion.

        Returns
        =======

        Quaternion
            Returns the p-th power of the current quaternion.
            Returns the inverse if p = -1.

        Examples
        ========

        >>> from sympy import Quaternion
        >>> q = Quaternion(1, 2, 3, 4)
        >>> q.pow(4)
        668 + (-224)*i + (-336)*j + (-448)*k

        """
        try:
            q, p = (self, as_int(p))
        except ValueError:
            return NotImplemented
        if p < 0:
            q, p = (q.inverse(), -p)
        if p == 1:
            return q
        res = Quaternion(1, 0, 0, 0)
        while p > 0:
            if p & 1:
                res *= q
            q *= q
            p >>= 1
        return res
