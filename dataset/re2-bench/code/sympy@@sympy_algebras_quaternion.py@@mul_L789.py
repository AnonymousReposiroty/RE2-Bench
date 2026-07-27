from typing import TYPE_CHECKING, overload
from sympy.functions.elementary.complexes import (conjugate, im, re, sign)
from sympy.core.expr import Expr
from sympy.core.sympify import sympify, _sympify

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

    def mul(self, other: SExpr | Quaternion) -> Quaternion:
        """Multiplies quaternions.

        Parameters
        ==========

        other : Quaternion or symbol
            The quaternion to multiply to current (self) quaternion.

        Returns
        =======

        Quaternion
            The resultant quaternion after multiplying self with other

        Examples
        ========

        >>> from sympy import Quaternion
        >>> from sympy import symbols
        >>> q1 = Quaternion(1, 2, 3, 4)
        >>> q2 = Quaternion(5, 6, 7, 8)
        >>> q1.mul(q2)
        (-60) + 12*i + 30*j + 24*k
        >>> q1.mul(2)
        2 + 4*i + 6*j + 8*k
        >>> x = symbols('x', real = True)
        >>> q1.mul(x)
        x + 2*x*i + 3*x*j + 4*x*k

        Quaternions over complex fields :

        >>> from sympy import Quaternion
        >>> from sympy import I
        >>> q3 = Quaternion(3 + 4*I, 2 + 5*I, 0, 7 + 8*I, real_field = False)
        >>> q3.mul(2 + 3*I)
        (2 + 3*I)*(3 + 4*I) + (2 + 3*I)*(2 + 5*I)*i + 0*j + (2 + 3*I)*(7 + 8*I)*k

        """
        return self._generic_mul(self, _sympify(other))

    @overload
    @staticmethod
    def _generic_mul(q1: Quaternion, q2: Expr) -> Quaternion:
        ...

    @overload
    @staticmethod
    def _generic_mul(q1: Expr, q2: Quaternion) -> Quaternion:
        ...

    @overload
    @staticmethod
    def _generic_mul(q1: Expr, q2: Expr) -> Expr:
        ...

    @staticmethod
    def _generic_mul(q1: Quaternion | Expr, q2: Quaternion | Expr) -> Quaternion | Expr:
        """Generic multiplication.

        Parameters
        ==========

        q1 : Quaternion or symbol
        q2 : Quaternion or symbol

        It is important to note that if neither q1 nor q2 is a Quaternion,
        this function simply returns q1 * q2.

        Returns
        =======

        Quaternion
            The resultant quaternion after multiplying q1 and q2

        Examples
        ========

        >>> from sympy import Quaternion
        >>> from sympy import Symbol, S
        >>> q1 = Quaternion(1, 2, 3, 4)
        >>> q2 = Quaternion(5, 6, 7, 8)
        >>> Quaternion._generic_mul(q1, q2)
        (-60) + 12*i + 30*j + 24*k
        >>> Quaternion._generic_mul(q1, S(2))
        2 + 4*i + 6*j + 8*k
        >>> x = Symbol('x', real = True)
        >>> Quaternion._generic_mul(q1, x)
        x + 2*x*i + 3*x*j + 4*x*k

        Quaternions over complex fields :

        >>> from sympy import I
        >>> q3 = Quaternion(3 + 4*I, 2 + 5*I, 0, 7 + 8*I, real_field = False)
        >>> Quaternion._generic_mul(q3, 2 + 3*I)
        (2 + 3*I)*(3 + 4*I) + (2 + 3*I)*(2 + 5*I)*i + 0*j + (2 + 3*I)*(7 + 8*I)*k

        """
        if isinstance(q1, Quaternion) and isinstance(q2, Quaternion):
            if q1._norm is None and q2._norm is None:
                norm = None
            else:
                norm = q1.norm() * q2.norm()
            return Quaternion(-q1.b * q2.b - q1.c * q2.c - q1.d * q2.d + q1.a * q2.a, q1.b * q2.a + q1.c * q2.d - q1.d * q2.c + q1.a * q2.b, -q1.b * q2.d + q1.c * q2.a + q1.d * q2.b + q1.a * q2.c, q1.b * q2.c - q1.c * q2.b + q1.d * q2.a + q1.a * q2.d, norm=norm)
        elif isinstance(q2, Quaternion):
            if q2.real_field and q1.is_complex:
                return Quaternion(re(q1), im(q1), 0, 0) * q2
            elif q1.is_commutative:
                return Quaternion(q1 * q2.a, q1 * q2.b, q1 * q2.c, q1 * q2.d)
            else:
                raise ValueError('Only commutative expressions can be multiplied with a Quaternion.')
        elif isinstance(q1, Quaternion):
            if q1.real_field and q2.is_complex:
                return q1 * Quaternion(re(q2), im(q2), 0, 0)
            elif q2.is_commutative:
                return Quaternion(q2 * q1.a, q2 * q1.b, q2 * q1.c, q2 * q1.d)
            else:
                raise ValueError('Only commutative expressions can be multiplied with a Quaternion.')
        else:
            return q1 * q2
