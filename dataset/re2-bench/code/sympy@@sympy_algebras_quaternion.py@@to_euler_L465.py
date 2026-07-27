from typing import TYPE_CHECKING, overload
from sympy.core.singleton import S
from sympy.core.relational import is_eq
from sympy.functions.elementary.complexes import (conjugate, im, re, sign)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (acos, asin, atan2)
from sympy.simplify.trigsimp import trigsimp
from sympy.core.expr import Expr

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

    @property
    def a(self) -> Expr:
        return self.args[0]

    @property
    def b(self) -> Expr:
        return self.args[1]

    @property
    def c(self) -> Expr:
        return self.args[2]

    @property
    def d(self) -> Expr:
        return self.args[3]

    def to_euler(self, seq: str, angle_addition: bool=True, avoid_square_root: bool=False) -> tuple[Expr, Expr, Expr]:
        """Returns Euler angles representing same rotation as the quaternion,
        in the sequence given by ``seq``. This implements the method described
        in [1]_.

        For degenerate cases (gymbal lock cases), the third angle is
        set to zero.

        Parameters
        ==========

        seq : string of length 3
            Represents the sequence of rotations.
            For extrinsic rotations, seq must be all lowercase and its elements
            must be from the set ``{'x', 'y', 'z'}``
            For intrinsic rotations, seq must be all uppercase and its elements
            must be from the set ``{'X', 'Y', 'Z'}``

        angle_addition : bool
            When True, first and third angles are given as an addition and
            subtraction of two simpler ``atan2`` expressions. When False, the
            first and third angles are each given by a single more complicated
            ``atan2`` expression. This equivalent expression is given by:

            .. math::

                \\operatorname{atan_2} (b,a) \\pm \\operatorname{atan_2} (d,c) =
                \\operatorname{atan_2} (bc\\pm ad, ac\\mp bd)

            Default value: True

        avoid_square_root : bool
            When True, the second angle is calculated with an expression based
            on ``acos``, which is slightly more complicated but avoids a square
            root. When False, second angle is calculated with ``atan2``, which
            is simpler and can be better for numerical reasons (some
            numerical implementations of ``acos`` have problems near zero).
            Default value: False


        Returns
        =======

        Tuple
            The Euler angles calculated from the quaternion

        Examples
        ========

        >>> from sympy import Quaternion
        >>> from sympy.abc import a, b, c, d
        >>> euler = Quaternion(a, b, c, d).to_euler('zyz')
        >>> euler
        (-atan2(-b, c) + atan2(d, a),
         2*atan2(sqrt(b**2 + c**2), sqrt(a**2 + d**2)),
         atan2(-b, c) + atan2(d, a))


        References
        ==========

        .. [1] https://doi.org/10.1371/journal.pone.0276302

        """
        if self.is_zero_quaternion():
            raise ValueError('Cannot convert a quaternion with norm 0.')
        extrinsic = _is_extrinsic(seq)
        i1, j1, k1 = list(seq.lower())
        i = 'xyz'.index(i1) + 1
        j = 'xyz'.index(j1) + 1
        k = 'xyz'.index(k1) + 1
        if not extrinsic:
            i, k = (k, i)
        symmetric = i == k
        if symmetric:
            k = 6 - i - j
        sign = (i - j) * (j - k) * (k - i) // 2
        elements = [self.a, self.b, self.c, self.d]
        a = elements[0]
        b = elements[i]
        c = elements[j]
        d = elements[k] * sign
        if not symmetric:
            a, b, c, d = (a - c, b + d, c + a, d - b)
        if avoid_square_root:
            if symmetric:
                n2 = self.norm() ** 2
                angles1 = acos((a * a + b * b - c * c - d * d) / n2)
            else:
                n2 = 2 * self.norm() ** 2
                angles1 = asin((c * c + d * d - a * a - b * b) / n2)
        else:
            angles1 = 2 * atan2(sqrt(c * c + d * d), sqrt(a * a + b * b))
            if not symmetric:
                angles1 -= S.Pi / 2
        case = 0
        if is_eq(c, S.Zero) and is_eq(d, S.Zero):
            case = 1
        if is_eq(a, S.Zero) and is_eq(b, S.Zero):
            case = 2
        if case == 0:
            if angle_addition:
                angles0 = atan2(b, a) + atan2(d, c)
                angles2 = atan2(b, a) - atan2(d, c)
            else:
                angles0 = atan2(b * c + a * d, a * c - b * d)
                angles2 = atan2(b * c - a * d, a * c + b * d)
        elif case == 1:
            if extrinsic:
                angles0 = S.Zero
                angles2 = 2 * atan2(b, a)
            else:
                angles0 = 2 * atan2(b, a)
                angles2 = S.Zero
        elif extrinsic:
            angles0 = S.Zero
            angles2 = -2 * atan2(d, c)
        else:
            angles0 = 2 * atan2(d, c)
            angles2 = S.Zero
        if not symmetric:
            angles0 *= sign
        if extrinsic:
            return (angles2, angles1, angles0)
        else:
            return (angles0, angles1, angles2)

    def norm(self) -> Expr:
        """Returns the norm of the quaternion."""
        if self._norm is None:
            q = self
            return sqrt(trigsimp(q.a ** 2 + q.b ** 2 + q.c ** 2 + q.d ** 2))
        return self._norm

    def is_zero_quaternion(self) -> bool | None:
        """
        Returns true if the quaternion is a zero quaternion or false if it is not a zero quaternion
        and None if the value is unknown.

        Explanation
        ===========

        A zero quaternion is a quaternion with both scalar part and
        vector part equal to 0.

        Examples
        ========

        >>> from sympy.algebras.quaternion import Quaternion
        >>> q = Quaternion(1, 0, 0, 0)
        >>> q.is_zero_quaternion()
        False

        >>> q = Quaternion(0, 0, 0, 0)
        >>> q.is_zero_quaternion()
        True

        See Also
        ========
        scalar_part
        vector_part

        """
        return self.norm().is_zero
