from typing import TYPE_CHECKING, overload
from sympy.matrices.dense import MutableDenseMatrix as Matrix
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

    def to_rotation_matrix(self, v: tuple[SExpr, SExpr, SExpr] | None=None, homogeneous: bool=True) -> Matrix:
        """Returns the equivalent rotation transformation matrix of the quaternion
        which represents rotation about the origin if ``v`` is not passed.

        Parameters
        ==========

        v : tuple or None
            Default value: None
        homogeneous : bool
            When True, gives an expression that may be more efficient for
            symbolic calculations but less so for direct evaluation. Both
            formulas are mathematically equivalent.
            Default value: True

        Returns
        =======

        tuple
            Returns the equivalent rotation transformation matrix of the quaternion
            which represents rotation about the origin if v is not passed.

        Examples
        ========

        >>> from sympy import Quaternion
        >>> from sympy import symbols, trigsimp, cos, sin
        >>> x = symbols('x')
        >>> q = Quaternion(cos(x/2), 0, 0, sin(x/2))
        >>> trigsimp(q.to_rotation_matrix())
        Matrix([
        [cos(x), -sin(x), 0],
        [sin(x),  cos(x), 0],
        [     0,       0, 1]])

        Generates a 4x4 transformation matrix (used for rotation about a point
        other than the origin) if the point(v) is passed as an argument.
        """
        q = self
        s = q.norm() ** (-2)
        if homogeneous:
            m00 = s * (q.a ** 2 + q.b ** 2 - q.c ** 2 - q.d ** 2)
            m11 = s * (q.a ** 2 - q.b ** 2 + q.c ** 2 - q.d ** 2)
            m22 = s * (q.a ** 2 - q.b ** 2 - q.c ** 2 + q.d ** 2)
        else:
            m00 = 1 - 2 * s * (q.c ** 2 + q.d ** 2)
            m11 = 1 - 2 * s * (q.b ** 2 + q.d ** 2)
            m22 = 1 - 2 * s * (q.b ** 2 + q.c ** 2)
        m01 = 2 * s * (q.b * q.c - q.d * q.a)
        m02 = 2 * s * (q.b * q.d + q.c * q.a)
        m10 = 2 * s * (q.b * q.c + q.d * q.a)
        m12 = 2 * s * (q.c * q.d - q.b * q.a)
        m20 = 2 * s * (q.b * q.d - q.c * q.a)
        m21 = 2 * s * (q.c * q.d + q.b * q.a)
        if not v:
            return Matrix([[m00, m01, m02], [m10, m11, m12], [m20, m21, m22]])
        else:
            x, y, z = v
            m03 = x - x * m00 - y * m01 - z * m02
            m13 = y - x * m10 - y * m11 - z * m12
            m23 = z - x * m20 - y * m21 - z * m22
            m30 = m31 = m32 = 0
            m33 = 1
            return Matrix([[m00, m01, m02, m03], [m10, m11, m12, m13], [m20, m21, m22, m23], [m30, m31, m32, m33]])
