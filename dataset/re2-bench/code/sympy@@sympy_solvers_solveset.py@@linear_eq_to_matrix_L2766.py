from sympy.core.sympify import sympify
from sympy.core import (S, Pow, Dummy, pi, Expr, Wild, Mul,
                        Add, Basic)
from sympy.core.relational import Eq, Ne, Relational
from sympy.polys.matrices.linsolve import _linear_eq_to_dict
from sympy.matrices import zeros, Matrix, MatrixBase
from sympy.polys.solvers import (sympy_eqs_to_ring, solve_lin_sys,
    PolyNonlinearError)
from sympy.utilities import filldedent
from sympy.utilities.iterables import (numbered_symbols, has_dups,
                                       is_sequence, iterable)

def linear_eq_to_matrix(equations, *symbols):
    r"""
    Converts a given System of Equations into Matrix form. Here ``equations``
    must be a linear system of equations in ``symbols``. Element ``M[i, j]``
    corresponds to the coefficient of the jth symbol in the ith equation.

    The Matrix form corresponds to the augmented matrix form. For example:

    .. math::

       4x + 2y + 3z & = 1 \\
       3x +  y +  z & = -6 \\
       2x + 4y + 9z & = 2

    This system will return :math:`A` and :math:`b` as:

    .. math::

       A = \left[\begin{array}{ccc}
       4 & 2 & 3 \\
       3 & 1 & 1 \\
       2 & 4 & 9
       \end{array}\right] \\

    .. math::

       b = \left[\begin{array}{c}
       1 \\ -6 \\ 2
       \end{array}\right]

    The only simplification performed is to convert
    ``Eq(a, b)`` :math:`\Rightarrow a - b`.

    Raises
    ======

    NonlinearError
        The equations contain a nonlinear term.
    ValueError
        The symbols are not given or are not unique.

    Examples
    ========

    >>> from sympy import linear_eq_to_matrix, symbols
    >>> c, x, y, z = symbols('c, x, y, z')

    The coefficients (numerical or symbolic) of the symbols will
    be returned as matrices:

    >>> eqns = [c*x + z - 1 - c, y + z, x - y]
    >>> A, b = linear_eq_to_matrix(eqns, [x, y, z])
    >>> A
    Matrix([
    [c,  0, 1],
    [0,  1, 1],
    [1, -1, 0]])
    >>> b
    Matrix([
    [c + 1],
    [    0],
    [    0]])

    This routine does not simplify expressions and will raise an error
    if nonlinearity is encountered:

    >>> eqns = [
    ...     (x**2 - 3*x)/(x - 3) - 3,
    ...     y**2 - 3*y - y*(y - 4) + x - 4]
    >>> linear_eq_to_matrix(eqns, [x, y])
    Traceback (most recent call last):
    ...
    NonlinearError:
    symbol-dependent term can be ignored using `strict=False`

    Simplifying these equations will discard the removable singularity in the
    first and reveal the linear structure of the second:

    >>> [e.simplify() for e in eqns]
    [x - 3, x + y - 4]

    Any such simplification needed to eliminate nonlinear terms must be done
    *before* calling this routine.

    """
    if not symbols:
        raise ValueError(filldedent('''
            Symbols must be given, for which coefficients
            are to be found.
            '''))

    # Check if 'symbols' is a set and raise an error if it is
    if isinstance(symbols[0], set):
        raise TypeError(
            "Unordered 'set' type is not supported as input for symbols.")

    if hasattr(symbols[0], '__iter__'):
        symbols = symbols[0]

    if has_dups(symbols):
        raise ValueError('Symbols must be unique')

    equations = sympify(equations)
    if isinstance(equations, MatrixBase):
        equations = list(equations)
    elif isinstance(equations, (Expr, Eq)):
        equations = [equations]
    elif not is_sequence(equations):
        raise ValueError(filldedent('''
            Equation(s) must be given as a sequence, Expr,
            Eq or Matrix.
            '''))

    # construct the dictionaries
    try:
        eq, c = _linear_eq_to_dict(equations, symbols)
    except PolyNonlinearError as err:
        raise NonlinearError(str(err))
    # prepare output matrices
    n, m = shape = len(eq), len(symbols)
    ix = dict(zip(symbols, range(m)))
    A = zeros(*shape)
    for row, d in enumerate(eq):
        for k in d:
            col = ix[k]
            A[row, col] = d[k]
    b = Matrix(n, 1, [-i for i in c])
    return A, b
