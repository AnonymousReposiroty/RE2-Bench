from sympy.concrete.expr_with_limits import AddWithLimits
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.polys import Poly, PolynomialError
from sympy.utilities.exceptions import sympy_deprecation_warning

class Integral(AddWithLimits):
    """Represents unevaluated integral."""
    __slots__ = ()
    args: tuple[Expr, Tuple]

    def __new__(cls, function, *symbols, **assumptions) -> Integral:
        """Create an unevaluated integral.

        Explanation
        ===========

        Arguments are an integrand followed by one or more limits.

        If no limits are given and there is only one free symbol in the
        expression, that symbol will be used, otherwise an error will be
        raised.

        >>> from sympy import Integral
        >>> from sympy.abc import x, y
        >>> Integral(x)
        Integral(x, x)
        >>> Integral(y)
        Integral(y, y)

        When limits are provided, they are interpreted as follows (using
        ``x`` as though it were the variable of integration):

            (x,) or x - indefinite integral
            (x, a) - "evaluate at" integral is an abstract antiderivative
            (x, a, b) - definite integral

        The ``as_dummy`` method can be used to see which symbols cannot be
        targeted by subs: those with a prepended underscore cannot be
        changed with ``subs``. (Also, the integration variables themselves --
        the first element of a limit -- can never be changed by subs.)

        >>> i = Integral(x, x)
        >>> at = Integral(x, (x, x))
        >>> i.as_dummy()
        Integral(x, x)
        >>> at.as_dummy()
        Integral(_0, (_0, x))

        """
        if hasattr(function, '_eval_Integral'):
            return function._eval_Integral(*symbols, **assumptions)
        if isinstance(function, Poly):
            sympy_deprecation_warning('\n                integrate(Poly) and Integral(Poly) are deprecated. Instead,\n                use the Poly.integrate() method, or convert the Poly to an\n                Expr first with the Poly.as_expr() method.\n                ', deprecated_since_version='1.6', active_deprecations_target='deprecated-integrate-poly')
        obj = AddWithLimits.__new__(cls, function, *symbols, **assumptions)
        return obj
