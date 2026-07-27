from .expr import Expr, AtomicExpr
from .logic import fuzzy_and, fuzzy_or, fuzzy_not, FuzzyBool
from .numbers import Rational, Float, Integer
from sympy.utilities.lambdify import MPMATH_TRANSLATIONS
import mpmath
from mpmath import mpf, mpc

class Function(Application, Expr):
    """
    Base class for applied mathematical functions.

    It also serves as a constructor for undefined function classes.

    See the :ref:`custom-functions` guide for details on how to subclass
    ``Function`` and what methods can be defined.

    Examples
    ========

    **Undefined Functions**

    To create an undefined function, pass a string of the function name to
    ``Function``.

    >>> from sympy import Function, Symbol
    >>> x = Symbol('x')
    >>> f = Function('f')
    >>> g = Function('g')(x)
    >>> f
    f
    >>> f(x)
    f(x)
    >>> g
    g(x)
    >>> f(x).diff(x)
    Derivative(f(x), x)
    >>> g.diff(x)
    Derivative(g(x), x)

    Assumptions can be passed to ``Function`` the same as with a
    :class:`~.Symbol`. Alternatively, you can use a ``Symbol`` with
    assumptions for the function name and the function will inherit the name
    and assumptions associated with the ``Symbol``:

    >>> f_real = Function('f', real=True)
    >>> f_real(x).is_real
    True
    >>> f_real_inherit = Function(Symbol('f', real=True))
    >>> f_real_inherit(x).is_real
    True

    Note that assumptions on a function are unrelated to the assumptions on
    the variables it is called on. If you want to add a relationship, subclass
    ``Function`` and define custom assumptions handler methods. See the
    :ref:`custom-functions-assumptions` section of the :ref:`custom-functions`
    guide for more details.

    **Custom Function Subclasses**

    The :ref:`custom-functions` guide has several
    :ref:`custom-functions-complete-examples` of how to subclass ``Function``
    to create a custom function.

    """

    def _eval_evalf(self, prec):

        def _get_mpmath_func(fname):
            """Lookup mpmath function based on name"""
            if isinstance(self, AppliedUndef):
                return None
            if not hasattr(mpmath, fname):
                fname = MPMATH_TRANSLATIONS.get(fname, None)
                if fname is None:
                    return None
            return getattr(mpmath, fname)
        _eval_mpmath = getattr(self, '_eval_mpmath', None)
        if _eval_mpmath is None:
            func = _get_mpmath_func(self.func.__name__)
            args = self.args
        else:
            func, args = _eval_mpmath()
        if func is None:
            imp = getattr(self, '_imp_', None)
            if imp is None:
                return None
            try:
                return Float(imp(*[i.evalf(prec) for i in self.args]), prec)
            except (TypeError, ValueError):
                return None
        try:
            args = [arg._to_mpmath(prec + 5) for arg in args]

            def bad(m):
                from mpmath import mpf, mpc
                if isinstance(m, mpf):
                    m = m._mpf_
                    return m[1] != 1 and m[-1] == 1
                elif isinstance(m, mpc):
                    m, n = m._mpc_
                    return m[1] != 1 and m[-1] == 1 and (n[1] != 1) and (n[-1] == 1)
                else:
                    return False
            if any((bad(a) for a in args)):
                raise ValueError
        except ValueError:
            return
        with mpmath.workprec(prec):
            v = func(*args)
        return Expr._from_mpmath(v, prec)
    _singularities: FuzzyBool | tuple[Expr, ...] = None
