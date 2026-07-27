from typing import Any
from sympy.core import S, Rational, Pow, Basic, Mul, Number
from .precedence import precedence, PRECEDENCE
from .printer import Printer, print_function

class StrPrinter(Printer):
    printmethod = '_sympystr'
    _default_settings: dict[str, Any] = {'order': None, 'full_prec': 'auto', 'sympy_integers': False, 'abbrev': False, 'perm_cyclic': True, 'min': None, 'max': None, 'dps': None}
    _relationals: dict[str, str] = {}

    def parenthesize(self, item, level, strict=False):
        if precedence(item) < level or (not strict and precedence(item) <= level):
            return '(%s)' % self._print(item)
        else:
            return self._print(item)

    def _print_Pow(self, expr, rational=False):
        """Printing helper function for ``Pow``

        Parameters
        ==========

        rational : bool, optional
            If ``True``, it will not attempt printing ``sqrt(x)`` or
            ``x**S.Half`` as ``sqrt``, and will use ``x**(1/2)``
            instead.

            See examples for additional details

        Examples
        ========

        >>> from sympy import sqrt, StrPrinter
        >>> from sympy.abc import x

        How ``rational`` keyword works with ``sqrt``:

        >>> printer = StrPrinter()
        >>> printer._print_Pow(sqrt(x), rational=True)
        'x**(1/2)'
        >>> printer._print_Pow(sqrt(x), rational=False)
        'sqrt(x)'
        >>> printer._print_Pow(1/sqrt(x), rational=True)
        'x**(-1/2)'
        >>> printer._print_Pow(1/sqrt(x), rational=False)
        '1/sqrt(x)'

        Notes
        =====

        ``sqrt(x)`` is canonicalized as ``Pow(x, S.Half)`` in SymPy,
        so there is no need of defining a separate printer for ``sqrt``.
        Instead, it should be handled here as well.
        """
        PREC = precedence(expr)
        if expr.exp is S.Half and (not rational):
            return 'sqrt(%s)' % self._print(expr.base)
        if expr.is_commutative:
            if -expr.exp is S.Half and (not rational):
                return '%s/sqrt(%s)' % tuple((self._print(arg) for arg in (S.One, expr.base)))
            if expr.exp is -S.One:
                return '%s/%s' % (self._print(S.One), self.parenthesize(expr.base, PREC, strict=False))
        e = self.parenthesize(expr.exp, PREC, strict=False)
        if self.printmethod == '_sympyrepr' and expr.exp.is_Rational and (expr.exp.q != 1):
            if e.startswith('(Rational'):
                return '%s**%s' % (self.parenthesize(expr.base, PREC, strict=False), e[1:-1])
        return '%s**%s' % (self.parenthesize(expr.base, PREC, strict=False), e)
    _print_MatrixSymbol = _print_Symbol
    _print_RandomSymbol = _print_Symbol

    def _print(self, expr, **kwargs) -> str:
        """Internal dispatcher

        Tries the following concepts to print an expression:
            1. Let the object print itself if it knows how.
            2. Take the best fitting method defined in the printer.
            3. As fall-back use the emptyPrinter method for the printer.
        """
        self._print_level += 1
        try:
            if self.printmethod and hasattr(expr, self.printmethod):
                if not (isinstance(expr, type) and issubclass(expr, Basic)):
                    return getattr(expr, self.printmethod)(self, **kwargs)
            classes = type(expr).__mro__
            if AppliedUndef in classes:
                classes = classes[classes.index(AppliedUndef):]
            if UndefinedFunction in classes:
                classes = classes[classes.index(UndefinedFunction):]
            if Function in classes:
                i = classes.index(Function)
                classes = tuple((c for c in classes[:i] if c.__name__ == classes[0].__name__ or c.__name__.endswith('Base'))) + classes[i:]
            for cls in classes:
                printmethodname = '_print_' + cls.__name__
                printmethod = getattr(self, printmethodname, None)
                if printmethod is not None:
                    return printmethod(expr, **kwargs)
            return self.emptyPrinter(expr)
        finally:
            self._print_level -= 1
