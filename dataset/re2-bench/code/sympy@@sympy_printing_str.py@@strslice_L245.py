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

    def _print_MatrixSlice(self, expr):

        def strslice(x, dim):
            x = list(x)
            if x[2] == 1:
                del x[2]
            if x[0] == 0:
                x[0] = ''
            if x[1] == dim:
                x[1] = ''
            return ':'.join((self._print(arg) for arg in x))
        return self.parenthesize(expr.parent, PRECEDENCE['Atom'], strict=True) + '[' + strslice(expr.rowslice, expr.parent.rows) + ', ' + strslice(expr.colslice, expr.parent.cols) + ']'
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
