from typing import Any
from sympy.core import S, Rational, Pow, Basic, Mul, Number
from sympy.core.mul import _keep_coeff
from sympy.utilities.iterables import sift
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

    def _print_Mul(self, expr):
        prec = precedence(expr)
        args = expr.args
        if args[0] is S.One or any((isinstance(a, Number) or (a.is_Pow and all((ai.is_Integer for ai in a.args))) for a in args[1:])):
            d, n = sift(args, lambda x: isinstance(x, Pow) and bool(x.exp.as_coeff_Mul()[0] < 0), binary=True)
            for i, di in enumerate(d):
                if di.exp.is_Number:
                    e = -di.exp
                else:
                    dargs = list(di.exp.args)
                    dargs[0] = -dargs[0]
                    e = Mul._from_args(dargs)
                d[i] = Pow(di.base, e, evaluate=False) if e - 1 else di.base
            pre = []
            if n and (not n[0].is_Add) and n[0].could_extract_minus_sign():
                pre = [self._print(n.pop(0))]
            nfactors = pre + [self.parenthesize(a, prec, strict=False) for a in n]
            if not nfactors:
                nfactors = ['1']
            if len(d) > 1 and d[0].could_extract_minus_sign():
                pre = [self._print(d.pop(0))]
            else:
                pre = []
            dfactors = pre + [self.parenthesize(a, prec, strict=False) for a in d]
            n = '*'.join(nfactors)
            d = '*'.join(dfactors)
            if len(dfactors) > 1:
                return '%s/(%s)' % (n, d)
            elif dfactors:
                return '%s/%s' % (n, d)
            return n
        c, e = expr.as_coeff_Mul()
        if c < 0:
            expr = _keep_coeff(-c, e)
            sign = '-'
        else:
            sign = ''
        a = []
        b = []
        pow_paren = []
        if self.order not in ('old', 'none'):
            args = expr.as_ordered_factors()
        else:
            args = Mul.make_args(expr)

        def apow(i):
            b, e = i.as_base_exp()
            eargs = list(Mul.make_args(e))
            if eargs[0] is S.NegativeOne:
                eargs = eargs[1:]
            else:
                eargs[0] = -eargs[0]
            e = Mul._from_args(eargs)
            if isinstance(i, Pow):
                return i.func(b, e, evaluate=False)
            return i.func(e, evaluate=False)
        for item in args:
            if item.is_commutative and isinstance(item, Pow) and bool(item.exp.as_coeff_Mul()[0] < 0):
                if item.exp is not S.NegativeOne:
                    b.append(apow(item))
                else:
                    if len(item.args[0].args) != 1 and isinstance(item.base, (Mul, Pow)):
                        pow_paren.append(item)
                    b.append(item.base)
            elif item.is_Rational and item is not S.Infinity:
                if item.p != 1:
                    a.append(Rational(item.p))
                if item.q != 1:
                    b.append(Rational(item.q))
            else:
                a.append(item)
        a = a or [S.One]
        a_str = [self.parenthesize(x, prec, strict=False) for x in a]
        b_str = [self.parenthesize(x, prec, strict=False) for x in b]
        for item in pow_paren:
            if item.base in b:
                b_str[b.index(item.base)] = '(%s)' % b_str[b.index(item.base)]
        if not b:
            return sign + '*'.join(a_str)
        elif len(b) == 1:
            return sign + '*'.join(a_str) + '/' + b_str[0]
        else:
            return sign + '*'.join(a_str) + '/(%s)' % '*'.join(b_str)
    _print_MatrixSymbol = _print_Symbol
    _print_RandomSymbol = _print_Symbol

    @property
    def order(self):
        if 'order' in self._settings:
            return self._settings['order']
        else:
            raise AttributeError('No order defined.')

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
