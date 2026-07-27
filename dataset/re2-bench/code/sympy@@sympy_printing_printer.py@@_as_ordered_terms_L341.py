from typing import Any, Type
from functools import cmp_to_key, update_wrapper
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.numbers import Rational
from sympy.core.symbol import Wild
from sympy.series.order import Order

class Printer:
    """ Generic printer

    Its job is to provide infrastructure for implementing new printers easily.

    If you want to define your custom Printer or your custom printing method
    for your custom class then see the example above: printer_example_ .
    """
    _global_settings: dict[str, Any] = {}
    _default_settings: dict[str, Any] = {}
    printmethod: str = None

    @classmethod
    def _get_initial_settings(cls):
        settings = cls._default_settings.copy()
        for key, val in cls._global_settings.items():
            if key in cls._default_settings:
                settings[key] = val
        return settings

    def __init__(self, settings=None):
        self._str = str
        self._settings = self._get_initial_settings()
        self._context = {}
        if settings is not None:
            self._settings.update(settings)
            if len(self._settings) > len(self._default_settings):
                for key in self._settings:
                    if key not in self._default_settings:
                        raise TypeError("Unknown setting '%s'." % key)
        self._print_level = 0

    @property
    def order(self):
        if 'order' in self._settings:
            return self._settings['order']
        else:
            raise AttributeError('No order defined.')

    def _as_ordered_terms(self, expr, order=None):
        """A compatibility function for ordering terms in Add. """
        order = order or self.order
        if order == 'old':
            return sorted(Add.make_args(expr), key=cmp_to_key(self._compare_pretty))
        elif order == 'none':
            return list(expr.args)
        else:
            return expr.as_ordered_terms(order=order)

    def _compare_pretty(self, a, b):
        """return -1, 0, 1 if a is canonically less, equal or
        greater than b. This is used when 'order=old' is selected
        for printing. This puts Order last, orders Rationals
        according to value, puts terms in order wrt the power of
        the last power appearing in a term. Ties are broken using
        Basic.compare.
        """
        from sympy.core.numbers import Rational
        from sympy.core.symbol import Wild
        from sympy.series.order import Order
        if isinstance(a, Order) and (not isinstance(b, Order)):
            return 1
        if not isinstance(a, Order) and isinstance(b, Order):
            return -1
        if isinstance(a, Rational) and isinstance(b, Rational):
            l = a.p * b.q
            r = b.p * a.q
            return (l > r) - (l < r)
        else:
            p1, p2, p3 = (Wild('p1'), Wild('p2'), Wild('p3'))
            r_a = a.match(p1 * p2 ** p3)
            if r_a and p3 in r_a:
                a3 = r_a[p3]
                r_b = b.match(p1 * p2 ** p3)
                if r_b and p3 in r_b:
                    b3 = r_b[p3]
                    c = Basic.compare(a3, b3)
                    if c != 0:
                        return c
        return Basic.compare(a, b)
