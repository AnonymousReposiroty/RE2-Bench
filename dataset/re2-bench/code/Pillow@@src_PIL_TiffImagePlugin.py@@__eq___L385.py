from fractions import Fraction
from numbers import Number, Rational

class IFDRational(Rational):
    """Implements a rational class where 0/0 is a legal value to match
    the in the wild use of exif rationals.

    e.g., DigitalZoomRatio - 0.00/0.00  indicates that no digital zoom was used
    """
    " If the denominator is 0, store this as a float('nan'), otherwise store\n    as a fractions.Fraction(). Delegate as appropriate\n\n    "
    __slots__ = ('_numerator', '_denominator', '_val')

    def __init__(self, value, denominator=1):
        """
        :param value: either an integer numerator, a
        float/rational/other number, or an IFDRational
        :param denominator: Optional integer denominator
        """
        if isinstance(value, IFDRational):
            self._numerator = value.numerator
            self._denominator = value.denominator
            self._val = value._val
            return
        if isinstance(value, Fraction):
            self._numerator = value.numerator
            self._denominator = value.denominator
        else:
            self._numerator = value
            self._denominator = denominator
        if denominator == 0:
            self._val = float('nan')
        elif denominator == 1:
            self._val = Fraction(value)
        else:
            self._val = Fraction(value, denominator)

    def __eq__(self, other):
        val = self._val
        if isinstance(other, IFDRational):
            other = other._val
        if isinstance(other, float):
            val = float(val)
        return val == other
    ' a = [\'add\',\'radd\', \'sub\', \'rsub\', \'mul\', \'rmul\',\n             \'truediv\', \'rtruediv\', \'floordiv\', \'rfloordiv\',\n             \'mod\',\'rmod\', \'pow\',\'rpow\', \'pos\', \'neg\',\n             \'abs\', \'trunc\', \'lt\', \'gt\', \'le\', \'ge\', \'bool\',\n             \'ceil\', \'floor\', \'round\']\n        print("\n".join("__%s__ = _delegate(\'__%s__\')" % (s,s) for s in a))\n        '
    __add__ = _delegate('__add__')
    __radd__ = _delegate('__radd__')
    __sub__ = _delegate('__sub__')
    __rsub__ = _delegate('__rsub__')
    __mul__ = _delegate('__mul__')
    __rmul__ = _delegate('__rmul__')
    __truediv__ = _delegate('__truediv__')
    __rtruediv__ = _delegate('__rtruediv__')
    __floordiv__ = _delegate('__floordiv__')
    __rfloordiv__ = _delegate('__rfloordiv__')
    __mod__ = _delegate('__mod__')
    __rmod__ = _delegate('__rmod__')
    __pow__ = _delegate('__pow__')
    __rpow__ = _delegate('__rpow__')
    __pos__ = _delegate('__pos__')
    __neg__ = _delegate('__neg__')
    __abs__ = _delegate('__abs__')
    __trunc__ = _delegate('__trunc__')
    __lt__ = _delegate('__lt__')
    __gt__ = _delegate('__gt__')
    __le__ = _delegate('__le__')
    __ge__ = _delegate('__ge__')
    __bool__ = _delegate('__bool__')
    __ceil__ = _delegate('__ceil__')
    __floor__ = _delegate('__floor__')
    __round__ = _delegate('__round__')
    if hasattr(Fraction, '__int__'):
        __int__ = _delegate('__int__')
