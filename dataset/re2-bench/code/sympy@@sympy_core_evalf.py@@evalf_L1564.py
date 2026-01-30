from mpmath.libmp.libmpf import dps_to_prec, prec_to_dps
from .singleton import S
from sympy.utilities.iterables import is_sequence
from sympy.core.expr import Expr
from sympy.functions.elementary.complexes import Abs, re, im
from .numbers import Float, Rational, Integer, AlgebraicNumber, Number
from sympy.functions.elementary.complexes import re, im
from .numbers import Float
from .numbers import Float
from .numbers import Float, Integer
from .numbers import Float, equal_valued
from .numbers import Float
from .numbers import Exp1, Float, Half, ImaginaryUnit, Integer, NaN, NegativeOne, One, Pi, Rational, \
    Zero, ComplexInfinity, AlgebraicNumber
from sympy.functions.elementary.complexes import Abs, im, re
from .numbers import Float, Number
from .expr import _mag

class EvalfMixin:
    """Mixin class adding evalf capability."""
    __slots__: tuple[str, ...] = ()

    def evalf(self, n=15, subs=None, maxn=100, chop=False, strict=False, quad=None, verbose=False):
        """
        Evaluate the given formula to an accuracy of *n* digits.

        Parameters
        ==========

        subs : dict, optional
            Substitute numerical values for symbols, e.g.
            ``subs={x:3, y:1+pi}``. The substitutions must be given as a
            dictionary.

        maxn : int, optional
            Allow a maximum temporary working precision of maxn digits.

        chop : bool or number, optional
            Specifies how to replace tiny real or imaginary parts in
            subresults by exact zeros.

            When ``True`` the chop value defaults to standard precision.

            Otherwise the chop value is used to determine the
            magnitude of "small" for purposes of chopping.

            >>> from sympy import N
            >>> x = 1e-4
            >>> N(x, chop=True)
            0.000100000000000000
            >>> N(x, chop=1e-5)
            0.000100000000000000
            >>> N(x, chop=1e-4)
            0

        strict : bool, optional
            Raise ``PrecisionExhausted`` if any subresult fails to
            evaluate to full accuracy, given the available maxprec.

        quad : str, optional
            Choose algorithm for numerical quadrature. By default,
            tanh-sinh quadrature is used. For oscillatory
            integrals on an infinite interval, try ``quad='osc'``.

        verbose : bool, optional
            Print debug information.

        Notes
        =====

        When Floats are naively substituted into an expression,
        precision errors may adversely affect the result. For example,
        adding 1e16 (a Float) to 1 will truncate to 1e16; if 1e16 is
        then subtracted, the result will be 0.
        That is exactly what happens in the following:

        >>> from sympy.abc import x, y, z
        >>> values = {x: 1e16, y: 1, z: 1e16}
        >>> (x + y - z).subs(values)
        0

        Using the subs argument for evalf is the accurate way to
        evaluate such an expression:

        >>> (x + y - z).evalf(subs=values)
        1.00000000000000
        """
        from .numbers import Float, Number
        n = n if n is not None else 15
        if subs and is_sequence(subs):
            raise TypeError('subs must be given as a dictionary')
        if n == 1 and isinstance(self, Number):
            from .expr import _mag
            rv = self.evalf(2, subs, maxn, chop, strict, quad, verbose)
            m = _mag(rv)
            rv = rv.round(1 - m)
            return rv
        if not evalf_table:
            _create_evalf_table()
        prec = dps_to_prec(n)
        options = {'maxprec': max(prec, int(maxn * LG10)), 'chop': chop, 'strict': strict, 'verbose': verbose}
        if subs is not None:
            options['subs'] = subs
        if quad is not None:
            options['quad'] = quad
        try:
            result = evalf(self, prec + 4, options)
        except NotImplementedError:
            if hasattr(self, 'subs') and subs is not None:
                v = self.subs(subs)._eval_evalf(prec)
            else:
                v = self._eval_evalf(prec)
            if v is None:
                return self
            elif not v.is_number:
                return v
            try:
                result = evalf(v, prec, options)
            except NotImplementedError:
                return v
        if result is S.ComplexInfinity:
            return result
        re, im, re_acc, im_acc = result
        if re is S.NaN or im is S.NaN:
            return S.NaN
        if re:
            p = max(min(prec, re_acc), 1)
            re = Float._new(re, p)
        else:
            re = S.Zero
        if im:
            p = max(min(prec, im_acc), 1)
            im = Float._new(im, p)
            return re + im * S.ImaginaryUnit
        else:
            return re
    n = evalf

    def _eval_evalf(self, prec: int) -> Expr | None:
        return None
