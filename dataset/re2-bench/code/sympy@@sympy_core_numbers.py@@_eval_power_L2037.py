from .singleton import S, Singleton
from .intfunc import num_digits, igcd, ilcm, mod_inverse, integer_nthroot
from .power import Pow
from sympy.ntheory.factor_ import perfect_power

class Integer(Rational):
    """Represents integer numbers of any size.

    Examples
    ========

    >>> from sympy import Integer
    >>> Integer(3)
    3

    If a float or a rational is passed to Integer, the fractional part
    will be discarded; the effect is of rounding toward zero.

    >>> Integer(3.8)
    3
    >>> Integer(-3.8)
    -3

    A string is acceptable input if it can be parsed as an integer:

    >>> Integer("9" * 20)
    99999999999999999999

    It is rarely needed to explicitly instantiate an Integer, because
    Python integers are automatically converted to Integer when they
    are used in SymPy expressions.
    """
    q = 1
    is_integer = True
    is_number = True
    is_Integer = True
    __slots__ = ()

    def _eval_power(self, expt):
        """
        Tries to do some simplifications on self**expt

        Returns None if no further simplifications can be done.

        Explanation
        ===========

        When exponent is a fraction (so we have for example a square root),
        we try to find a simpler representation by factoring the argument
        up to factors of 2**15, e.g.

          - sqrt(4) becomes 2
          - sqrt(-4) becomes 2*I
          - (2**(3+7)*3**(6+7))**Rational(1,7) becomes 6*18**(3/7)

        Further simplification would require a special call to factorint on
        the argument which is not done here for sake of speed.

        """
        from sympy.ntheory.factor_ import perfect_power
        if expt is S.Infinity:
            if self.p > S.One:
                return S.Infinity
            return S.Infinity + S.ImaginaryUnit * S.Infinity
        if expt is S.NegativeInfinity:
            return Rational._new(1, self, 1) ** S.Infinity
        if not isinstance(expt, Number):
            if self.is_negative and expt.is_even:
                return (-self) ** expt
        if isinstance(expt, Float):
            return super()._eval_power(expt)
        if not isinstance(expt, Rational):
            return
        if expt is S.Half and self.is_negative:
            return S.ImaginaryUnit * Pow(-self, expt)
        if expt.is_negative:
            ne = -expt
            if self.is_negative:
                return S.NegativeOne ** expt * Rational._new(1, -self.p, 1) ** ne
            else:
                return Rational._new(1, self.p, 1) ** ne
        x, xexact = integer_nthroot(abs(self.p), expt.q)
        if xexact:
            result = Integer(x ** abs(expt.p))
            if self.is_negative:
                result *= S.NegativeOne ** expt
            return result
        b_pos = int(abs(self.p))
        p = perfect_power(b_pos)
        if p is not False:
            dict = {p[0]: p[1]}
        else:
            dict = Integer(b_pos).factors(limit=2 ** 15)
        out_int = 1
        out_rad = 1
        sqr_int = 1
        sqr_gcd = 0
        sqr_dict = {}
        for prime, exponent in dict.items():
            exponent *= expt.p
            div_e, div_m = divmod(exponent, expt.q)
            if div_e > 0:
                out_int *= prime ** div_e
            if div_m > 0:
                g = igcd(div_m, expt.q)
                if g != 1:
                    out_rad *= Pow(prime, Rational._new(div_m // g, expt.q // g, 1))
                else:
                    sqr_dict[prime] = div_m
        for p, ex in sqr_dict.items():
            if sqr_gcd == 0:
                sqr_gcd = ex
            else:
                sqr_gcd = igcd(sqr_gcd, ex)
                if sqr_gcd == 1:
                    break
        for k, v in sqr_dict.items():
            sqr_int *= k ** (v // sqr_gcd)
        if sqr_int == b_pos and out_int == 1 and (out_rad == 1):
            result = None
        else:
            result = out_int * out_rad * Pow(sqr_int, Rational(sqr_gcd, expt.q))
            if self.is_negative:
                result *= Pow(S.NegativeOne, expt)
        return result
