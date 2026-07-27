from .singleton import S, Singleton
from .power import Pow
from .mul import Mul
from .add import Add
from sympy.functions.elementary.exponential import log

class Exp1(NumberSymbol, metaclass=Singleton):
    """The `e` constant.

    Explanation
    ===========

    The transcendental number `e = 2.718281828\\ldots` is the base of the
    natural logarithm and of the exponential function, `e = \\exp(1)`.
    Sometimes called Euler's number or Napier's constant.

    Exp1 is a singleton, and can be accessed by ``S.Exp1``,
    or can be imported as ``E``.

    Examples
    ========

    >>> from sympy import exp, log, E
    >>> E is exp(1)
    True
    >>> log(E)
    1

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/E_%28mathematical_constant%29
    """
    is_real = True
    is_positive = True
    is_negative = False
    is_irrational = True
    is_number = True
    is_algebraic = False
    is_transcendental = True
    __slots__ = ()

    def _eval_power_exp_is_pow(self, arg):
        if arg.is_Number:
            if arg is oo:
                return oo
            elif arg == -oo:
                return S.Zero
        from sympy.functions.elementary.exponential import log
        if isinstance(arg, log):
            return arg.args[0]
        elif not arg.is_Add:
            Ioo = I * oo
            if arg in [Ioo, -Ioo]:
                return nan
            coeff = arg.coeff(pi * I)
            if coeff:
                if (2 * coeff).is_integer:
                    if coeff.is_even:
                        return S.One
                    elif coeff.is_odd:
                        return S.NegativeOne
                    elif (coeff + S.Half).is_even:
                        return -I
                    elif (coeff + S.Half).is_odd:
                        return I
                elif coeff.is_Rational:
                    ncoeff = coeff % 2
                    if ncoeff > 1:
                        ncoeff -= 2
                    if ncoeff != coeff:
                        return S.Exp1 ** (ncoeff * S.Pi * S.ImaginaryUnit)
            coeff, terms = arg.as_coeff_Mul()
            if coeff in (oo, -oo):
                return
            coeffs, log_term = ([coeff], None)
            for term in Mul.make_args(terms):
                if isinstance(term, log):
                    if log_term is None:
                        log_term = term.args[0]
                    else:
                        return
                elif term.is_comparable:
                    coeffs.append(term)
                else:
                    return
            return log_term ** Mul(*coeffs) if log_term else None
        elif arg.is_Add:
            out = []
            add = []
            argchanged = False
            for a in arg.args:
                if a is S.One:
                    add.append(a)
                    continue
                newa = self ** a
                if isinstance(newa, Pow) and newa.base is self:
                    if newa.exp != a:
                        add.append(newa.exp)
                        argchanged = True
                    else:
                        add.append(a)
                else:
                    out.append(newa)
            if out or argchanged:
                return Mul(*out) * Pow(self, Add(*add), evaluate=False)
        elif arg.is_Matrix:
            return arg.exp()
