from .coreerrors import NonCommutativeExpression
from collections import defaultdict
from sympy.polys.polytools import gcd, factor

class Term:
    """Efficient representation of ``coeff*(numer/denom)``. """
    __slots__ = ('coeff', 'numer', 'denom')

    def __init__(self, term, numer=None, denom=None):
        if numer is None and denom is None:
            if not term.is_commutative:
                raise NonCommutativeExpression('commutative expression expected')
            coeff, factors = term.as_coeff_mul()
            numer, denom = (defaultdict(int), defaultdict(int))
            for factor in factors:
                base, exp = decompose_power(factor)
                if base.is_Add:
                    cont, base = base.primitive()
                    coeff *= cont ** exp
                if exp > 0:
                    numer[base] += exp
                else:
                    denom[base] += -exp
            numer = Factors(numer)
            denom = Factors(denom)
        else:
            coeff = term
            if numer is None:
                numer = Factors()
            if denom is None:
                denom = Factors()
        self.coeff = coeff
        self.numer = numer
        self.denom = denom

    def mul(self, other):
        coeff = self.coeff * other.coeff
        numer = self.numer.mul(other.numer)
        denom = self.denom.mul(other.denom)
        numer, denom = numer.normal(denom)
        return Term(coeff, numer, denom)
