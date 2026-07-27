from sympy.core.function import DefinedFunction, ArgumentIndexError, PoleError, expand_mul
from sympy.core.logic import fuzzy_not, fuzzy_or, FuzzyBool, fuzzy_and
from sympy.core.singleton import S

class TrigonometricFunction(DefinedFunction):
    """Base class for trigonometric functions. """
    unbranched = True
    _singularities = (S.ComplexInfinity,)

    def _eval_is_rational(self):
        s = self.func(*self.args)
        if s.func == self.func:
            if s.args[0].is_rational and fuzzy_not(s.args[0].is_zero):
                return False
        else:
            return s.is_rational
