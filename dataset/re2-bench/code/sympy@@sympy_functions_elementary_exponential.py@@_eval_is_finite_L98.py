from sympy.core.function import (DefinedFunction, ArgumentIndexError, expand_log,
    expand_mul, FunctionClass, PoleError, expand_multinomial, expand_complex)
from sympy.core.singleton import S
from sympy.functions.elementary.complexes import arg, unpolarify, im, re, Abs

class ExpBase(DefinedFunction):
    unbranched = True
    _singularities = (S.ComplexInfinity,)

    @property
    def exp(self):
        """
        Returns the exponent of the function.
        """
        return self.args[0]

    def _eval_is_finite(self):
        arg = self.exp
        if arg.is_infinite:
            if arg.is_extended_negative:
                return True
            if arg.is_extended_positive:
                return False
        if arg.is_finite:
            return True
