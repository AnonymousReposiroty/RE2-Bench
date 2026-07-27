from sympy.core.function import DefinedFunction, ArgumentIndexError, PoleError, expand_mul
from sympy.core.singleton import S
from sympy.functions.elementary.complexes import arg as arg_f, im, re
from sympy.functions.elementary.complexes import re
from sympy.functions.elementary.complexes import re
from sympy.functions.elementary.complexes import re
from sympy.functions.elementary.complexes import re

class TrigonometricFunction(DefinedFunction):
    """Base class for trigonometric functions. """
    unbranched = True
    _singularities = (S.ComplexInfinity,)

    def _as_real_imag(self, deep=True, **hints):
        if self.args[0].is_extended_real:
            if deep:
                hints['complex'] = False
                return (self.args[0].expand(deep, **hints), S.Zero)
            else:
                return (self.args[0], S.Zero)
        if deep:
            re, im = self.args[0].expand(deep, **hints).as_real_imag()
        else:
            re, im = self.args[0].as_real_imag()
        return (re, im)
