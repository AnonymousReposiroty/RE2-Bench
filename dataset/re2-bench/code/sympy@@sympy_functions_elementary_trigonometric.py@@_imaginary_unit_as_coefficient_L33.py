from sympy.core.numbers import Rational, pi, Integer, Float, equal_valued
from sympy.core.singleton import S

def _imaginary_unit_as_coefficient(arg):
    """ Helper to extract symbolic coefficient for imaginary unit """
    if isinstance(arg, Float):
        return None
    else:
        return arg.as_coefficient(S.ImaginaryUnit)
