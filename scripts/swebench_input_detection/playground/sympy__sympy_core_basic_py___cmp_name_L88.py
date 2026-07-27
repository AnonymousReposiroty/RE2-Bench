# Problem: sympy@@sympy_core_basic.py@@_cmp_name_L88
# Module: sympy.core.basic
# Function: _cmp_name
# Line: 88

from sympy.core.basic import Basic
from sympy.core.numbers import ImaginaryUnit
from sympy.core.numbers import Pi
ordering_of_classes = [
    # singleton numbers
    'Zero', 'One', 'Half', 'Infinity', 'NaN', 'NegativeOne', 'NegativeInfinity',
    # numbers
    'Integer', 'Rational', 'Float',
    # singleton symbols
    'Exp1', 'Pi', 'ImaginaryUnit',
    # symbols
    'Symbol', 'Wild',
    # arithmetic operations
    'Pow', 'Mul', 'Add',
    # function values
    'Derivative', 'Integral',
    # defined singleton functions
    'Abs', 'Sign', 'Sqrt',
    'Floor', 'Ceiling',
    'Re', 'Im', 'Arg',
    'Conjugate',
    'Exp', 'Log',
    'Sin', 'Cos', 'Tan', 'Cot', 'ASin', 'ACos', 'ATan', 'ACot',
    'Sinh', 'Cosh', 'Tanh', 'Coth', 'ASinh', 'ACosh', 'ATanh', 'ACoth',
    'RisingFactorial', 'FallingFactorial',
    'factorial', 'binomial',
    'Gamma', 'LowerGamma', 'UpperGamma', 'PolyGamma',
    'Erf',
    # special polynomials
    'Chebyshev', 'Chebyshev2',
    # undefined functions
    'Function', 'WildFunction',
    # anonymous functions
    'Lambda',
    # Landau O symbol
    'Order',
    # relational operations
    'Equality', 'Unequality', 'StrictGreaterThan', 'StrictLessThan',
    'GreaterThan', 'LessThan',
]
def _cmp_name(x: type, y: type) -> int:
    """return -1, 0, 1 if the name of x is before that of y.
    A string comparison is done if either name does not appear
    in `ordering_of_classes`. This is the helper for
    ``Basic.compare``

    Examples
    ========

    >>> from sympy import cos, tan, sin
    >>> from sympy.core import basic
    >>> save = basic.ordering_of_classes
    >>> basic.ordering_of_classes = ()
    >>> basic._cmp_name(cos, tan)
    -1
    >>> basic.ordering_of_classes = ["tan", "sin", "cos"]
    >>> basic._cmp_name(cos, tan)
    1
    >>> basic._cmp_name(sin, cos)
    -1
    >>> basic.ordering_of_classes = save

    """
    n1 = x.__name__
    n2 = y.__name__
    if n1 == n2:
        return 0

    # If the other object is not a Basic subclass, then we are not equal to it.
    if not issubclass(y, Basic):
        return -1

    UNKNOWN = len(ordering_of_classes) + 1
    try:
        i1 = ordering_of_classes.index(n1)
    except ValueError:
        i1 = UNKNOWN
    try:
        i2 = ordering_of_classes.index(n2)
    except ValueError:
        i2 = UNKNOWN
    if i1 == UNKNOWN and i2 == UNKNOWN:
        return (n1 > n2) - (n1 < n2)
    return (i1 > i2) - (i1 < i2)


def test_input(pred_input):
    assert _cmp_name(x = ImaginaryUnit, y = Pi)==_cmp_name(x = pred_input['args']['x'], y = pred_input['args']['y']), 'Prediction failed!'
