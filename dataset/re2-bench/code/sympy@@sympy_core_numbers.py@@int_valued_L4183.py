from sympy.external.gmpy import SYMPY_INTS, gmpy, flint

def int_valued(x):
    """return True only for a literal Number whose internal
    representation as a fraction has a denominator of 1,
    else False, i.e. integer, with no fractional part.
    """
    if isinstance(x, (SYMPY_INTS, int)):
        return True
    if type(x) is float:
        return x.is_integer()
    if isinstance(x, Integer):
        return True
    if isinstance(x, Float):
        # x = s*m*2**p; _mpf_ = s,m,e,p
        return x._mpf_[2] >= 0
    return False  # or add new types to recognize
