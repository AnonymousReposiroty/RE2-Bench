from .singleton import S

def _unevaluated_Mul(*args):
    """Return a well-formed unevaluated Mul: Numbers are collected and
    put in slot 0, any arguments that are Muls will be flattened, and args
    are sorted. Use this when args have changed but you still want to return
    an unevaluated Mul.

    Examples
    ========

    >>> from sympy.core.mul import _unevaluated_Mul as uMul
    >>> from sympy import S, sqrt, Mul
    >>> from sympy.abc import x
    >>> a = uMul(*[S(3.0), x, S(2)])
    >>> a.args[0]
    6.00000000000000
    >>> a.args[1]
    x

    Two unevaluated Muls with the same arguments will
    always compare as equal during testing:

    >>> m = uMul(sqrt(2), sqrt(3))
    >>> m == uMul(sqrt(3), sqrt(2))
    True
    >>> u = Mul(sqrt(3), sqrt(2), evaluate=False)
    >>> m == uMul(u)
    True
    >>> m == Mul(*m.args)
    False

    """
    cargs = []
    ncargs = []
    args = list(args)
    co = S.One
    for a in args:
        if a.is_Mul:
            a_c, a_nc = a.args_cnc()
            args.extend(a_c)  # grow args
            ncargs.extend(a_nc)
        elif a.is_Number:
            co *= a
        elif a.is_commutative:
            cargs.append(a)
        else:
            ncargs.append(a)
    _mulsort(cargs)
    if co is not S.One:
        cargs.insert(0, co)
    return Mul._from_args(cargs+ncargs)
