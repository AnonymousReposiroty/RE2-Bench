import operator
from typing import TypeVar, Callable, Literal, SupportsIndex, SupportsInt, overload

def as_int(n: SupportsIndex | SupportsInt, strict: bool = True) -> int:
    """
    Convert the argument to a builtin integer.

    The return value is guaranteed to be equal to the input. ValueError is
    raised if the input has a non-integral value. When ``strict`` is True, this
    uses `__index__ <https://docs.python.org/3/reference/datamodel.html#object.__index__>`_
    and when it is False it uses ``int``.


    Examples
    ========

    >>> from sympy.utilities.misc import as_int
    >>> from sympy import sqrt, S

    The function is primarily concerned with sanitizing input for
    functions that need to work with builtin integers, so anything that
    is unambiguously an integer should be returned as an int:

    >>> as_int(S(3))
    3

    Floats, being of limited precision, are not assumed to be exact and
    will raise an error unless the ``strict`` flag is False. This
    precision issue becomes apparent for large floating point numbers:

    >>> big = 1e23
    >>> type(big) is float
    True
    >>> big == int(big)
    True
    >>> as_int(big)
    Traceback (most recent call last):
    ...
    ValueError: ... is not an integer
    >>> as_int(big, strict=False)
    99999999999999991611392

    Input that might be a complex representation of an integer value is
    also rejected by default:

    >>> one = sqrt(3 + 2*sqrt(2)) - sqrt(2)
    >>> int(one) == 1
    True
    >>> as_int(one)
    Traceback (most recent call last):
    ...
    ValueError: ... is not an integer
    """
    if strict:
        try:
            if isinstance(n, bool):
                raise TypeError
            return operator.index(n) # type: ignore
        except TypeError:
            raise ValueError('%s is not an integer' % (n,))
    else:
        try:
            result = int(n)
        except TypeError:
            raise ValueError('%s is not an integer' % (n,))
        if n - result: # type: ignore
            raise ValueError('%s is not an integer' % (n,))
        return result
