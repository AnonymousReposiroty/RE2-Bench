from decimal import Decimal
from numbers import Number
import numpy as np
from numpy import ma

def _is_natively_supported(x):
    """
    Return whether *x* is of a type that Matplotlib natively supports or an
    array of objects of such types.
    """
    # Matplotlib natively supports all number types except Decimal.
    if np.iterable(x):
        # Assume lists are homogeneous as other functions in unit system.
        for thisx in x:
            if thisx is ma.masked:
                continue
            return isinstance(thisx, Number) and not isinstance(thisx, Decimal)
    else:
        return isinstance(x, Number) and not isinstance(x, Decimal)
