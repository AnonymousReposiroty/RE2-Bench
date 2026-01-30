import collections
import collections.abc
import math
import numpy as np

def _safe_first_finite(obj):
    """
    Return the first finite element in *obj* if one is available and skip_nonfinite is
    True. Otherwise, return the first element.

    This is a method for internal use.

    This is a type-independent way of obtaining the first finite element, supporting
    both index access and the iterator protocol.
    """
    def safe_isfinite(val):
        if val is None:
            return False
        try:
            return math.isfinite(val)
        except (TypeError, ValueError):
            # if the outer object is 2d, then val is a 1d array, and
            # - math.isfinite(numpy.zeros(3)) raises TypeError
            # - math.isfinite(torch.zeros(3)) raises ValueError
            pass
        try:
            return np.isfinite(val) if np.isscalar(val) else True
        except TypeError:
            # This is something that NumPy cannot make heads or tails of,
            # assume "finite"
            return True

    if isinstance(obj, np.flatiter):
        # TODO do the finite filtering on this
        return obj[0]
    elif isinstance(obj, collections.abc.Iterator):
        raise RuntimeError("matplotlib does not support generators as input")
    else:
        for val in obj:
            if safe_isfinite(val):
                return val
        return safe_first_element(obj)
