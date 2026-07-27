import numpy
from sklearn.utils._unique import cached_unique

def _union1d(a, b, xp):
    if _is_numpy_namespace(xp):
        # avoid circular import
        from sklearn.utils._unique import cached_unique

        a_unique, b_unique = cached_unique(a, b, xp=xp)
        return xp.asarray(numpy.union1d(a_unique, b_unique))
    assert a.ndim == b.ndim == 1
    return xp.unique_values(xp.concat([xp.unique_values(a), xp.unique_values(b)]))
