from sklearn.utils._array_api import get_namespace
from sklearn.utils._unique import attach_unique, cached_unique

def _unique_multiclass(y, xp=None):
    xp, is_array_api_compliant = get_namespace(y, xp=xp)
    if hasattr(y, "__array__") or is_array_api_compliant:
        return cached_unique(xp.asarray(y), xp=xp)
    else:
        return set(y)
