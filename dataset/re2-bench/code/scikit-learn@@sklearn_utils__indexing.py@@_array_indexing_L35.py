import numpy as np
from scipy.sparse import issparse
from sklearn.utils._array_api import (
    _is_numpy_namespace,
    get_namespace,
    get_namespace_and_device,
    move_to,
)

def _array_indexing(array, key, key_dtype, axis):
    """Index an array or scipy.sparse consistently across NumPy version."""
    xp, is_array_api, device_ = get_namespace_and_device(array)
    if is_array_api:
        key = move_to(key, xp=xp, device=device_)
        return xp.take(array, key, axis=axis)
    if issparse(array) and key_dtype == "bool":
        key = np.asarray(key)
    if isinstance(key, tuple):
        key = list(key)
    return array[key, ...] if axis == 0 else array[:, key]
