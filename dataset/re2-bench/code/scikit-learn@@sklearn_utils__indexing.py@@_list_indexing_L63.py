from itertools import compress, islice
import numpy as np

def _list_indexing(X, key, key_dtype):
    """Index a Python list."""
    if np.isscalar(key) or isinstance(key, slice):
        # key is a slice or a scalar
        return X[key]
    if key_dtype == "bool":
        # key is a boolean array-like
        return list(compress(X, key))
    # key is an integer array-like of key
    return [X[idx] for idx in key]
