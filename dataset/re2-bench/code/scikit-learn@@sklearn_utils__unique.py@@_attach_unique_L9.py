import numpy as np

def _attach_unique(y):
    """Attach unique values of y to y and return the result.

    The result is a view of y, and the metadata (unique) is not attached to y.
    """
    if not isinstance(y, np.ndarray):
        return y
    try:
        # avoid recalculating unique in nested calls.
        if "unique" in y.dtype.metadata:
            return y
    except (AttributeError, TypeError):
        pass

    unique = np.unique(y)
    unique_dtype = np.dtype(y.dtype, metadata={"unique": unique})
    return y.view(dtype=unique_dtype)
