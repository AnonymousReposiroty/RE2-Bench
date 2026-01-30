import numpy as np

def _to_object_array(sequence):
    """Convert sequence to a 1-D NumPy array of object dtype.

    numpy.array constructor has a similar use but it's output
    is ambiguous. It can be 1-D NumPy array of object dtype if
    the input is a ragged array, but if the input is a list of
    equal length arrays, then the output is a 2D numpy.array.
    _to_object_array solves this ambiguity by guarantying that
    the output is a 1-D NumPy array of objects for any input.

    Parameters
    ----------
    sequence : array-like of shape (n_elements,)
        The sequence to be converted.

    Returns
    -------
    out : ndarray of shape (n_elements,), dtype=object
        The converted sequence into a 1-D NumPy array of object dtype.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.utils.validation import _to_object_array
    >>> _to_object_array([np.array([0]), np.array([1])])
    array([array([0]), array([1])], dtype=object)
    >>> _to_object_array([np.array([0]), np.array([1, 2])])
    array([array([0]), array([1, 2])], dtype=object)
    >>> _to_object_array([np.array([0]), np.array([1, 2])])
    array([array([0]), array([1, 2])], dtype=object)
    """
    out = np.empty(len(sequence), dtype=object)
    out[:] = sequence
    return out
