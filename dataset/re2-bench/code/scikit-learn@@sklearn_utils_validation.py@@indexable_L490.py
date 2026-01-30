def indexable(*iterables):
    """Make arrays indexable for cross-validation.

    Checks consistent length, passes through None, and ensures that everything
    can be indexed by converting sparse matrices to csr and converting
    non-iterable objects to arrays.

    Parameters
    ----------
    *iterables : {lists, dataframes, ndarrays, sparse matrices}
        List of objects to ensure sliceability.

    Returns
    -------
    result : list of {ndarray, sparse matrix, dataframe} or None
        Returns a list containing indexable arrays (i.e. NumPy array,
        sparse matrix, or dataframe) or `None`.

    Examples
    --------
    >>> from sklearn.utils import indexable
    >>> from scipy.sparse import csr_matrix
    >>> import numpy as np
    >>> iterables = [
    ...     [1, 2, 3], np.array([2, 3, 4]), None, csr_matrix([[5], [6], [7]])
    ... ]
    >>> indexable(*iterables)
    [[1, 2, 3], array([2, 3, 4]), None, <...Sparse...dtype 'int64'...shape (3, 1)>]
    """

    result = [_make_indexable(X) for X in iterables]
    check_consistent_length(*result)
    return result
