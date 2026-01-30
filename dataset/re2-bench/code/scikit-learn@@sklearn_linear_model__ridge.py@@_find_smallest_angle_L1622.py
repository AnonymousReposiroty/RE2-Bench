from sklearn.utils._array_api import (
    _is_numpy_namespace,
    _max_precision_float_dtype,
    _ravel,
    device,
    get_namespace,
    get_namespace_and_device,
    move_to,
)

def _find_smallest_angle(query, vectors):
    """Find the column of vectors that is most aligned with the query.

    Both query and the columns of vectors must have their l2 norm equal to 1.

    Parameters
    ----------
    query : ndarray of shape (n_samples,)
        Normalized query vector.

    vectors : ndarray of shape (n_samples, n_features)
        Vectors to which we compare query, as columns. Must be normalized.
    """
    xp, _ = get_namespace(query)
    abs_cosine = xp.abs(query @ vectors)
    index = xp.argmax(abs_cosine)
    return index
