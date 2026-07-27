import numpy as np

def _gs_decorrelation(w, W, j):
    """
    Orthonormalize w wrt the first j rows of W.

    Parameters
    ----------
    w : ndarray of shape (n,)
        Array to be orthogonalized

    W : ndarray of shape (p, n)
        Null space definition

    j : int < p
        The no of (from the first) rows of Null space W wrt which w is
        orthogonalized.

    Notes
    -----
    Assumes that W is orthogonal
    w changed in place
    """
    w -= np.linalg.multi_dot([w, W[:j].T, W[:j]])
    return w
