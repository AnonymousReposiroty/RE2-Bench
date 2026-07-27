import numpy as np
from scipy.linalg import pinv, svd

def _get_first_singular_vectors_svd(X, y):
    """Return the first left and right singular vectors of X'y.

    Here the whole SVD is computed.
    """
    C = np.dot(X.T, y)
    U, _, Vt = svd(C, full_matrices=False)
    return U[:, 0], Vt[0, :]
