import numpy as np
from matplotlib import _api

def _safe_inv22_vectorized(M):
    """
    Inversion of arrays of (2, 2) matrices, returns 0 for rank-deficient
    matrices.

    *M* : array of (2, 2) matrices to inverse, shape (n, 2, 2)
    """
    _api.check_shape((None, 2, 2), M=M)
    M_inv = np.empty_like(M)
    prod1 = M[:, 0, 0]*M[:, 1, 1]
    delta = prod1 - M[:, 0, 1]*M[:, 1, 0]

    # We set delta_inv to 0. in case of a rank deficient matrix; a
    # rank-deficient input matrix *M* will lead to a null matrix in output
    rank2 = (np.abs(delta) > 1e-8*np.abs(prod1))
    if np.all(rank2):
        # Normal 'optimized' flow.
        delta_inv = 1./delta
    else:
        # 'Pathologic' flow.
        delta_inv = np.zeros(M.shape[0])
        delta_inv[rank2] = 1./delta[rank2]

    M_inv[:, 0, 0] = M[:, 1, 1]*delta_inv
    M_inv[:, 0, 1] = -M[:, 0, 1]*delta_inv
    M_inv[:, 1, 0] = -M[:, 1, 0]*delta_inv
    M_inv[:, 1, 1] = M[:, 0, 0]*delta_inv
    return M_inv
