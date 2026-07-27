import numpy as np
from matplotlib import _api

def _pseudo_inv22sym_vectorized(M):
    """
    Inversion of arrays of (2, 2) SYMMETRIC matrices; returns the
    (Moore-Penrose) pseudo-inverse for rank-deficient matrices.

    In case M is of rank 1, we have M = trace(M) x P where P is the orthogonal
    projection on Im(M), and we return trace(M)^-1 x P == M / trace(M)**2
    In case M is of rank 0, we return the null matrix.

    *M* : array of (2, 2) matrices to inverse, shape (n, 2, 2)
    """
    _api.check_shape((None, 2, 2), M=M)
    M_inv = np.empty_like(M)
    prod1 = M[:, 0, 0]*M[:, 1, 1]
    delta = prod1 - M[:, 0, 1]*M[:, 1, 0]
    rank2 = (np.abs(delta) > 1e-8*np.abs(prod1))

    if np.all(rank2):
        # Normal 'optimized' flow.
        M_inv[:, 0, 0] = M[:, 1, 1] / delta
        M_inv[:, 0, 1] = -M[:, 0, 1] / delta
        M_inv[:, 1, 0] = -M[:, 1, 0] / delta
        M_inv[:, 1, 1] = M[:, 0, 0] / delta
    else:
        # 'Pathologic' flow.
        # Here we have to deal with 2 sub-cases
        # 1) First sub-case: matrices of rank 2:
        delta = delta[rank2]
        M_inv[rank2, 0, 0] = M[rank2, 1, 1] / delta
        M_inv[rank2, 0, 1] = -M[rank2, 0, 1] / delta
        M_inv[rank2, 1, 0] = -M[rank2, 1, 0] / delta
        M_inv[rank2, 1, 1] = M[rank2, 0, 0] / delta
        # 2) Second sub-case: rank-deficient matrices of rank 0 and 1:
        rank01 = ~rank2
        tr = M[rank01, 0, 0] + M[rank01, 1, 1]
        tr_zeros = (np.abs(tr) < 1.e-8)
        sq_tr_inv = (1.-tr_zeros) / (tr**2+tr_zeros)
        # sq_tr_inv = 1. / tr**2
        M_inv[rank01, 0, 0] = M[rank01, 0, 0] * sq_tr_inv
        M_inv[rank01, 0, 1] = M[rank01, 0, 1] * sq_tr_inv
        M_inv[rank01, 1, 0] = M[rank01, 1, 0] * sq_tr_inv
        M_inv[rank01, 1, 1] = M[rank01, 1, 1] * sq_tr_inv

    return M_inv
