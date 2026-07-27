import numpy as np
from scipy.sparse import csgraph, issparse

def _get_finite_row_indices(matrix):
    """
    Returns the indices of the purely finite rows of a
    sparse matrix or dense ndarray
    """
    if issparse(matrix):
        row_indices = np.array(
            [i for i, row in enumerate(matrix.tolil().data) if np.all(np.isfinite(row))]
        )
    else:
        (row_indices,) = np.isfinite(matrix.sum(axis=1)).nonzero()
    return row_indices
