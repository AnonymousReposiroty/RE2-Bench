import numpy as np
from scipy.sparse import csgraph, issparse
from sklearn.cluster._hdbscan._linkage import (
    MST_edge_dtype,
    make_single_linkage,
    mst_from_data_matrix,
    mst_from_mutual_reachability,
)

def _brute_mst(mutual_reachability, min_samples):
    """
    Builds a minimum spanning tree (MST) from the provided mutual-reachability
    values. This function dispatches to a custom Cython implementation for
    dense arrays, and `scipy.sparse.csgraph.minimum_spanning_tree` for sparse
    arrays/matrices.

    Parameters
    ----------
    mututal_reachability_graph: {ndarray, sparse matrix} of shape \
            (n_samples, n_samples)
        Weighted adjacency matrix of the mutual reachability graph.

    min_samples : int, default=None
        The number of samples in a neighborhood for a point
        to be considered as a core point. This includes the point itself.

    Returns
    -------
    mst : ndarray of shape (n_samples - 1,), dtype=MST_edge_dtype
        The MST representation of the mutual-reachability graph. The MST is
        represented as a collection of edges.
    """
    if not issparse(mutual_reachability):
        return mst_from_mutual_reachability(mutual_reachability)

    # Check if the mutual reachability matrix has any rows which have
    # less than `min_samples` non-zero elements.
    indptr = mutual_reachability.indptr
    num_points = mutual_reachability.shape[0]
    if any((indptr[i + 1] - indptr[i]) < min_samples for i in range(num_points)):
        raise ValueError(
            f"There exists points with fewer than {min_samples} neighbors. Ensure"
            " your distance matrix has non-zero values for at least"
            f" `min_sample`={min_samples} neighbors for each points (i.e. K-nn"
            " graph), or specify a `max_distance` in `metric_params` to use when"
            " distances are missing."
        )
    # Check connected component on mutual reachability.
    # If more than one connected component is present,
    # it means that the graph is disconnected.
    n_components = csgraph.connected_components(
        mutual_reachability, directed=False, return_labels=False
    )
    if n_components > 1:
        raise ValueError(
            f"Sparse mutual reachability matrix has {n_components} connected"
            " components. HDBSCAN cannot be performed on a disconnected graph. Ensure"
            " that the sparse distance matrix has only one connected component."
        )

    # Compute the minimum spanning tree for the sparse graph
    sparse_min_spanning_tree = csgraph.minimum_spanning_tree(mutual_reachability)
    rows, cols = sparse_min_spanning_tree.nonzero()
    mst = np.rec.fromarrays(
        [rows, cols, sparse_min_spanning_tree.data],
        dtype=MST_edge_dtype,
    )
    return mst
