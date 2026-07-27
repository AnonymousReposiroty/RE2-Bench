import numpy as np
from sklearn.utils.validation import _to_object_array, check_is_fitted, validate_data

def _radius_neighbors_from_graph(graph, radius, return_distance):
    """Decompose a nearest neighbors sparse graph into distances and indices.

    Parameters
    ----------
    graph : sparse matrix of shape (n_samples, n_samples)
        Neighbors graph as given by `kneighbors_graph` or
        `radius_neighbors_graph`. Matrix should be of format CSR format.

    radius : float
        Radius of neighborhoods which should be strictly positive.

    return_distance : bool
        Whether or not to return the distances.

    Returns
    -------
    neigh_dist : ndarray of shape (n_samples,) of arrays
        Distances to nearest neighbors. Only present if `return_distance=True`.

    neigh_ind : ndarray of shape (n_samples,) of arrays
        Indices of nearest neighbors.
    """
    assert graph.format == "csr"

    no_filter_needed = bool(graph.data.max() <= radius)

    if no_filter_needed:
        data, indices, indptr = graph.data, graph.indices, graph.indptr
    else:
        mask = graph.data <= radius
        if return_distance:
            data = np.compress(mask, graph.data)
        indices = np.compress(mask, graph.indices)
        indptr = np.concatenate(([0], np.cumsum(mask)))[graph.indptr]

    indices = indices.astype(np.intp, copy=no_filter_needed)

    if return_distance:
        neigh_dist = _to_object_array(np.split(data, indptr[1:-1]))
    neigh_ind = _to_object_array(np.split(indices, indptr[1:-1]))

    if return_distance:
        return neigh_dist, neigh_ind
    else:
        return neigh_ind
