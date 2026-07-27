from scipy.sparse import csr_matrix, issparse
from sklearn.utils import check_array, gen_even_slices, get_tags

def _check_precomputed(X):
    """Check precomputed distance matrix.

    If the precomputed distance matrix is sparse, it checks that the non-zero
    entries are sorted by distances. If not, the matrix is copied and sorted.

    Parameters
    ----------
    X : {sparse matrix, array-like}, (n_samples, n_samples)
        Distance matrix to other samples. X may be a sparse matrix, in which
        case only non-zero elements may be considered neighbors.

    Returns
    -------
    X : {sparse matrix, array-like}, (n_samples, n_samples)
        Distance matrix to other samples. X may be a sparse matrix, in which
        case only non-zero elements may be considered neighbors.
    """
    if not issparse(X):
        X = check_array(X, ensure_non_negative=True, input_name="X")
        return X
    else:
        graph = X

    if graph.format not in ("csr", "csc", "coo", "lil"):
        raise TypeError(
            "Sparse matrix in {!r} format is not supported due to "
            "its handling of explicit zeros".format(graph.format)
        )
    copied = graph.format != "csr"
    graph = check_array(
        graph,
        accept_sparse="csr",
        ensure_non_negative=True,
        input_name="precomputed distance matrix",
    )
    graph = sort_graph_by_row_values(graph, copy=not copied, warn_when_not_sorted=True)

    return graph
