from scipy import sparse
from scipy.sparse.csgraph import connected_components
from sklearn.utils import check_array, check_random_state, check_symmetric
from sklearn.utils.fixes import parse_version, sp_version

def _graph_is_connected(graph):
    """Return whether the graph is connected (True) or Not (False).

    Parameters
    ----------
    graph : {array-like, sparse matrix} of shape (n_samples, n_samples)
        Adjacency matrix of the graph, non-zero weight means an edge
        between the nodes.

    Returns
    -------
    is_connected : bool
        True means the graph is fully connected and False means not.
    """
    if sparse.issparse(graph):
        # Before Scipy 1.11.3, `connected_components` only supports 32-bit indices.
        # PR: https://github.com/scipy/scipy/pull/18913
        # First integration in 1.11.3: https://github.com/scipy/scipy/pull/19279
        # TODO(jjerphan): Once SciPy 1.11.3 is the minimum supported version, use
        # `accept_large_sparse=True`.
        accept_large_sparse = sp_version >= parse_version("1.11.3")
        graph = check_array(
            graph, accept_sparse=True, accept_large_sparse=accept_large_sparse
        )
        # sparse graph, find all the connected components
        n_connected_components, _ = connected_components(graph)
        return n_connected_components == 1
    else:
        # dense graph, find all connected components start from node 0
        return _graph_connected_component(graph, 0).sum() == graph.shape[0]
