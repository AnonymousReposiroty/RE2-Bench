import numpy as np

def _extract_xi_labels(ordering, clusters):
    """Extracts the labels from the clusters returned by `_xi_cluster`.
    We rely on the fact that clusters are stored
    with the smaller clusters coming before the larger ones.

    Parameters
    ----------
    ordering : array-like of shape (n_samples,)
        The ordering of points calculated by OPTICS

    clusters : array-like of shape (n_clusters, 2)
        List of clusters i.e. (start, end) tuples,
        as returned by `_xi_cluster`.

    Returns
    -------
    labels : ndarray of shape (n_samples,)
    """

    labels = np.full(len(ordering), -1, dtype=int)
    label = 0
    for c in clusters:
        if not np.any(labels[c[0] : (c[1] + 1)] != -1):
            labels[c[0] : (c[1] + 1)] = label
            label += 1
    labels[ordering] = labels.copy()
    return labels
