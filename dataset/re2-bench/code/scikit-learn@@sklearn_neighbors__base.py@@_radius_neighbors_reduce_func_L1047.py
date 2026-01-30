import numpy as np

class RadiusNeighborsMixin:
    """Mixin for radius-based neighbors searches."""

    def _radius_neighbors_reduce_func(self, dist, start, radius, return_distance):
        """Reduce a chunk of distances to the nearest neighbors.

        Callback to :func:`sklearn.metrics.pairwise.pairwise_distances_chunked`

        Parameters
        ----------
        dist : ndarray of shape (n_samples_chunk, n_samples)
            The distance matrix.

        start : int
            The index in X which the first row of dist corresponds to.

        radius : float
            The radius considered when making the nearest neighbors search.

        return_distance : bool
            Whether or not to return the distances.

        Returns
        -------
        dist : list of ndarray of shape (n_samples_chunk,)
            Returned only if `return_distance=True`.

        neigh : list of ndarray of shape (n_samples_chunk,)
            The neighbors indices.
        """
        neigh_ind = [np.where(d <= radius)[0] for d in dist]
        if return_distance:
            if self.effective_metric_ == 'euclidean':
                dist = [np.sqrt(d[neigh_ind[i]]) for i, d in enumerate(dist)]
            else:
                dist = [d[neigh_ind[i]] for i, d in enumerate(dist)]
            results = (dist, neigh_ind)
        else:
            results = neigh_ind
        return results
