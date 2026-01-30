import numbers
from functools import partial
import numpy as np
from joblib import effective_n_jobs
from scipy.sparse import csr_matrix, issparse
from sklearn.metrics import DistanceMetric, pairwise_distances_chunked
from sklearn.metrics._pairwise_distances_reduction import ArgKmin, RadiusNeighbors
from sklearn.utils import check_array, gen_even_slices, get_tags
from sklearn.utils.parallel import Parallel, delayed
from sklearn.utils.validation import _to_object_array, check_is_fitted, validate_data

class KNeighborsMixin:
    """Mixin for k-neighbors searches."""

    def _kneighbors_reduce_func(self, dist, start, n_neighbors, return_distance):
        """Reduce a chunk of distances to the nearest neighbors.

        Callback to :func:`sklearn.metrics.pairwise.pairwise_distances_chunked`

        Parameters
        ----------
        dist : ndarray of shape (n_samples_chunk, n_samples)
            The distance matrix.

        start : int
            The index in X which the first row of dist corresponds to.

        n_neighbors : int
            Number of neighbors required for each sample.

        return_distance : bool
            Whether or not to return the distances.

        Returns
        -------
        dist : array of shape (n_samples_chunk, n_neighbors)
            Returned only if `return_distance=True`.

        neigh : array of shape (n_samples_chunk, n_neighbors)
            The neighbors indices.
        """
        sample_range = np.arange(dist.shape[0])[:, None]
        neigh_ind = np.argpartition(dist, n_neighbors - 1, axis=1)
        neigh_ind = neigh_ind[:, :n_neighbors]
        neigh_ind = neigh_ind[sample_range, np.argsort(dist[sample_range, neigh_ind])]
        if return_distance:
            if self.effective_metric_ == 'euclidean':
                result = (np.sqrt(dist[sample_range, neigh_ind]), neigh_ind)
            else:
                result = (dist[sample_range, neigh_ind], neigh_ind)
        else:
            result = neigh_ind
        return result

    def kneighbors(self, X=None, n_neighbors=None, return_distance=True):
        """Find the K-neighbors of a point.

        Returns indices of and distances to the neighbors of each point.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_queries, n_features),             or (n_queries, n_indexed) if metric == 'precomputed', default=None
            The query point or points.
            If not provided, neighbors of each indexed point are returned.
            In this case, the query point is not considered its own neighbor.

        n_neighbors : int, default=None
            Number of neighbors required for each sample. The default is the
            value passed to the constructor.

        return_distance : bool, default=True
            Whether or not to return the distances.

        Returns
        -------
        neigh_dist : ndarray of shape (n_queries, n_neighbors)
            Array representing the lengths to points, only present if
            return_distance=True.

        neigh_ind : ndarray of shape (n_queries, n_neighbors)
            Indices of the nearest points in the population matrix.

        Examples
        --------
        In the following example, we construct a NearestNeighbors
        class from an array representing our data set and ask who's
        the closest point to [1,1,1]

        >>> samples = [[0., 0., 0.], [0., .5, 0.], [1., 1., .5]]
        >>> from sklearn.neighbors import NearestNeighbors
        >>> neigh = NearestNeighbors(n_neighbors=1)
        >>> neigh.fit(samples)
        NearestNeighbors(n_neighbors=1)
        >>> print(neigh.kneighbors([[1., 1., 1.]]))
        (array([[0.5]]), array([[2]]))

        As you can see, it returns [[0.5]], and [[2]], which means that the
        element is at distance 0.5 and is the third element of samples
        (indexes start at 0). You can also query for multiple points:

        >>> X = [[0., 1., 0.], [1., 0., 1.]]
        >>> neigh.kneighbors(X, return_distance=False)
        array([[1],
               [2]]...)
        """
        check_is_fitted(self)
        if n_neighbors is None:
            n_neighbors = self.n_neighbors
        elif n_neighbors <= 0:
            raise ValueError('Expected n_neighbors > 0. Got %d' % n_neighbors)
        elif not isinstance(n_neighbors, numbers.Integral):
            raise TypeError('n_neighbors does not take %s value, enter integer value' % type(n_neighbors))
        ensure_all_finite = 'allow-nan' if get_tags(self).input_tags.allow_nan else True
        query_is_train = X is None
        if query_is_train:
            X = self._fit_X
            n_neighbors += 1
        elif self.metric == 'precomputed':
            X = _check_precomputed(X)
        else:
            X = validate_data(self, X, ensure_all_finite=ensure_all_finite, accept_sparse='csr', reset=False, order='C')
        n_samples_fit = self.n_samples_fit_
        if n_neighbors > n_samples_fit:
            if query_is_train:
                n_neighbors -= 1
                inequality_str = 'n_neighbors < n_samples_fit'
            else:
                inequality_str = 'n_neighbors <= n_samples_fit'
            raise ValueError(f'Expected {inequality_str}, but n_neighbors = {n_neighbors}, n_samples_fit = {n_samples_fit}, n_samples = {X.shape[0]}')
        n_jobs = effective_n_jobs(self.n_jobs)
        chunked_results = None
        use_pairwise_distances_reductions = self._fit_method == 'brute' and ArgKmin.is_usable_for(X if X is not None else self._fit_X, self._fit_X, self.effective_metric_)
        if use_pairwise_distances_reductions:
            results = ArgKmin.compute(X=X, Y=self._fit_X, k=n_neighbors, metric=self.effective_metric_, metric_kwargs=self.effective_metric_params_, strategy='auto', return_distance=return_distance)
        elif self._fit_method == 'brute' and self.metric == 'precomputed' and issparse(X):
            results = _kneighbors_from_graph(X, n_neighbors=n_neighbors, return_distance=return_distance)
        elif self._fit_method == 'brute':
            reduce_func = partial(self._kneighbors_reduce_func, n_neighbors=n_neighbors, return_distance=return_distance)
            if self.effective_metric_ == 'euclidean':
                kwds = {'squared': True}
            else:
                kwds = self.effective_metric_params_
            chunked_results = list(pairwise_distances_chunked(X, self._fit_X, reduce_func=reduce_func, metric=self.effective_metric_, n_jobs=n_jobs, **kwds))
        elif self._fit_method in ['ball_tree', 'kd_tree']:
            if issparse(X):
                raise ValueError("%s does not work with sparse matrices. Densify the data, or set algorithm='brute'" % self._fit_method)
            chunked_results = Parallel(n_jobs, prefer='threads')((delayed(self._tree.query)(X[s], n_neighbors, return_distance) for s in gen_even_slices(X.shape[0], n_jobs)))
        else:
            raise ValueError('internal: _fit_method not recognized')
        if chunked_results is not None:
            if return_distance:
                neigh_dist, neigh_ind = zip(*chunked_results)
                results = (np.vstack(neigh_dist), np.vstack(neigh_ind))
            else:
                results = np.vstack(chunked_results)
        if not query_is_train:
            return results
        else:
            if return_distance:
                neigh_dist, neigh_ind = results
            else:
                neigh_ind = results
            n_queries, _ = X.shape
            sample_range = np.arange(n_queries)[:, None]
            sample_mask = neigh_ind != sample_range
            dup_gr_nbrs = np.all(sample_mask, axis=1)
            sample_mask[:, 0][dup_gr_nbrs] = False
            neigh_ind = np.reshape(neigh_ind[sample_mask], (n_queries, n_neighbors - 1))
            if return_distance:
                neigh_dist = np.reshape(neigh_dist[sample_mask], (n_queries, n_neighbors - 1))
                return (neigh_dist, neigh_ind)
            return neigh_ind

    def kneighbors_graph(self, X=None, n_neighbors=None, mode='connectivity'):
        """Compute the (weighted) graph of k-Neighbors for points in X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_queries, n_features),             or (n_queries, n_indexed) if metric == 'precomputed', default=None
            The query point or points.
            If not provided, neighbors of each indexed point are returned.
            In this case, the query point is not considered its own neighbor.
            For ``metric='precomputed'`` the shape should be
            (n_queries, n_indexed). Otherwise the shape should be
            (n_queries, n_features).

        n_neighbors : int, default=None
            Number of neighbors for each sample. The default is the value
            passed to the constructor.

        mode : {'connectivity', 'distance'}, default='connectivity'
            Type of returned matrix: 'connectivity' will return the
            connectivity matrix with ones and zeros, in 'distance' the
            edges are distances between points, type of distance
            depends on the selected metric parameter in
            NearestNeighbors class.

        Returns
        -------
        A : sparse-matrix of shape (n_queries, n_samples_fit)
            `n_samples_fit` is the number of samples in the fitted data.
            `A[i, j]` gives the weight of the edge connecting `i` to `j`.
            The matrix is of CSR format.

        See Also
        --------
        NearestNeighbors.radius_neighbors_graph : Compute the (weighted) graph
            of Neighbors for points in X.

        Examples
        --------
        >>> X = [[0], [3], [1]]
        >>> from sklearn.neighbors import NearestNeighbors
        >>> neigh = NearestNeighbors(n_neighbors=2)
        >>> neigh.fit(X)
        NearestNeighbors(n_neighbors=2)
        >>> A = neigh.kneighbors_graph(X)
        >>> A.toarray()
        array([[1., 0., 1.],
               [0., 1., 1.],
               [1., 0., 1.]])
        """
        check_is_fitted(self)
        if n_neighbors is None:
            n_neighbors = self.n_neighbors
        if mode == 'connectivity':
            A_ind = self.kneighbors(X, n_neighbors, return_distance=False)
            n_queries = A_ind.shape[0]
            A_data = np.ones(n_queries * n_neighbors)
        elif mode == 'distance':
            A_data, A_ind = self.kneighbors(X, n_neighbors, return_distance=True)
            A_data = np.ravel(A_data)
        else:
            raise ValueError(f'Unsupported mode, must be one of "connectivity", or "distance" but got "{mode}" instead')
        n_queries = A_ind.shape[0]
        n_samples_fit = self.n_samples_fit_
        n_nonzero = n_queries * n_neighbors
        A_indptr = np.arange(0, n_nonzero + 1, n_neighbors)
        kneighbors_graph = csr_matrix((A_data, A_ind.ravel(), A_indptr), shape=(n_queries, n_samples_fit))
        return kneighbors_graph
