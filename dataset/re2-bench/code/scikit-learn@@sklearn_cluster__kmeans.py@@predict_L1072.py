from abc import ABC, abstractmethod
from numbers import Integral, Real
import numpy as np
from sklearn.base import (
    BaseEstimator,
    ClassNamePrefixFeaturesOutMixin,
    ClusterMixin,
    TransformerMixin,
    _fit_context,
)
from sklearn.utils._param_validation import Interval, StrOptions, validate_params
from sklearn.utils.validation import (
    _check_sample_weight,
    _is_arraylike_not_scalar,
    check_is_fitted,
    validate_data,
)

class _BaseKMeans(ClassNamePrefixFeaturesOutMixin, TransformerMixin, ClusterMixin, BaseEstimator, ABC):
    """Base class for KMeans and MiniBatchKMeans"""
    _parameter_constraints: dict = {'n_clusters': [Interval(Integral, 1, None, closed='left')], 'init': [StrOptions({'k-means++', 'random'}), callable, 'array-like'], 'n_init': [StrOptions({'auto'}), Interval(Integral, 1, None, closed='left')], 'max_iter': [Interval(Integral, 1, None, closed='left')], 'tol': [Interval(Real, 0, None, closed='left')], 'verbose': ['verbose'], 'random_state': ['random_state']}

    def __init__(self, n_clusters, *, init, n_init, max_iter, tol, verbose, random_state):
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.verbose = verbose
        self.random_state = random_state

    def _check_test_data(self, X):
        X = validate_data(self, X, accept_sparse='csr', reset=False, dtype=[np.float64, np.float32], order='C', accept_large_sparse=False)
        return X

    def predict(self, X):
        """Predict the closest cluster each sample in X belongs to.

        In the vector quantization literature, `cluster_centers_` is called
        the code book and each value returned by `predict` is the index of
        the closest code in the code book.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to predict.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        check_is_fitted(self)
        X = self._check_test_data(X)
        sample_weight = np.ones(X.shape[0], dtype=X.dtype)
        labels = _labels_inertia_threadpool_limit(X, sample_weight, self.cluster_centers_, n_threads=self._n_threads, return_inertia=False)
        return labels
