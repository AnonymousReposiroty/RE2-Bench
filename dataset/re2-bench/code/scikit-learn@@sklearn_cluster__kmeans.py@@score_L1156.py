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

    def score(self, X, y=None, sample_weight=None):
        """Opposite of the value of X on the K-means objective.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data.

        y : Ignored
            Not used, present here for API consistency by convention.

        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X. If None, all observations
            are assigned equal weight.

        Returns
        -------
        score : float
            Opposite of the value of X on the K-means objective.
        """
        check_is_fitted(self)
        X = self._check_test_data(X)
        sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)
        _, scores = _labels_inertia_threadpool_limit(X, sample_weight, self.cluster_centers_, self._n_threads)
        return -scores
