from abc import ABC, abstractmethod
from numbers import Integral, Real
import numpy as np
import scipy.sparse as sp
from sklearn.base import (
    BaseEstimator,
    ClassNamePrefixFeaturesOutMixin,
    ClusterMixin,
    TransformerMixin,
    _fit_context,
)
from sklearn.cluster._k_means_common import (
    CHUNK_SIZE,
    _inertia_dense,
    _inertia_sparse,
    _is_same_clustering,
)
from sklearn.utils._param_validation import Interval, StrOptions, validate_params
from sklearn.utils.parallel import (
    _get_threadpool_controller,
    _threadpool_controller_decorator,
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

    @abstractmethod
    def _warn_mkl_vcomp(self, n_active_threads):
        """Issue an estimator specific warning when vcomp and mkl are both present

        This method is called by `_check_mkl_vcomp`.
        """

    def _check_mkl_vcomp(self, X, n_samples):
        """Check when vcomp and mkl are both present"""
        if sp.issparse(X):
            return
        n_active_threads = int(np.ceil(n_samples / CHUNK_SIZE))
        if n_active_threads < self._n_threads:
            modules = _get_threadpool_controller().info()
            has_vcomp = 'vcomp' in [module['prefix'] for module in modules]
            has_mkl = ('mkl', 'intel') in [(module['internal_api'], module.get('threading_layer', None)) for module in modules]
            if has_vcomp and has_mkl:
                self._warn_mkl_vcomp(n_active_threads)
