from abc import ABCMeta, abstractmethod
from numbers import Integral
import numpy as np
from sklearn.base import BaseEstimator, BiclusterMixin, _fit_context
from sklearn.cluster._kmeans import KMeans, MiniBatchKMeans
from sklearn.utils._param_validation import Interval, StrOptions

class BaseSpectral(BiclusterMixin, BaseEstimator, metaclass=ABCMeta):
    """Base class for spectral biclustering."""
    _parameter_constraints: dict = {'svd_method': [StrOptions({'randomized', 'arpack'})], 'n_svd_vecs': [Interval(Integral, 0, None, closed='left'), None], 'mini_batch': ['boolean'], 'init': [StrOptions({'k-means++', 'random'}), np.ndarray], 'n_init': [Interval(Integral, 1, None, closed='left')], 'random_state': ['random_state']}

    @abstractmethod
    def __init__(self, n_clusters=3, svd_method='randomized', n_svd_vecs=None, mini_batch=False, init='k-means++', n_init=10, random_state=None):
        self.n_clusters = n_clusters
        self.svd_method = svd_method
        self.n_svd_vecs = n_svd_vecs
        self.mini_batch = mini_batch
        self.init = init
        self.n_init = n_init
        self.random_state = random_state

    def _k_means(self, data, n_clusters):
        if self.mini_batch:
            model = MiniBatchKMeans(n_clusters, init=self.init, n_init=self.n_init, random_state=self.random_state)
        else:
            model = KMeans(n_clusters, init=self.init, n_init=self.n_init, random_state=self.random_state)
        model.fit(data)
        centroid = model.cluster_centers_
        labels = model.labels_
        return (centroid, labels)
