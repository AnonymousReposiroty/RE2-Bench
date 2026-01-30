import itertools
from abc import ABCMeta, abstractmethod
from numbers import Integral, Real
from sklearn.base import BaseEstimator, MultiOutputMixin, is_classifier
from sklearn.utils._param_validation import Interval, StrOptions, validate_params

class NeighborsBase(MultiOutputMixin, BaseEstimator, metaclass=ABCMeta):
    """Base class for nearest neighbors estimators."""
    _parameter_constraints: dict = {'n_neighbors': [Interval(Integral, 1, None, closed='left'), None], 'radius': [Interval(Real, 0, None, closed='both'), None], 'algorithm': [StrOptions({'auto', 'ball_tree', 'kd_tree', 'brute'})], 'leaf_size': [Interval(Integral, 1, None, closed='left')], 'p': [Interval(Real, 0, None, closed='right'), None], 'metric': [StrOptions(set(itertools.chain(*VALID_METRICS.values()))), callable], 'metric_params': [dict, None], 'n_jobs': [Integral, None]}

    @abstractmethod
    def __init__(self, n_neighbors=None, radius=None, algorithm='auto', leaf_size=30, metric='minkowski', p=2, metric_params=None, n_jobs=None):
        self.n_neighbors = n_neighbors
        self.radius = radius
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.metric = metric
        self.metric_params = metric_params
        self.p = p
        self.n_jobs = n_jobs

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = True
        tags.input_tags.pairwise = self.metric == 'precomputed'
        tags.input_tags.positive_only = tags.input_tags.pairwise
        tags.input_tags.allow_nan = self.metric == 'nan_euclidean'
        return tags
