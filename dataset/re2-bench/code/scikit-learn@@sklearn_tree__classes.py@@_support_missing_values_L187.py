from abc import ABCMeta, abstractmethod
from numbers import Integral, Real
from scipy.sparse import issparse
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    MultiOutputMixin,
    RegressorMixin,
    _fit_context,
    clone,
    is_classifier,
)
from sklearn.utils import (
    Bunch,
    check_random_state,
    compute_sample_weight,
    metadata_routing,
)
from sklearn.utils._param_validation import Hidden, Interval, RealNotInt, StrOptions

class BaseDecisionTree(MultiOutputMixin, BaseEstimator, metaclass=ABCMeta):
    """Base class for decision trees.

    Warning: This class should not be used directly.
    Use derived classes instead.
    """
    __metadata_request__predict = {'check_input': metadata_routing.UNUSED}
    _parameter_constraints: dict = {'splitter': [StrOptions({'best', 'random'})], 'max_depth': [Interval(Integral, 1, None, closed='left'), None], 'min_samples_split': [Interval(Integral, 2, None, closed='left'), Interval(RealNotInt, 0.0, 1.0, closed='right')], 'min_samples_leaf': [Interval(Integral, 1, None, closed='left'), Interval(RealNotInt, 0.0, 1.0, closed='neither')], 'min_weight_fraction_leaf': [Interval(Real, 0.0, 0.5, closed='both')], 'max_features': [Interval(Integral, 1, None, closed='left'), Interval(RealNotInt, 0.0, 1.0, closed='right'), StrOptions({'sqrt', 'log2'}), None], 'random_state': ['random_state'], 'max_leaf_nodes': [Interval(Integral, 2, None, closed='left'), None], 'min_impurity_decrease': [Interval(Real, 0.0, None, closed='left')], 'ccp_alpha': [Interval(Real, 0.0, None, closed='left')], 'monotonic_cst': ['array-like', None]}

    @abstractmethod
    def __init__(self, *, criterion, splitter, max_depth, min_samples_split, min_samples_leaf, min_weight_fraction_leaf, max_features, max_leaf_nodes, random_state, min_impurity_decrease, class_weight=None, ccp_alpha=0.0, monotonic_cst=None):
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.random_state = random_state
        self.min_impurity_decrease = min_impurity_decrease
        self.class_weight = class_weight
        self.ccp_alpha = ccp_alpha
        self.monotonic_cst = monotonic_cst

    def _support_missing_values(self, X):
        return not issparse(X) and self.__sklearn_tags__().input_tags.allow_nan and (self.monotonic_cst is None)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = True
        return tags
