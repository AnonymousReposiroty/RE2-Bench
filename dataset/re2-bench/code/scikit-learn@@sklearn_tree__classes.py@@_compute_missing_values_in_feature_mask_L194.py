from abc import ABCMeta, abstractmethod
from numbers import Integral, Real
import numpy as np
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
from sklearn.tree._utils import _any_isnan_axis0
from sklearn.utils import (
    Bunch,
    check_random_state,
    compute_sample_weight,
    metadata_routing,
)
from sklearn.utils._param_validation import Hidden, Interval, RealNotInt, StrOptions
from sklearn.utils.validation import (
    _assert_all_finite_element_wise,
    _check_n_features,
    _check_sample_weight,
    assert_all_finite,
    check_is_fitted,
    validate_data,
)

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

    def _compute_missing_values_in_feature_mask(self, X, estimator_name=None):
        """Return boolean mask denoting if there are missing values for each feature.

        This method also ensures that X is finite.

        Parameter
        ---------
        X : array-like of shape (n_samples, n_features), dtype=DOUBLE
            Input data.

        estimator_name : str or None, default=None
            Name to use when raising an error. Defaults to the class name.

        Returns
        -------
        missing_values_in_feature_mask : ndarray of shape (n_features,), or None
            Missing value mask. If missing values are not supported or there
            are no missing values, return None.
        """
        estimator_name = estimator_name or self.__class__.__name__
        common_kwargs = dict(estimator_name=estimator_name, input_name='X')
        if not self._support_missing_values(X):
            assert_all_finite(X, **common_kwargs)
            return None
        with np.errstate(over='ignore'):
            overall_sum = np.sum(X)
        if not np.isfinite(overall_sum):
            _assert_all_finite_element_wise(X, xp=np, allow_nan=True, **common_kwargs)
        if not np.isnan(overall_sum):
            return None
        missing_values_in_feature_mask = _any_isnan_axis0(X)
        return missing_values_in_feature_mask

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = True
        return tags
