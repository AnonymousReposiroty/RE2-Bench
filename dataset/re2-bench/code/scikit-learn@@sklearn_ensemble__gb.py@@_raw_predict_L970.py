from abc import ABCMeta, abstractmethod
from numbers import Integral, Real
import numpy as np
from sklearn.base import ClassifierMixin, RegressorMixin, _fit_context, is_classifier
from sklearn.ensemble._base import BaseEnsemble
from sklearn.ensemble._gradient_boosting import (
    _random_sample_mask,
    predict_stage,
    predict_stages,
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils._param_validation import HasMethods, Interval, StrOptions
from sklearn.utils.validation import (
    _check_sample_weight,
    check_is_fitted,
    validate_data,
)

class BaseGradientBoosting(BaseEnsemble, metaclass=ABCMeta):
    """Abstract base class for Gradient Boosting."""
    _parameter_constraints: dict = {**DecisionTreeRegressor._parameter_constraints, 'learning_rate': [Interval(Real, 0.0, None, closed='left')], 'n_estimators': [Interval(Integral, 1, None, closed='left')], 'criterion': [StrOptions({'friedman_mse', 'squared_error'})], 'subsample': [Interval(Real, 0.0, 1.0, closed='right')], 'verbose': ['verbose'], 'warm_start': ['boolean'], 'validation_fraction': [Interval(Real, 0.0, 1.0, closed='neither')], 'n_iter_no_change': [Interval(Integral, 1, None, closed='left'), None], 'tol': [Interval(Real, 0.0, None, closed='left')]}
    _parameter_constraints.pop('splitter')
    _parameter_constraints.pop('monotonic_cst')

    @abstractmethod
    def __init__(self, *, loss, learning_rate, n_estimators, criterion, min_samples_split, min_samples_leaf, min_weight_fraction_leaf, max_depth, min_impurity_decrease, init, subsample, max_features, ccp_alpha, random_state, alpha=0.9, verbose=0, max_leaf_nodes=None, warm_start=False, validation_fraction=0.1, n_iter_no_change=None, tol=0.0001):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.loss = loss
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.subsample = subsample
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha = ccp_alpha
        self.init = init
        self.random_state = random_state
        self.alpha = alpha
        self.verbose = verbose
        self.max_leaf_nodes = max_leaf_nodes
        self.warm_start = warm_start
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol

    def _check_initialized(self):
        """Check that the estimator is initialized, raising an error if not."""
        check_is_fitted(self)

    def _raw_predict_init(self, X):
        """Check input and compute raw predictions of the init estimator."""
        self._check_initialized()
        X = self.estimators_[0, 0]._validate_X_predict(X, check_input=True)
        if self.init_ == 'zero':
            raw_predictions = np.zeros(shape=(X.shape[0], self.n_trees_per_iteration_), dtype=np.float64)
        else:
            raw_predictions = _init_raw_predictions(X, self.init_, self._loss, is_classifier(self))
        return raw_predictions

    def _raw_predict(self, X):
        """Return the sum of the trees raw predictions (+ init estimator)."""
        check_is_fitted(self)
        raw_predictions = self._raw_predict_init(X)
        predict_stages(self.estimators_, X, self.learning_rate, raw_predictions)
        return raw_predictions
