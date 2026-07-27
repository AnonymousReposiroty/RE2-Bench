from abc import ABC, abstractmethod
from numbers import Integral, Real
import numpy as np
from sklearn._loss.loss import (
    _LOSSES,
    BaseLoss,
    HalfBinomialLoss,
    HalfGammaLoss,
    HalfMultinomialLoss,
    HalfPoissonLoss,
    PinballLoss,
)
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    RegressorMixin,
    _fit_context,
    is_classifier,
)
from sklearn.ensemble._hist_gradient_boosting.common import G_H_DTYPE, X_DTYPE, Y_DTYPE
from sklearn.utils._missing import is_scalar_nan
from sklearn.utils._param_validation import Interval, RealNotInt, StrOptions

class BaseHistGradientBoosting(BaseEstimator, ABC):
    """Base class for histogram-based gradient boosting estimators."""
    _parameter_constraints: dict = {'loss': [BaseLoss], 'learning_rate': [Interval(Real, 0, None, closed='neither')], 'max_iter': [Interval(Integral, 1, None, closed='left')], 'max_leaf_nodes': [Interval(Integral, 2, None, closed='left'), None], 'max_depth': [Interval(Integral, 1, None, closed='left'), None], 'min_samples_leaf': [Interval(Integral, 1, None, closed='left')], 'l2_regularization': [Interval(Real, 0, None, closed='left')], 'max_features': [Interval(RealNotInt, 0, 1, closed='right')], 'monotonic_cst': ['array-like', dict, None], 'interaction_cst': [list, tuple, StrOptions({'pairwise', 'no_interactions'}), None], 'n_iter_no_change': [Interval(Integral, 1, None, closed='left')], 'validation_fraction': [Interval(RealNotInt, 0, 1, closed='neither'), Interval(Integral, 1, None, closed='left'), None], 'tol': [Interval(Real, 0, None, closed='left')], 'max_bins': [Interval(Integral, 2, 255, closed='both')], 'categorical_features': ['array-like', StrOptions({'from_dtype'}), None], 'warm_start': ['boolean'], 'early_stopping': [StrOptions({'auto'}), 'boolean'], 'scoring': [str, callable, None], 'verbose': ['verbose'], 'random_state': ['random_state']}

    @abstractmethod
    def __init__(self, loss, *, learning_rate, max_iter, max_leaf_nodes, max_depth, min_samples_leaf, l2_regularization, max_features, max_bins, categorical_features, monotonic_cst, interaction_cst, warm_start, early_stopping, scoring, validation_fraction, n_iter_no_change, tol, verbose, random_state):
        self.loss = loss
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.max_leaf_nodes = max_leaf_nodes
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.l2_regularization = l2_regularization
        self.max_features = max_features
        self.max_bins = max_bins
        self.monotonic_cst = monotonic_cst
        self.interaction_cst = interaction_cst
        self.categorical_features = categorical_features
        self.warm_start = warm_start
        self.early_stopping = early_stopping
        self.scoring = scoring
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        self.verbose = verbose
        self.random_state = random_state

    def _check_categories(self):
        """Check categories found by the preprocessor and return their encoded values.

        Returns a list of length ``self.n_features_in_``, with one entry per
        input feature.

        For non-categorical features, the corresponding entry is ``None``.

        For categorical features, the corresponding entry is an array
        containing the categories as encoded by the preprocessor (an
        ``OrdinalEncoder``), excluding missing values. The entry is therefore
        ``np.arange(n_categories)`` where ``n_categories`` is the number of
        unique values in the considered feature column, after removing missing
        values.

        If ``n_categories > self.max_bins`` for any feature, a ``ValueError``
        is raised.
        """
        encoder = self._preprocessor.named_transformers_['encoder']
        known_categories = [None] * self._preprocessor.n_features_in_
        categorical_column_indices = np.arange(self._preprocessor.n_features_in_)[self._preprocessor.output_indices_['encoder']]
        for feature_idx, categories in zip(categorical_column_indices, encoder.categories_):
            if len(categories) and is_scalar_nan(categories[-1]):
                categories = categories[:-1]
            if categories.size > self.max_bins:
                try:
                    feature_name = repr(encoder.feature_names_in_[feature_idx])
                except AttributeError:
                    feature_name = f'at index {feature_idx}'
                raise ValueError(f'Categorical feature {feature_name} is expected to have a cardinality <= {self.max_bins} but actually has a cardinality of {categories.size}.')
            known_categories[feature_idx] = np.arange(len(categories), dtype=X_DTYPE)
        return known_categories
