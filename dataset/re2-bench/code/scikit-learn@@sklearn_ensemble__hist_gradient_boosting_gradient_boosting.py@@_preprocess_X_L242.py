from abc import ABC, abstractmethod
from contextlib import contextmanager, nullcontext, suppress
from functools import partial
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
from sklearn.compose import ColumnTransformer
from sklearn.ensemble._hist_gradient_boosting.common import G_H_DTYPE, X_DTYPE, Y_DTYPE
from sklearn.preprocessing import FunctionTransformer, LabelEncoder, OrdinalEncoder
from sklearn.utils._missing import is_scalar_nan
from sklearn.utils._param_validation import Interval, RealNotInt, StrOptions
from sklearn.utils.validation import (
    _check_monotonic_cst,
    _check_sample_weight,
    _check_y,
    _is_pandas_df,
    check_array,
    check_consistent_length,
    check_is_fitted,
    validate_data,
)

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

    def _preprocess_X(self, X, *, reset):
        """Preprocess and validate X.

        Parameters
        ----------
        X : {array-like, pandas DataFrame} of shape (n_samples, n_features)
            Input data.

        reset : bool
            Whether to reset the `n_features_in_` and `feature_names_in_ attributes.

        Returns
        -------
        X : ndarray of shape (n_samples, n_features)
            Validated input data.

        known_categories : list of ndarray of shape (n_categories,)
            List of known categories for each categorical feature.
        """
        check_X_kwargs = dict(dtype=[X_DTYPE], ensure_all_finite=False)
        if not reset:
            if self._preprocessor is None:
                return validate_data(self, X, reset=False, **check_X_kwargs)
            return self._preprocessor.transform(X)
        self.is_categorical_ = self._check_categorical_features(X)
        if self.is_categorical_ is None:
            self._preprocessor = None
            self._is_categorical_remapped = None
            X = validate_data(self, X, **check_X_kwargs)
            return (X, None)
        n_features = X.shape[1]
        ordinal_encoder = OrdinalEncoder(categories='auto', handle_unknown='use_encoded_value', unknown_value=np.nan, encoded_missing_value=np.nan, dtype=X_DTYPE)
        check_X = partial(check_array, **check_X_kwargs)
        numerical_preprocessor = FunctionTransformer(check_X)
        self._preprocessor = ColumnTransformer([('encoder', ordinal_encoder, self.is_categorical_), ('numerical', numerical_preprocessor, ~self.is_categorical_)])
        self._preprocessor.set_output(transform='default')
        X = self._preprocessor.fit_transform(X)
        known_categories = self._check_categories()
        self.n_features_in_ = self._preprocessor.n_features_in_
        with suppress(AttributeError):
            self.feature_names_in_ = self._preprocessor.feature_names_in_
        categorical_remapped = np.zeros(n_features, dtype=bool)
        categorical_remapped[self._preprocessor.output_indices_['encoder']] = True
        self._is_categorical_remapped = categorical_remapped
        return (X, known_categories)

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

    def _check_categorical_features(self, X):
        """Check and validate categorical features in X

        Parameters
        ----------
        X : {array-like, pandas DataFrame} of shape (n_samples, n_features)
            Input data.

        Return
        ------
        is_categorical : ndarray of shape (n_features,) or None, dtype=bool
            Indicates whether a feature is categorical. If no feature is
            categorical, this is None.
        """
        if _is_pandas_df(X):
            X_is_dataframe = True
            categorical_columns_mask = np.asarray(X.dtypes == 'category')
        elif hasattr(X, '__dataframe__'):
            X_is_dataframe = True
            categorical_columns_mask = np.asarray([c.dtype[0].name == 'CATEGORICAL' for c in X.__dataframe__().get_columns()])
        else:
            X_is_dataframe = False
            categorical_columns_mask = None
        categorical_features = self.categorical_features
        categorical_by_dtype = isinstance(categorical_features, str) and categorical_features == 'from_dtype'
        no_categorical_dtype = categorical_features is None or (categorical_by_dtype and (not X_is_dataframe))
        if no_categorical_dtype:
            return None
        use_pandas_categorical = categorical_by_dtype and X_is_dataframe
        if use_pandas_categorical:
            categorical_features = categorical_columns_mask
        else:
            categorical_features = np.asarray(categorical_features)
        if categorical_features.size == 0:
            return None
        if categorical_features.dtype.kind not in ('i', 'b', 'U', 'O'):
            raise ValueError(f'categorical_features must be an array-like of bool, int or str, got: {categorical_features.dtype.name}.')
        if categorical_features.dtype.kind == 'O':
            types = set((type(f) for f in categorical_features))
            if types != {str}:
                raise ValueError(f"categorical_features must be an array-like of bool, int or str, got: {', '.join(sorted((t.__name__ for t in types)))}.")
        n_features = X.shape[1]
        feature_names_in_ = getattr(X, 'columns', None)
        if categorical_features.dtype.kind in ('U', 'O'):
            if feature_names_in_ is None:
                raise ValueError('categorical_features should be passed as an array of integers or as a boolean mask when the model is fitted on data without feature names.')
            is_categorical = np.zeros(n_features, dtype=bool)
            feature_names = list(feature_names_in_)
            for feature_name in categorical_features:
                try:
                    is_categorical[feature_names.index(feature_name)] = True
                except ValueError as e:
                    raise ValueError(f"categorical_features has an item value '{feature_name}' which is not a valid feature name of the training data. Observed feature names: {feature_names}") from e
        elif categorical_features.dtype.kind == 'i':
            if np.max(categorical_features) >= n_features or np.min(categorical_features) < 0:
                raise ValueError('categorical_features set as integer indices must be in [0, n_features - 1]')
            is_categorical = np.zeros(n_features, dtype=bool)
            is_categorical[categorical_features] = True
        else:
            if categorical_features.shape[0] != n_features:
                raise ValueError(f'categorical_features set as a boolean mask must have shape (n_features,), got: {categorical_features.shape}')
            is_categorical = categorical_features
        if not np.any(is_categorical):
            return None
        return is_categorical
