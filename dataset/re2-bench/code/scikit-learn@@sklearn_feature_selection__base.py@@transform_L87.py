import warnings
from abc import ABCMeta, abstractmethod
import numpy as np
from sklearn.base import TransformerMixin
from sklearn.utils import _safe_indexing, check_array, safe_sqr
from sklearn.utils._set_output import _get_output_config
from sklearn.utils._tags import get_tags
from sklearn.utils.validation import (
    _check_feature_names_in,
    _is_pandas_df,
    check_is_fitted,
    validate_data,
)

class SelectorMixin(TransformerMixin, metaclass=ABCMeta):
    """
    Transformer mixin that performs feature selection given a support mask

    This mixin provides a feature selector implementation with `transform` and
    `inverse_transform` functionality given an implementation of
    `_get_support_mask`.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.base import BaseEstimator
    >>> from sklearn.feature_selection import SelectorMixin
    >>> class FeatureSelector(SelectorMixin, BaseEstimator):
    ...    def fit(self, X, y=None):
    ...        self.n_features_in_ = X.shape[1]
    ...        return self
    ...    def _get_support_mask(self):
    ...        mask = np.zeros(self.n_features_in_, dtype=bool)
    ...        mask[:2] = True  # select the first two features
    ...        return mask
    >>> X, y = load_iris(return_X_y=True)
    >>> FeatureSelector().fit_transform(X, y).shape
    (150, 2)
    """

    def get_support(self, indices=False):
        """
        Get a mask, or integer index, of the features selected.

        Parameters
        ----------
        indices : bool, default=False
            If True, the return value will be an array of integers, rather
            than a boolean mask.

        Returns
        -------
        support : array
            An index that selects the retained features from a feature vector.
            If `indices` is False, this is a boolean array of shape
            [# input features], in which an element is True iff its
            corresponding feature is selected for retention. If `indices` is
            True, this is an integer array of shape [# output features] whose
            values are indices into the input feature vector.
        """
        mask = self._get_support_mask()
        return mask if not indices else np.nonzero(mask)[0]

    @abstractmethod
    def _get_support_mask(self):
        """
        Get the boolean mask indicating which features are selected

        Returns
        -------
        support : boolean array of shape [# input features]
            An element is True iff its corresponding feature is selected for
            retention.
        """

    def transform(self, X):
        """Reduce X to the selected features.

        Parameters
        ----------
        X : array of shape [n_samples, n_features]
            The input samples.

        Returns
        -------
        X_r : array of shape [n_samples, n_selected_features]
            The input samples with only the selected features.
        """
        output_config_dense = _get_output_config('transform', estimator=self)['dense']
        preserve_X = output_config_dense != 'default' and _is_pandas_df(X)
        X = validate_data(self, X, dtype=None, accept_sparse='csr', ensure_all_finite=not get_tags(self).input_tags.allow_nan, skip_check_array=preserve_X, reset=False)
        return self._transform(X)

    def _transform(self, X):
        """Reduce X to the selected features."""
        mask = self.get_support()
        if not mask.any():
            warnings.warn('No features were selected: either the data is too noisy or the selection test too strict.', UserWarning)
            if hasattr(X, 'iloc'):
                return X.iloc[:, :0]
            return np.empty(0, dtype=X.dtype).reshape((X.shape[0], 0))
        return _safe_indexing(X, mask, axis=1)
