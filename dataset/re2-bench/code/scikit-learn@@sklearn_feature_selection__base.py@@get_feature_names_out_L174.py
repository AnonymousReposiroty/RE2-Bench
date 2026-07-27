from abc import ABCMeta, abstractmethod
import numpy as np
from sklearn.base import TransformerMixin
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

    def get_feature_names_out(self, input_features=None):
        """Mask feature names according to selected features.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Input features.

            - If `input_features` is `None`, then `feature_names_in_` is
              used as feature names in. If `feature_names_in_` is not defined,
              then the following input feature names are generated:
              `["x0", "x1", ..., "x(n_features_in_ - 1)"]`.
            - If `input_features` is an array-like, then `input_features` must
              match `feature_names_in_` if `feature_names_in_` is defined.

        Returns
        -------
        feature_names_out : ndarray of str objects
            Transformed feature names.
        """
        check_is_fitted(self)
        input_features = _check_feature_names_in(self, input_features)
        return input_features[self.get_support()]
