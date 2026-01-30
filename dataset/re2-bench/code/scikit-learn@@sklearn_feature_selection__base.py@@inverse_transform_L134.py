from abc import ABCMeta, abstractmethod
import numpy as np
from scipy.sparse import csc_matrix, issparse
from sklearn.base import TransformerMixin
from sklearn.utils import _safe_indexing, check_array, safe_sqr

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

    def inverse_transform(self, X):
        """Reverse the transformation operation.

        Parameters
        ----------
        X : array of shape [n_samples, n_selected_features]
            The input samples.

        Returns
        -------
        X_original : array of shape [n_samples, n_original_features]
            `X` with columns of zeros inserted where features would have
            been removed by :meth:`transform`.
        """
        if issparse(X):
            X = X.tocsc()
            it = self.inverse_transform(np.diff(X.indptr).reshape(1, -1))
            col_nonzeros = it.ravel()
            indptr = np.concatenate([[0], np.cumsum(col_nonzeros)])
            Xt = csc_matrix((X.data, X.indices, indptr), shape=(X.shape[0], len(indptr) - 1), dtype=X.dtype)
            return Xt
        support = self.get_support()
        X = check_array(X, dtype=None)
        if support.sum() != X.shape[1]:
            raise ValueError('X has a different shape than during fitting.')
        if X.ndim == 1:
            X = X[None, :]
        Xt = np.zeros((X.shape[0], support.size), dtype=X.dtype)
        Xt[:, support] = X
        return Xt
