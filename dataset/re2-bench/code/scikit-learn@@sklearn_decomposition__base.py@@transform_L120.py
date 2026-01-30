from abc import ABCMeta, abstractmethod
from sklearn.base import (
    BaseEstimator,
    ClassNamePrefixFeaturesOutMixin,
    TransformerMixin,
)
from sklearn.utils._array_api import _add_to_diagonal, device, get_namespace
from sklearn.utils.validation import check_array, check_is_fitted, validate_data

class _BasePCA(ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator, metaclass=ABCMeta):
    """Base class for PCA methods.

    Warning: This class should not be used directly.
    Use derived classes instead.
    """

    def transform(self, X):
        """Apply dimensionality reduction to X.

        X is projected on the first principal components previously extracted
        from a training set.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        Returns
        -------
        X_new : array-like of shape (n_samples, n_components)
            Projection of X in the first principal components, where `n_samples`
            is the number of samples and `n_components` is the number of the components.
        """
        xp, _ = get_namespace(X, self.components_, self.explained_variance_)
        check_is_fitted(self)
        X = validate_data(self, X, dtype=[xp.float64, xp.float32], accept_sparse=('csr', 'csc'), reset=False)
        return self._transform(X, xp=xp, x_is_centered=False)

    def _transform(self, X, xp, x_is_centered=False):
        X_transformed = X @ self.components_.T
        if not x_is_centered:
            X_transformed -= xp.reshape(self.mean_, (1, -1)) @ self.components_.T
        if self.whiten:
            scale = xp.sqrt(self.explained_variance_)
            min_scale = xp.finfo(scale.dtype).eps
            scale[scale < min_scale] = min_scale
            X_transformed /= scale
        return X_transformed
