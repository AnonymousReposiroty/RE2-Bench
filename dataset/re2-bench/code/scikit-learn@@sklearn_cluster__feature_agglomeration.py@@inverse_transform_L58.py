import numpy as np
from sklearn.base import TransformerMixin
from sklearn.utils.validation import check_is_fitted, validate_data

class AgglomerationTransform(TransformerMixin):
    """
    A class for feature agglomeration via the transform interface.
    """

    def inverse_transform(self, X):
        """
        Inverse the transformation and return a vector of size `n_features`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_clusters) or (n_clusters,)
            The values to be assigned to each cluster of samples.

        Returns
        -------
        X_original : ndarray of shape (n_samples, n_features) or (n_features,)
            A vector of size `n_samples` with the values of `X` assigned to
            each of the cluster of samples.
        """
        check_is_fitted(self)
        unil, inverse = np.unique(self.labels_, return_inverse=True)
        return X[..., inverse]
