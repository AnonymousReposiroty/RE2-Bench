from abc import ABCMeta, abstractmethod
import numpy as np
from sklearn.base import (
    BaseEstimator,
    ClassNamePrefixFeaturesOutMixin,
    TransformerMixin,
)
from sklearn.utils._array_api import _add_to_diagonal, device, get_namespace

class _BasePCA(ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator, metaclass=ABCMeta):
    """Base class for PCA methods.

    Warning: This class should not be used directly.
    Use derived classes instead.
    """

    def get_covariance(self):
        """Compute data covariance with the generative model.

        ``cov = components_.T * S**2 * components_ + sigma2 * eye(n_features)``
        where S**2 contains the explained variances, and sigma2 contains the
        noise variances.

        Returns
        -------
        cov : array of shape=(n_features, n_features)
            Estimated covariance of data.
        """
        xp, _ = get_namespace(self.components_)
        components_ = self.components_
        exp_var = self.explained_variance_
        if self.whiten:
            components_ = components_ * xp.sqrt(exp_var[:, np.newaxis])
        exp_var_diff = exp_var - self.noise_variance_
        exp_var_diff = xp.where(exp_var > self.noise_variance_, exp_var_diff, xp.asarray(0.0, device=device(exp_var), dtype=exp_var.dtype))
        cov = components_.T * exp_var_diff @ components_
        _add_to_diagonal(cov, self.noise_variance_, xp)
        return cov
