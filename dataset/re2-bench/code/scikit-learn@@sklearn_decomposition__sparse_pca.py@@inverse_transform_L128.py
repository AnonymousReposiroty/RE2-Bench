from numbers import Integral, Real
from sklearn.base import (
    BaseEstimator,
    ClassNamePrefixFeaturesOutMixin,
    TransformerMixin,
    _fit_context,
)
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import check_array, check_is_fitted, validate_data

class _BaseSparsePCA(ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator):
    """Base class for SparsePCA and MiniBatchSparsePCA"""
    _parameter_constraints: dict = {'n_components': [None, Interval(Integral, 1, None, closed='left')], 'alpha': [Interval(Real, 0.0, None, closed='left')], 'ridge_alpha': [Interval(Real, 0.0, None, closed='left')], 'max_iter': [Interval(Integral, 0, None, closed='left')], 'tol': [Interval(Real, 0.0, None, closed='left')], 'method': [StrOptions({'lars', 'cd'})], 'n_jobs': [Integral, None], 'verbose': ['verbose'], 'random_state': ['random_state']}

    def __init__(self, n_components=None, *, alpha=1, ridge_alpha=0.01, max_iter=1000, tol=1e-08, method='lars', n_jobs=None, verbose=False, random_state=None):
        self.n_components = n_components
        self.alpha = alpha
        self.ridge_alpha = ridge_alpha
        self.max_iter = max_iter
        self.tol = tol
        self.method = method
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state

    def inverse_transform(self, X):
        """Transform data from the latent space to the original space.

        This inversion is an approximation due to the loss of information
        induced by the forward decomposition.

        .. versionadded:: 1.2

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_components)
            Data in the latent space.

        Returns
        -------
        X_original : ndarray of shape (n_samples, n_features)
            Reconstructed data in the original space.
        """
        check_is_fitted(self)
        X = check_array(X)
        return X @ self.components_ + self.mean_
