from numbers import Integral, Real
from sklearn.base import (
    BaseEstimator,
    ClassNamePrefixFeaturesOutMixin,
    TransformerMixin,
    _fit_context,
)
from sklearn.linear_model import ridge_regression
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

    def transform(self, X):
        """Least Squares projection of the data onto the sparse components.

        To avoid instability issues in case the system is under-determined,
        regularization can be applied (Ridge regression) via the
        `ridge_alpha` parameter.

        Note that Sparse PCA components orthogonality is not enforced as in PCA
        hence one cannot use a simple linear projection.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test data to be transformed, must have the same number of
            features as the data used to train the model.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            Transformed data.
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        X = X - self.mean_
        U = ridge_regression(self.components_.T, X.T, self.ridge_alpha, solver='cholesky')
        return U
