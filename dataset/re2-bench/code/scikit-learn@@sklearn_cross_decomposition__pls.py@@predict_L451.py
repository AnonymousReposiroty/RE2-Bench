from abc import ABCMeta, abstractmethod
from numbers import Integral, Real
from sklearn.base import (
    BaseEstimator,
    ClassNamePrefixFeaturesOutMixin,
    MultiOutputMixin,
    RegressorMixin,
    TransformerMixin,
    _fit_context,
)
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import FLOAT_DTYPES, check_is_fitted, validate_data

class _PLS(ClassNamePrefixFeaturesOutMixin, TransformerMixin, RegressorMixin, MultiOutputMixin, BaseEstimator, metaclass=ABCMeta):
    """Partial Least Squares (PLS)

    This class implements the generic PLS algorithm.

    Main ref: Wegelin, a survey of Partial Least Squares (PLS) methods,
    with emphasis on the two-block case
    https://stat.uw.edu/sites/default/files/files/reports/2000/tr371.pdf
    """
    _parameter_constraints: dict = {'n_components': [Interval(Integral, 1, None, closed='left')], 'scale': ['boolean'], 'deflation_mode': [StrOptions({'regression', 'canonical'})], 'mode': [StrOptions({'A', 'B'})], 'algorithm': [StrOptions({'svd', 'nipals'})], 'max_iter': [Interval(Integral, 1, None, closed='left')], 'tol': [Interval(Real, 0, None, closed='left')], 'copy': ['boolean']}

    @abstractmethod
    def __init__(self, n_components=2, *, scale=True, deflation_mode='regression', mode='A', algorithm='nipals', max_iter=500, tol=1e-06, copy=True):
        self.n_components = n_components
        self.deflation_mode = deflation_mode
        self.mode = mode
        self.scale = scale
        self.algorithm = algorithm
        self.max_iter = max_iter
        self.tol = tol
        self.copy = copy

    def predict(self, X, copy=True):
        """Predict targets of given samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        copy : bool, default=True
            Whether to copy `X` or perform in-place normalization.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,) or (n_samples, n_targets)
            Returns predicted values.

        Notes
        -----
        This call requires the estimation of a matrix of shape
        `(n_features, n_targets)`, which may be an issue in high dimensional
        space.
        """
        check_is_fitted(self)
        X = validate_data(self, X, copy=copy, dtype=FLOAT_DTYPES, reset=False)
        X -= self._x_mean
        y_pred = X @ self.coef_.T + self.intercept_
        return y_pred.ravel() if self._predict_1d else y_pred
