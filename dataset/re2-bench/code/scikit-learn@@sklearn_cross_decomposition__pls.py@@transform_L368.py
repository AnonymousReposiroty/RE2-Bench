from abc import ABCMeta, abstractmethod
from numbers import Integral, Real
import numpy as np
from sklearn.base import (
    BaseEstimator,
    ClassNamePrefixFeaturesOutMixin,
    MultiOutputMixin,
    RegressorMixin,
    TransformerMixin,
    _fit_context,
)
from sklearn.utils import check_array, check_consistent_length
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

    def transform(self, X, y=None, copy=True):
        """Apply the dimension reduction.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to transform.

        y : array-like of shape (n_samples, n_targets), default=None
            Target vectors.

        copy : bool, default=True
            Whether to copy `X` and `y`, or perform in-place normalization.

        Returns
        -------
        x_scores, y_scores : array-like or tuple of array-like
            Return `x_scores` if `y` is not given, `(x_scores, y_scores)` otherwise.
        """
        check_is_fitted(self)
        X = validate_data(self, X, copy=copy, dtype=FLOAT_DTYPES, reset=False)
        X -= self._x_mean
        X /= self._x_std
        x_scores = np.dot(X, self.x_rotations_)
        if y is not None:
            y = check_array(y, input_name='y', ensure_2d=False, copy=copy, dtype=FLOAT_DTYPES)
            if y.ndim == 1:
                y = y.reshape(-1, 1)
            y -= self._y_mean
            y /= self._y_std
            y_scores = np.dot(y, self.y_rotations_)
            return (x_scores, y_scores)
        return x_scores
