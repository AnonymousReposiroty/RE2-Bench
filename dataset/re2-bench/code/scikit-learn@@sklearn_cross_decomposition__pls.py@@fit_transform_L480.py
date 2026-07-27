import warnings
from abc import ABCMeta, abstractmethod
from numbers import Integral, Real
import numpy as np
from scipy.linalg import pinv, svd
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

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        """Fit model to data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of predictors.

        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target vectors, where `n_samples` is the number of samples and
            `n_targets` is the number of response variables.

        Returns
        -------
        self : object
            Fitted model.
        """
        check_consistent_length(X, y)
        X = validate_data(self, X, dtype=np.float64, force_writeable=True, copy=self.copy, ensure_min_samples=2)
        y = check_array(y, input_name='y', dtype=np.float64, force_writeable=True, copy=self.copy, ensure_2d=False)
        if y.ndim == 1:
            self._predict_1d = True
            y = y.reshape(-1, 1)
        else:
            self._predict_1d = False
        n = X.shape[0]
        p = X.shape[1]
        q = y.shape[1]
        n_components = self.n_components
        rank_upper_bound = min(n, p) if self.deflation_mode == 'regression' else min(n, p, q)
        if n_components > rank_upper_bound:
            raise ValueError(f'`n_components` upper bound is {rank_upper_bound}. Got {n_components} instead. Reduce `n_components`.')
        self._norm_y_weights = self.deflation_mode == 'canonical'
        norm_y_weights = self._norm_y_weights
        Xk, yk, self._x_mean, self._y_mean, self._x_std, self._y_std = _center_scale_xy(X, y, self.scale)
        self.x_weights_ = np.zeros((p, n_components))
        self.y_weights_ = np.zeros((q, n_components))
        self._x_scores = np.zeros((n, n_components))
        self._y_scores = np.zeros((n, n_components))
        self.x_loadings_ = np.zeros((p, n_components))
        self.y_loadings_ = np.zeros((q, n_components))
        self.n_iter_ = []
        y_eps = np.finfo(yk.dtype).eps
        for k in range(n_components):
            if self.algorithm == 'nipals':
                yk_mask = np.all(np.abs(yk) < 10 * y_eps, axis=0)
                yk[:, yk_mask] = 0.0
                try:
                    x_weights, y_weights, n_iter_ = _get_first_singular_vectors_power_method(Xk, yk, mode=self.mode, max_iter=self.max_iter, tol=self.tol, norm_y_weights=norm_y_weights)
                except StopIteration as e:
                    if str(e) != 'y residual is constant':
                        raise
                    warnings.warn(f'y residual is constant at iteration {k}')
                    break
                self.n_iter_.append(n_iter_)
            elif self.algorithm == 'svd':
                x_weights, y_weights = _get_first_singular_vectors_svd(Xk, yk)
            _svd_flip_1d(x_weights, y_weights)
            x_scores = np.dot(Xk, x_weights)
            if norm_y_weights:
                y_ss = 1
            else:
                y_ss = np.dot(y_weights, y_weights)
            y_scores = np.dot(yk, y_weights) / y_ss
            x_loadings = np.dot(x_scores, Xk) / np.dot(x_scores, x_scores)
            Xk -= np.outer(x_scores, x_loadings)
            if self.deflation_mode == 'canonical':
                y_loadings = np.dot(y_scores, yk) / np.dot(y_scores, y_scores)
                yk -= np.outer(y_scores, y_loadings)
            if self.deflation_mode == 'regression':
                y_loadings = np.dot(x_scores, yk) / np.dot(x_scores, x_scores)
                yk -= np.outer(x_scores, y_loadings)
            self.x_weights_[:, k] = x_weights
            self.y_weights_[:, k] = y_weights
            self._x_scores[:, k] = x_scores
            self._y_scores[:, k] = y_scores
            self.x_loadings_[:, k] = x_loadings
            self.y_loadings_[:, k] = y_loadings
        self.x_rotations_ = np.dot(self.x_weights_, pinv(np.dot(self.x_loadings_.T, self.x_weights_), check_finite=False))
        self.y_rotations_ = np.dot(self.y_weights_, pinv(np.dot(self.y_loadings_.T, self.y_weights_), check_finite=False))
        self.coef_ = np.dot(self.x_rotations_, self.y_loadings_.T)
        self.coef_ = (self.coef_ * self._y_std).T / self._x_std
        self.intercept_ = self._y_mean
        self._n_features_out = self.x_rotations_.shape[1]
        return self

    def fit_transform(self, X, y=None):
        """Learn and apply the dimension reduction on the train data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of predictors.

        y : array-like of shape (n_samples, n_targets), default=None
            Target vectors, where `n_samples` is the number of samples and
            `n_targets` is the number of response variables.

        Returns
        -------
        self : ndarray of shape (n_samples, n_components)
            Return `x_scores` if `y` is not given, `(x_scores, y_scores)` otherwise.
        """
        return self.fit(X, y).transform(X, y)
