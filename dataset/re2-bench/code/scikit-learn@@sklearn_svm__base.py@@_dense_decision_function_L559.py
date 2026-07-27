from abc import ABCMeta, abstractmethod
from numbers import Integral, Real
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, _fit_context
from sklearn.svm import _libsvm as libsvm  # type: ignore[attr-defined]
from sklearn.utils import (
    check_array,
    check_random_state,
    column_or_1d,
    compute_class_weight,
)
from sklearn.utils._param_validation import Interval, StrOptions

class BaseLibSVM(BaseEstimator, metaclass=ABCMeta):
    """Base class for estimators that use libsvm as backing library.

    This implements support vector machine classification and regression.

    Parameter documentation is in the derived `SVC` class.
    """
    _parameter_constraints: dict = {'kernel': [StrOptions({'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'}), callable], 'degree': [Interval(Integral, 0, None, closed='left')], 'gamma': [StrOptions({'scale', 'auto'}), Interval(Real, 0.0, None, closed='left')], 'coef0': [Interval(Real, None, None, closed='neither')], 'tol': [Interval(Real, 0.0, None, closed='neither')], 'C': [Interval(Real, 0.0, None, closed='right')], 'nu': [Interval(Real, 0.0, 1.0, closed='right')], 'epsilon': [Interval(Real, 0.0, None, closed='left')], 'shrinking': ['boolean'], 'probability': ['boolean'], 'cache_size': [Interval(Real, 0, None, closed='neither')], 'class_weight': [StrOptions({'balanced'}), dict, None], 'verbose': ['verbose'], 'max_iter': [Interval(Integral, -1, None, closed='left')], 'random_state': ['random_state']}
    _sparse_kernels = ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']

    @abstractmethod
    def __init__(self, kernel, degree, gamma, coef0, tol, C, nu, epsilon, shrinking, probability, cache_size, class_weight, verbose, max_iter, random_state):
        if self._impl not in LIBSVM_IMPL:
            raise ValueError('impl should be one of %s, %s was given' % (LIBSVM_IMPL, self._impl))
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tol = tol
        self.C = C
        self.nu = nu
        self.epsilon = epsilon
        self.shrinking = shrinking
        self.probability = probability
        self.cache_size = cache_size
        self.class_weight = class_weight
        self.verbose = verbose
        self.max_iter = max_iter
        self.random_state = random_state

    def _dense_decision_function(self, X):
        X = check_array(X, dtype=np.float64, order='C', accept_large_sparse=False)
        kernel = self.kernel
        if callable(kernel):
            kernel = 'precomputed'
        return libsvm.decision_function(X, self.support_, self.support_vectors_, self._n_support, self._dual_coef_, self._intercept_, self._probA, self._probB, svm_type=LIBSVM_IMPL.index(self._impl), kernel=kernel, degree=self.degree, cache_size=self.cache_size, coef0=self.coef0, gamma=self._gamma)
