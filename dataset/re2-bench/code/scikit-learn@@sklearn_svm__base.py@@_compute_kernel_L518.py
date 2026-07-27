from abc import ABCMeta, abstractmethod
from numbers import Integral, Real
import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, ClassifierMixin, _fit_context
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

    def _compute_kernel(self, X):
        """Return the data transformed by a callable kernel"""
        if callable(self.kernel):
            kernel = self.kernel(X, self.__Xfit)
            if sp.issparse(kernel):
                kernel = kernel.toarray()
            X = np.asarray(kernel, dtype=np.float64, order='C')
        return X
