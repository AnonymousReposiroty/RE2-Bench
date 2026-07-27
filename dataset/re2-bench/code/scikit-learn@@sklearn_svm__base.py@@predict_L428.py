from abc import ABCMeta, abstractmethod
from numbers import Integral, Real
import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, ClassifierMixin, _fit_context
from sklearn.exceptions import ConvergenceWarning, NotFittedError
from sklearn.svm import _libsvm as libsvm  # type: ignore[attr-defined]
from sklearn.svm import _libsvm_sparse as libsvm_sparse  # type: ignore[attr-defined]
from sklearn.utils import (
    check_array,
    check_random_state,
    column_or_1d,
    compute_class_weight,
)
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import (
    _check_large_sparse,
    _check_sample_weight,
    _num_samples,
    check_consistent_length,
    check_is_fitted,
    validate_data,
)

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

    def predict(self, X):
        """Perform regression on samples in X.

        For a one-class model, +1 (inlier) or -1 (outlier) is returned.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            For kernel="precomputed", the expected shape of X is
            (n_samples_test, n_samples_train).

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            The predicted values.
        """
        X = self._validate_for_predict(X)
        predict = self._sparse_predict if self._sparse else self._dense_predict
        return predict(X)

    def _dense_predict(self, X):
        X = self._compute_kernel(X)
        if X.ndim == 1:
            X = check_array(X, order='C', accept_large_sparse=False)
        kernel = self.kernel
        if callable(self.kernel):
            kernel = 'precomputed'
            if X.shape[1] != self.shape_fit_[0]:
                raise ValueError('X.shape[1] = %d should be equal to %d, the number of samples at training time' % (X.shape[1], self.shape_fit_[0]))
        svm_type = LIBSVM_IMPL.index(self._impl)
        return libsvm.predict(X, self.support_, self.support_vectors_, self._n_support, self._dual_coef_, self._intercept_, self._probA, self._probB, svm_type=svm_type, kernel=kernel, degree=self.degree, coef0=self.coef0, gamma=self._gamma, cache_size=self.cache_size)

    def _sparse_predict(self, X):
        kernel = self.kernel
        if callable(kernel):
            kernel = 'precomputed'
        kernel_type = self._sparse_kernels.index(kernel)
        C = 0.0
        return libsvm_sparse.libsvm_sparse_predict(X.data, X.indices, X.indptr, self.support_vectors_.data, self.support_vectors_.indices, self.support_vectors_.indptr, self._dual_coef_.data, self._intercept_, LIBSVM_IMPL.index(self._impl), kernel_type, self.degree, self._gamma, self.coef0, self.tol, C, getattr(self, 'class_weight_', np.empty(0)), self.nu, self.epsilon, self.shrinking, self.probability, self._n_support, self._probA, self._probB)

    def _compute_kernel(self, X):
        """Return the data transformed by a callable kernel"""
        if callable(self.kernel):
            kernel = self.kernel(X, self.__Xfit)
            if sp.issparse(kernel):
                kernel = kernel.toarray()
            X = np.asarray(kernel, dtype=np.float64, order='C')
        return X

    def _validate_for_predict(self, X):
        check_is_fitted(self)
        if not callable(self.kernel):
            X = validate_data(self, X, accept_sparse='csr', dtype=np.float64, order='C', accept_large_sparse=False, reset=False)
        if self._sparse and (not sp.issparse(X)):
            X = sp.csr_matrix(X)
        if self._sparse:
            X.sort_indices()
        if sp.issparse(X) and (not self._sparse) and (not callable(self.kernel)):
            raise ValueError('cannot use sparse input in %r trained on dense data' % type(self).__name__)
        if self.kernel == 'precomputed':
            if X.shape[1] != self.shape_fit_[0]:
                raise ValueError('X.shape[1] = %d should be equal to %d, the number of samples at training time' % (X.shape[1], self.shape_fit_[0]))
        sv = self.support_vectors_
        if not self._sparse and sv.size > 0 and (self.n_support_.sum() != sv.shape[0]):
            raise ValueError(f'The internal representation of {self.__class__.__name__} was altered')
        return X

    @property
    def n_support_(self):
        """Number of support vectors for each class."""
        try:
            check_is_fitted(self)
        except NotFittedError:
            raise AttributeError
        svm_type = LIBSVM_IMPL.index(self._impl)
        if svm_type in (0, 1):
            return self._n_support
        else:
            return np.array([self._n_support[0]])
