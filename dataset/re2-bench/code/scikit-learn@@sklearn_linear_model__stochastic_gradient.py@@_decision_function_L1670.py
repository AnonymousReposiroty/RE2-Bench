from abc import ABCMeta, abstractmethod
from numbers import Integral, Real
from sklearn._loss._loss import CyHalfBinomialLoss, CyHalfSquaredError, CyHuberLoss
from sklearn.base import (
    BaseEstimator,
    OutlierMixin,
    RegressorMixin,
    _fit_context,
    clone,
    is_classifier,
)
from sklearn.linear_model._sgd_fast import (
    EpsilonInsensitive,
    Hinge,
    ModifiedHuber,
    SquaredEpsilonInsensitive,
    SquaredHinge,
    _plain_sgd32,
    _plain_sgd64,
)
from sklearn.utils._param_validation import Hidden, Interval, StrOptions
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.validation import (
    _check_sample_weight,
    check_is_fitted,
    validate_data,
)

class BaseSGDRegressor(RegressorMixin, BaseSGD):
    loss_functions = {'squared_error': (CyHalfSquaredError,), 'huber': (CyHuberLoss, DEFAULT_EPSILON), 'epsilon_insensitive': (EpsilonInsensitive, DEFAULT_EPSILON), 'squared_epsilon_insensitive': (SquaredEpsilonInsensitive, DEFAULT_EPSILON)}
    _parameter_constraints: dict = {**BaseSGD._parameter_constraints, 'loss': [StrOptions(set(loss_functions))], 'early_stopping': ['boolean'], 'validation_fraction': [Interval(Real, 0, 1, closed='neither')], 'n_iter_no_change': [Interval(Integral, 1, None, closed='left')]}

    @abstractmethod
    def __init__(self, loss='squared_error', *, penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, max_iter=1000, tol=0.001, shuffle=True, verbose=0, epsilon=DEFAULT_EPSILON, random_state=None, learning_rate='invscaling', eta0=0.01, power_t=0.25, early_stopping=False, validation_fraction=0.1, n_iter_no_change=5, warm_start=False, average=False):
        super().__init__(loss=loss, penalty=penalty, alpha=alpha, l1_ratio=l1_ratio, fit_intercept=fit_intercept, max_iter=max_iter, tol=tol, shuffle=shuffle, verbose=verbose, epsilon=epsilon, random_state=random_state, learning_rate=learning_rate, eta0=eta0, power_t=power_t, early_stopping=early_stopping, validation_fraction=validation_fraction, n_iter_no_change=n_iter_no_change, warm_start=warm_start, average=average)

    def _decision_function(self, X):
        """Predict using the linear model

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples,)
           Predicted target values per element in X.
        """
        check_is_fitted(self)
        X = validate_data(self, X, accept_sparse='csr', reset=False)
        scores = safe_sparse_dot(X, self.coef_.T, dense_output=True) + self.intercept_
        return scores.ravel()
