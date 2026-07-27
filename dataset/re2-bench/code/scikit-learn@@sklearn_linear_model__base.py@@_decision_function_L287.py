from abc import ABCMeta, abstractmethod
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    MultiOutputMixin,
    RegressorMixin,
    _fit_context,
)
from sklearn.utils.validation import (
    _check_sample_weight,
    check_is_fitted,
    validate_data,
)

class LinearModel(BaseEstimator, metaclass=ABCMeta):
    """Base class for Linear Models"""

    def _decision_function(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, accept_sparse=['csr', 'csc', 'coo'], reset=False)
        coef_ = self.coef_
        if coef_.ndim == 1:
            return X @ coef_ + self.intercept_
        else:
            return X @ coef_.T + self.intercept_
