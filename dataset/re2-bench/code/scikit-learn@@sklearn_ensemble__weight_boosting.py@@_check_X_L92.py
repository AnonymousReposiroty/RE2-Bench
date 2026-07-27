from abc import ABCMeta, abstractmethod
from numbers import Integral, Real
from sklearn.ensemble._base import BaseEnsemble
from sklearn.utils._param_validation import HasMethods, Interval, StrOptions
from sklearn.utils.validation import (
    _check_sample_weight,
    _num_samples,
    check_is_fitted,
    has_fit_parameter,
    validate_data,
)

class BaseWeightBoosting(BaseEnsemble, metaclass=ABCMeta):
    """Base class for AdaBoost estimators.

    Warning: This class should not be used directly. Use derived classes
    instead.
    """
    _parameter_constraints: dict = {'estimator': [HasMethods(['fit', 'predict']), None], 'n_estimators': [Interval(Integral, 1, None, closed='left')], 'learning_rate': [Interval(Real, 0, None, closed='neither')], 'random_state': ['random_state']}

    @abstractmethod
    def __init__(self, estimator=None, *, n_estimators=50, estimator_params=tuple(), learning_rate=1.0, random_state=None):
        super().__init__(estimator=estimator, n_estimators=n_estimators, estimator_params=estimator_params)
        self.learning_rate = learning_rate
        self.random_state = random_state

    def _check_X(self, X):
        return validate_data(self, X, accept_sparse=['csr', 'csc'], ensure_2d=True, allow_nd=True, dtype=None, reset=False)
