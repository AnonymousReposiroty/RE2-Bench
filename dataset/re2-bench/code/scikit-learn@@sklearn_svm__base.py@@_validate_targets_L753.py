from abc import ABCMeta, abstractmethod
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, _fit_context
from sklearn.utils import (
    check_array,
    check_random_state,
    column_or_1d,
    compute_class_weight,
)
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.multiclass import (
    _ovr_decision_function,
    check_classification_targets,
)

class BaseSVC(ClassifierMixin, BaseLibSVM, metaclass=ABCMeta):
    """ABC for LibSVM-based classifiers."""
    _parameter_constraints: dict = {**BaseLibSVM._parameter_constraints, 'decision_function_shape': [StrOptions({'ovr', 'ovo'})], 'break_ties': ['boolean']}
    for unused_param in ['epsilon', 'nu']:
        _parameter_constraints.pop(unused_param)

    @abstractmethod
    def __init__(self, kernel, degree, gamma, coef0, tol, C, nu, shrinking, probability, cache_size, class_weight, verbose, max_iter, decision_function_shape, random_state, break_ties):
        self.decision_function_shape = decision_function_shape
        self.break_ties = break_ties
        super().__init__(kernel=kernel, degree=degree, gamma=gamma, coef0=coef0, tol=tol, C=C, nu=nu, epsilon=0.0, shrinking=shrinking, probability=probability, cache_size=cache_size, class_weight=class_weight, verbose=verbose, max_iter=max_iter, random_state=random_state)

    def _validate_targets(self, y):
        y_ = column_or_1d(y, warn=True)
        check_classification_targets(y)
        cls, y = np.unique(y_, return_inverse=True)
        self.class_weight_ = compute_class_weight(self.class_weight, classes=cls, y=y_)
        if len(cls) < 2:
            raise ValueError('The number of classes has to be greater than one; got %d class' % len(cls))
        self.classes_ = cls
        return np.asarray(y, dtype=np.float64, order='C')
