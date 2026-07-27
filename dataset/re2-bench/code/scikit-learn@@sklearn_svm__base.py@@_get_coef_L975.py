from abc import ABCMeta, abstractmethod
import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, ClassifierMixin, _fit_context
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.extmath import safe_sparse_dot

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

    def _get_coef(self):
        if self.dual_coef_.shape[0] == 1:
            coef = safe_sparse_dot(self.dual_coef_, self.support_vectors_)
        else:
            coef = _one_vs_one_coef(self.dual_coef_, self._n_support, self.support_vectors_)
            if sp.issparse(coef[0]):
                coef = sp.vstack(coef).tocsr()
            else:
                coef = np.vstack(coef)
        return coef
