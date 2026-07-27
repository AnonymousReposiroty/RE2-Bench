from abc import ABCMeta, abstractmethod
from sklearn.base import BaseEstimator, ClassifierMixin, _fit_context
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

    def decision_function(self, X):
        """Evaluate the decision function for the samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        X : ndarray of shape (n_samples, n_classes * (n_classes-1) / 2)
            Returns the decision function of the sample for each class
            in the model.
            If decision_function_shape='ovr', the shape is (n_samples,
            n_classes).

        Notes
        -----
        If decision_function_shape='ovo', the function values are proportional
        to the distance of the samples X to the separating hyperplane. If the
        exact distances are required, divide the function values by the norm of
        the weight vector (``coef_``). See also `this question
        <https://stats.stackexchange.com/questions/14876/
        interpreting-distance-from-hyperplane-in-svm>`_ for further details.
        If decision_function_shape='ovr', the decision function is a monotonic
        transformation of ovo decision function.
        """
        dec = self._decision_function(X)
        if self.decision_function_shape == 'ovr' and len(self.classes_) > 2:
            return _ovr_decision_function(dec < 0, -dec, len(self.classes_))
        return dec

    def _decision_function(self, X):
        """Evaluates the decision function for the samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        X : array-like of shape (n_samples, n_class * (n_class-1) / 2)
            Returns the decision function of the sample for each class
            in the model.
        """
        X = self._validate_for_predict(X)
        X = self._compute_kernel(X)
        if self._sparse:
            dec_func = self._sparse_decision_function(X)
        else:
            dec_func = self._dense_decision_function(X)
        if self._impl in ['c_svc', 'nu_svc'] and len(self.classes_) == 2:
            return -dec_func.ravel()
        return dec_func
