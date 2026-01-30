from abc import ABCMeta, abstractmethod
from sklearn.base import BaseEstimator, ClassifierMixin, _fit_context
from sklearn.utils._array_api import (
    _average,
    _convert_to_numpy,
    _find_matching_floating_dtype,
    _isin,
    _logsumexp,
    get_namespace,
    get_namespace_and_device,
    size,
)
from sklearn.utils.validation import (
    _check_n_features,
    _check_sample_weight,
    check_is_fitted,
    check_non_negative,
    validate_data,
)

class _BaseNB(ClassifierMixin, BaseEstimator, metaclass=ABCMeta):
    """Abstract base class for naive Bayes estimators"""

    @abstractmethod
    def _joint_log_likelihood(self, X):
        """Compute the unnormalized posterior log probability of X

        I.e. ``log P(c) + log P(x|c)`` for all rows x of X, as an array-like of
        shape (n_samples, n_classes).

        Public methods predict, predict_proba, predict_log_proba, and
        predict_joint_log_proba pass the input through _check_X before handing it
        over to _joint_log_likelihood. The term "joint log likelihood" is used
        interchangibly with "joint log probability".
        """

    @abstractmethod
    def _check_X(self, X):
        """To be overridden in subclasses with the actual checks.

        Only used in predict* methods.
        """

    def predict(self, X):
        """
        Perform classification on an array of test vectors X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        C : ndarray of shape (n_samples,)
            Predicted target values for X.
        """
        check_is_fitted(self)
        xp, _ = get_namespace(X)
        X = self._check_X(X)
        jll = self._joint_log_likelihood(X)
        pred_indices = xp.argmax(jll, axis=1)
        if isinstance(self.classes_[0], str):
            pred_indices = _convert_to_numpy(pred_indices, xp=xp)
        return self.classes_[pred_indices]
