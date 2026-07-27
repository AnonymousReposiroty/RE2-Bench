import numpy as np
from scipy.special import expit
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    MultiOutputMixin,
    RegressorMixin,
    _fit_context,
)
from sklearn.utils._array_api import (
    _asarray_with_order,
    _average,
    get_namespace,
    get_namespace_and_device,
    indexing_dtype,
    supported_float_dtypes,
)
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.validation import (
    _check_sample_weight,
    check_is_fitted,
    validate_data,
)

class LinearClassifierMixin(ClassifierMixin):
    """Mixin for linear classifiers.

    Handles prediction for sparse and dense X.
    """

    def decision_function(self, X):
        """
        Predict confidence scores for samples.

        The confidence score for a sample is proportional to the signed
        distance of that sample to the hyperplane.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data matrix for which we want to get the confidence scores.

        Returns
        -------
        scores : ndarray of shape (n_samples,) or (n_samples, n_classes)
            Confidence scores per `(n_samples, n_classes)` combination. In the
            binary case, confidence score for `self.classes_[1]` where >0 means
            this class would be predicted.
        """
        check_is_fitted(self)
        xp, _ = get_namespace(X)
        X = validate_data(self, X, accept_sparse='csr', reset=False)
        coef_T = self.coef_.T if self.coef_.ndim == 2 else self.coef_
        scores = safe_sparse_dot(X, coef_T, dense_output=True) + self.intercept_
        return xp.reshape(scores, (-1,)) if scores.ndim > 1 and scores.shape[1] == 1 else scores

    def _predict_proba_lr(self, X):
        """Probability estimation for OvR logistic regression.

        Positive class probabilities are computed as
        1. / (1. + np.exp(-self.decision_function(X)));
        multiclass is handled by normalizing that over all classes.
        """
        prob = self.decision_function(X)
        expit(prob, out=prob)
        if prob.ndim == 1:
            return np.vstack([1 - prob, prob]).T
        else:
            prob /= prob.sum(axis=1).reshape((prob.shape[0], -1))
            return prob
