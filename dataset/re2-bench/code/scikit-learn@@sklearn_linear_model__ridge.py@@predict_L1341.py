from sklearn.linear_model._base import (
    LinearClassifierMixin,
    LinearModel,
    _preprocess_data,
    _rescale_data,
)
from sklearn.utils._array_api import (
    _is_numpy_namespace,
    _max_precision_float_dtype,
    _ravel,
    device,
    get_namespace,
    get_namespace_and_device,
    move_to,
)
from sklearn.utils.extmath import row_norms, safe_sparse_dot
from sklearn.utils.validation import (
    _check_sample_weight,
    check_is_fitted,
    validate_data,
)

class _RidgeClassifierMixin(LinearClassifierMixin):

    def predict(self, X):
        """Predict class labels for samples in `X`.

        Parameters
        ----------
        X : {array-like, spare matrix} of shape (n_samples, n_features)
            The data matrix for which we want to predict the targets.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,) or (n_samples, n_outputs)
            Vector or matrix containing the predictions. In binary and
            multiclass problems, this is a vector containing `n_samples`. In
            a multilabel problem, it returns a matrix of shape
            `(n_samples, n_outputs)`.
        """
        check_is_fitted(self, attributes=['_label_binarizer'])
        if self._label_binarizer.y_type_.startswith('multilabel'):
            decision = self.decision_function(X)
            xp, _ = get_namespace(decision)
            scores = 2.0 * xp.astype(decision > 0, decision.dtype) - 1.0
            return self._label_binarizer.inverse_transform(scores)
        return super().predict(X)

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
