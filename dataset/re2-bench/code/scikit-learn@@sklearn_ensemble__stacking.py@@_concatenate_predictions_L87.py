from abc import ABCMeta, abstractmethod
from numbers import Integral
import numpy as np
import scipy.sparse as sparse
from sklearn.base import (
    ClassifierMixin,
    RegressorMixin,
    TransformerMixin,
    _fit_context,
    clone,
    is_classifier,
    is_regressor,
)
from sklearn.ensemble._base import _BaseHeterogeneousEnsemble, _fit_single_estimator
from sklearn.utils._param_validation import HasMethods, StrOptions

class _BaseStacking(TransformerMixin, _BaseHeterogeneousEnsemble, metaclass=ABCMeta):
    """Base class for stacking method."""
    _parameter_constraints: dict = {'estimators': [list], 'final_estimator': [None, HasMethods('fit')], 'cv': ['cv_object', StrOptions({'prefit'})], 'n_jobs': [None, Integral], 'passthrough': ['boolean'], 'verbose': ['verbose']}

    @abstractmethod
    def __init__(self, estimators, final_estimator=None, *, cv=None, stack_method='auto', n_jobs=None, verbose=0, passthrough=False):
        super().__init__(estimators=estimators)
        self.final_estimator = final_estimator
        self.cv = cv
        self.stack_method = stack_method
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.passthrough = passthrough

    def _concatenate_predictions(self, X, predictions):
        """Concatenate the predictions of each first layer learner and
        possibly the input dataset `X`.

        If `X` is sparse and `self.passthrough` is False, the output of
        `transform` will be dense (the predictions). If `X` is sparse
        and `self.passthrough` is True, the output of `transform` will
        be sparse.

        This helper is in charge of ensuring the predictions are 2D arrays and
        it will drop one of the probability column when using probabilities
        in the binary case. Indeed, the p(y|c=0) = 1 - p(y|c=1)

        When `y` type is `"multilabel-indicator"`` and the method used is
        `predict_proba`, `preds` can be either a `ndarray` of shape
        `(n_samples, n_class)` or for some estimators a list of `ndarray`.
        This function will drop one of the probability column in this situation as well.
        """
        X_meta = []
        for est_idx, preds in enumerate(predictions):
            if isinstance(preds, list):
                for pred in preds:
                    X_meta.append(pred[:, 1:])
            elif preds.ndim == 1:
                X_meta.append(preds.reshape(-1, 1))
            elif self.stack_method_[est_idx] == 'predict_proba' and len(self.classes_) == 2:
                X_meta.append(preds[:, 1:])
            else:
                X_meta.append(preds)
        self._n_feature_outs = [pred.shape[1] for pred in X_meta]
        if self.passthrough:
            X_meta.append(X)
            if sparse.issparse(X):
                return sparse.hstack(X_meta, format=X.format)
        return np.hstack(X_meta)
