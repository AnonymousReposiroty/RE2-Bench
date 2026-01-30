from abc import ABCMeta, abstractmethod
from numbers import Integral
import numpy as np
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
from sklearn.utils.validation import (
    _check_feature_names_in,
    _check_response_method,
    _estimator_has,
    check_is_fitted,
    column_or_1d,
)

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

    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Input features. The input feature names are only used when `passthrough` is
            `True`.

            - If `input_features` is `None`, then `feature_names_in_` is
              used as feature names in. If `feature_names_in_` is not defined,
              then names are generated: `[x0, x1, ..., x(n_features_in_ - 1)]`.
            - If `input_features` is an array-like, then `input_features` must
              match `feature_names_in_` if `feature_names_in_` is defined.

            If `passthrough` is `False`, then only the names of `estimators` are used
            to generate the output feature names.

        Returns
        -------
        feature_names_out : ndarray of str objects
            Transformed feature names.
        """
        check_is_fitted(self, 'n_features_in_')
        input_features = _check_feature_names_in(self, input_features, generate_names=self.passthrough)
        class_name = self.__class__.__name__.lower()
        non_dropped_estimators = (name for name, est in self.estimators if est != 'drop')
        meta_names = []
        for est, n_features_out in zip(non_dropped_estimators, self._n_feature_outs):
            if n_features_out == 1:
                meta_names.append(f'{class_name}_{est}')
            else:
                meta_names.extend((f'{class_name}_{est}{i}' for i in range(n_features_out)))
        if self.passthrough:
            return np.concatenate((meta_names, input_features))
        return np.asarray(meta_names, dtype=object)
