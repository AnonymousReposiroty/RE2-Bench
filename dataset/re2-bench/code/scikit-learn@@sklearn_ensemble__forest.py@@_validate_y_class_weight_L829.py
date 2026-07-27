from abc import ABCMeta, abstractmethod
from warnings import catch_warnings, simplefilter, warn
import numpy as np
from sklearn.base import (
    ClassifierMixin,
    MultiOutputMixin,
    RegressorMixin,
    TransformerMixin,
    _fit_context,
    is_classifier,
)
from sklearn.utils import check_random_state, compute_sample_weight
from sklearn.utils.multiclass import check_classification_targets, type_of_target

class ForestClassifier(ClassifierMixin, BaseForest, metaclass=ABCMeta):
    """
    Base class for forest of trees-based classifiers.

    Warning: This class should not be used directly. Use derived classes
    instead.
    """

    @abstractmethod
    def __init__(self, estimator, n_estimators=100, *, estimator_params=tuple(), bootstrap=False, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, class_weight=None, max_samples=None):
        super().__init__(estimator=estimator, n_estimators=n_estimators, estimator_params=estimator_params, bootstrap=bootstrap, oob_score=oob_score, n_jobs=n_jobs, random_state=random_state, verbose=verbose, warm_start=warm_start, class_weight=class_weight, max_samples=max_samples)

    def _validate_y_class_weight(self, y):
        check_classification_targets(y)
        y = np.copy(y)
        expanded_class_weight = None
        if self.class_weight is not None:
            y_original = np.copy(y)
        self.classes_ = []
        self.n_classes_ = []
        y_store_unique_indices = np.zeros(y.shape, dtype=int)
        for k in range(self.n_outputs_):
            classes_k, y_store_unique_indices[:, k] = np.unique(y[:, k], return_inverse=True)
            self.classes_.append(classes_k)
            self.n_classes_.append(classes_k.shape[0])
        y = y_store_unique_indices
        if self.class_weight is not None:
            valid_presets = ('balanced', 'balanced_subsample')
            if isinstance(self.class_weight, str):
                if self.class_weight not in valid_presets:
                    raise ValueError('Valid presets for class_weight include "balanced" and "balanced_subsample".Given "%s".' % self.class_weight)
                if self.warm_start:
                    warn('class_weight presets "balanced" or "balanced_subsample" are not recommended for warm_start if the fitted data differs from the full dataset. In order to use "balanced" weights, use compute_class_weight ("balanced", classes, y). In place of y you can use a large enough sample of the full training set target to properly estimate the class frequency distributions. Pass the resulting weights as the class_weight parameter.')
            if self.class_weight != 'balanced_subsample' or not self.bootstrap:
                if self.class_weight == 'balanced_subsample':
                    class_weight = 'balanced'
                else:
                    class_weight = self.class_weight
                expanded_class_weight = compute_sample_weight(class_weight, y_original)
        return (y, expanded_class_weight)
