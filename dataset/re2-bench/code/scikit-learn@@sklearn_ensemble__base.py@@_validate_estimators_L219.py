from abc import ABCMeta, abstractmethod
from sklearn.base import (
    BaseEstimator,
    MetaEstimatorMixin,
    clone,
    is_classifier,
    is_regressor,
)
from sklearn.utils.metaestimators import _BaseComposition

class _BaseHeterogeneousEnsemble(MetaEstimatorMixin, _BaseComposition, metaclass=ABCMeta):
    """Base class for heterogeneous ensemble of learners.

    Parameters
    ----------
    estimators : list of (str, estimator) tuples
        The ensemble of estimators to use in the ensemble. Each element of the
        list is defined as a tuple of string (i.e. name of the estimator) and
        an estimator instance. An estimator can be set to `'drop'` using
        `set_params`.

    Attributes
    ----------
    estimators_ : list of estimators
        The elements of the estimators parameter, having been fitted on the
        training data. If an estimator has been set to `'drop'`, it will not
        appear in `estimators_`.
    """

    @abstractmethod
    def __init__(self, estimators):
        self.estimators = estimators

    def _validate_estimators(self):
        if len(self.estimators) == 0 or not all((isinstance(item, (tuple, list)) and isinstance(item[0], str) for item in self.estimators)):
            raise ValueError("Invalid 'estimators' attribute, 'estimators' should be a non-empty list of (string, estimator) tuples.")
        names, estimators = zip(*self.estimators)
        self._validate_names(names)
        has_estimator = any((est != 'drop' for est in estimators))
        if not has_estimator:
            raise ValueError('All estimators are dropped. At least one is required to be an estimator.')
        is_estimator_type = is_classifier if is_classifier(self) else is_regressor
        for est in estimators:
            if est != 'drop' and (not is_estimator_type(est)):
                raise ValueError('The estimator {} should be a {}.'.format(est.__class__.__name__, is_estimator_type.__name__[3:]))
        return (names, estimators)
