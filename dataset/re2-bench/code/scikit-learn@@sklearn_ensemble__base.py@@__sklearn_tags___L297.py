from abc import ABCMeta, abstractmethod
from sklearn.base import (
    BaseEstimator,
    MetaEstimatorMixin,
    clone,
    is_classifier,
    is_regressor,
)
from sklearn.utils._tags import get_tags
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

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        try:
            tags.input_tags.allow_nan = all((get_tags(est[1]).input_tags.allow_nan if est[1] != 'drop' else True for est in self.estimators))
            tags.input_tags.sparse = all((get_tags(est[1]).input_tags.sparse if est[1] != 'drop' else True for est in self.estimators))
        except Exception:
            pass
        return tags
