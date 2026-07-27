from abc import ABCMeta, abstractmethod
from sklearn.base import (
    BaseEstimator,
    MetaEstimatorMixin,
    clone,
    is_classifier,
    is_regressor,
)

class BaseEnsemble(MetaEstimatorMixin, BaseEstimator, metaclass=ABCMeta):
    """Base class for all ensemble classes.

    Warning: This class should not be used directly. Use derived classes
    instead.

    Parameters
    ----------
    estimator : object
        The base estimator from which the ensemble is built.

    n_estimators : int, default=10
        The number of estimators in the ensemble.

    estimator_params : list of str, default=tuple()
        The list of attributes to use as parameters when instantiating a
        new base estimator. If none are given, default parameters are used.

    Attributes
    ----------
    estimator_ : estimator
        The base estimator from which the ensemble is grown.

    estimators_ : list of estimators
        The collection of fitted base estimators.
    """

    @abstractmethod
    def __init__(self, estimator=None, *, n_estimators=10, estimator_params=tuple()):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.estimator_params = estimator_params

    def _make_estimator(self, append=True, random_state=None):
        """Make and configure a copy of the `estimator_` attribute.

        Warning: This method should be used to properly instantiate new
        sub-estimators.
        """
        estimator = clone(self.estimator_)
        estimator.set_params(**{p: getattr(self, p) for p in self.estimator_params})
        if random_state is not None:
            _set_random_states(estimator, random_state)
        if append:
            self.estimators_.append(estimator)
        return estimator
