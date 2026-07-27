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

    def get_params(self, deep=True):
        """
        Get the parameters of an estimator from the ensemble.

        Returns the parameters given in the constructor as well as the
        estimators contained within the `estimators` parameter.

        Parameters
        ----------
        deep : bool, default=True
            Setting it to True gets the various estimators and the parameters
            of the estimators as well.

        Returns
        -------
        params : dict
            Parameter and estimator names mapped to their values or parameter
            names mapped to their values.
        """
        return super()._get_params('estimators', deep=deep)
