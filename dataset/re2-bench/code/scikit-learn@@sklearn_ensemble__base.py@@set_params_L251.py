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

    def set_params(self, **params):
        """
        Set the parameters of an estimator from the ensemble.

        Valid parameter keys can be listed with `get_params()`. Note that you
        can directly set the parameters of the estimators contained in
        `estimators`.

        Parameters
        ----------
        **params : keyword arguments
            Specific parameters using e.g.
            `set_params(parameter_name=new_value)`. In addition, to setting the
            parameters of the estimator, the individual estimator of the
            estimators can also be set, or can be removed by setting them to
            'drop'.

        Returns
        -------
        self : object
            Estimator instance.
        """
        super()._set_params('estimators', **params)
        return self
