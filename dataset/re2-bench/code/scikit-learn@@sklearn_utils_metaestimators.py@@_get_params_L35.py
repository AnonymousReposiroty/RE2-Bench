from abc import ABCMeta, abstractmethod
from sklearn.base import BaseEstimator

class _BaseComposition(BaseEstimator, metaclass=ABCMeta):
    """Base class for estimators that are composed of named sub-estimators.

    This abstract class provides parameter management functionality for
    meta-estimators that contain collections of named estimators. It handles
    the complex logic for getting and setting parameters on nested estimators
    using the "estimator_name__parameter" syntax.

    The class is designed to work with any attribute containing a list of
    (name, estimator) tuples.
    """

    @abstractmethod
    def __init__(self):
        pass

    def _get_params(self, attr, deep=True):
        out = super().get_params(deep=deep)
        if not deep:
            return out
        estimators = getattr(self, attr)
        try:
            out.update(estimators)
        except (TypeError, ValueError):
            return out
        for name, estimator in estimators:
            if hasattr(estimator, 'get_params'):
                for key, value in estimator.get_params(deep=True).items():
                    out['%s__%s' % (name, key)] = value
        return out
