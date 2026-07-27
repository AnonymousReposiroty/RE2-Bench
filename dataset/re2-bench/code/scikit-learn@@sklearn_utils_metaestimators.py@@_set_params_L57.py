from abc import ABCMeta, abstractmethod
from contextlib import suppress
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

    def _set_params(self, attr, **params):
        if attr in params:
            setattr(self, attr, params.pop(attr))
        items = getattr(self, attr)
        if isinstance(items, list) and items:
            with suppress(TypeError):
                item_names, _ = zip(*items)
                for name in list(params.keys()):
                    if '__' not in name and name in item_names:
                        self._replace_estimator(attr, name, params.pop(name))
        super().set_params(**params)
        return self

    def _replace_estimator(self, attr, name, new_val):
        new_estimators = list(getattr(self, attr))
        for i, (estimator_name, _) in enumerate(new_estimators):
            if estimator_name == name:
                new_estimators[i] = (name, new_val)
                break
        setattr(self, attr, new_estimators)
