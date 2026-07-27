import inspect
from collections import defaultdict
from sklearn.utils._metadata_requests import _MetadataRequester, _routing_enabled
from sklearn.utils._repr_html.base import ReprHTMLMixin, _HTMLDocumentationLinkMixin
from sklearn.utils._repr_html.estimator import estimator_html_repr

class BaseEstimator(ReprHTMLMixin, _HTMLDocumentationLinkMixin, _MetadataRequester):
    """Base class for all estimators in scikit-learn.

    Inheriting from this class provides default implementations of:

    - setting and getting parameters used by `GridSearchCV` and friends;
    - textual and HTML representation displayed in terminals and IDEs;
    - estimator serialization;
    - parameters validation;
    - data validation;
    - feature names validation.

    Read more in the :ref:`User Guide <rolling_your_own_estimator>`.


    Notes
    -----
    All estimators should specify all the parameters that can be set
    at the class level in their ``__init__`` as explicit keyword
    arguments (no ``*args`` or ``**kwargs``).

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.base import BaseEstimator
    >>> class MyEstimator(BaseEstimator):
    ...     def __init__(self, *, param=1):
    ...         self.param = param
    ...     def fit(self, X, y=None):
    ...         self.is_fitted_ = True
    ...         return self
    ...     def predict(self, X):
    ...         return np.full(shape=X.shape[0], fill_value=self.param)
    >>> estimator = MyEstimator(param=2)
    >>> estimator.get_params()
    {'param': 2}
    >>> X = np.array([[1, 2], [2, 3], [3, 4]])
    >>> y = np.array([1, 0, 1])
    >>> estimator.fit(X, y).predict(X)
    array([2, 2, 2])
    >>> estimator.set_params(param=3).fit(X, y).predict(X)
    array([3, 3, 3])
    """
    _html_repr = estimator_html_repr

    @classmethod
    def _get_param_names(cls):
        """Get parameter names for the estimator"""
        init = getattr(cls.__init__, 'deprecated_original', cls.__init__)
        if init is object.__init__:
            return []
        init_signature = inspect.signature(init)
        parameters = [p for p in init_signature.parameters.values() if p.name != 'self' and p.kind != p.VAR_KEYWORD]
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError("scikit-learn estimators should always specify their parameters in the signature of their __init__ (no varargs). %s with constructor %s doesn't  follow this convention." % (cls, init_signature))
        return sorted([p.name for p in parameters])

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        out = dict()
        for key in self._get_param_names():
            value = getattr(self, key)
            if deep and hasattr(value, 'get_params') and (not isinstance(value, type)):
                deep_items = value.get_params().items()
                out.update(((key + '__' + k, val) for k, val in deep_items))
            out[key] = value
        return out

    def set_params(self, **params):
        """Set the parameters of this estimator.

        The method works on simple estimators as well as on nested objects
        (such as :class:`~sklearn.pipeline.Pipeline`). The latter have
        parameters of the form ``<component>__<parameter>`` so that it's
        possible to update each component of a nested object.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : estimator instance
            Estimator instance.
        """
        if not params:
            return self
        valid_params = self.get_params(deep=True)
        nested_params = defaultdict(dict)
        for key, value in params.items():
            key, delim, sub_key = key.partition('__')
            if key not in valid_params:
                local_valid_params = self._get_param_names()
                raise ValueError(f'Invalid parameter {key!r} for estimator {self}. Valid parameters are: {local_valid_params!r}.')
            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)
                valid_params[key] = value
        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)
        return self
