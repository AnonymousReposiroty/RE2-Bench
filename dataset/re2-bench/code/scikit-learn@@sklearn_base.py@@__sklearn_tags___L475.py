from sklearn.utils._metadata_requests import _MetadataRequester, _routing_enabled
from sklearn.utils._repr_html.base import ReprHTMLMixin, _HTMLDocumentationLinkMixin
from sklearn.utils._repr_html.estimator import estimator_html_repr
from sklearn.utils._tags import (
    ClassifierTags,
    RegressorTags,
    Tags,
    TargetTags,
    TransformerTags,
    get_tags,
)

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

    def __sklearn_tags__(self):
        return Tags(estimator_type=None, target_tags=TargetTags(required=False), transformer_tags=None, regressor_tags=None, classifier_tags=None)
