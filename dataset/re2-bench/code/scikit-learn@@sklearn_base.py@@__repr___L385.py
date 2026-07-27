import re
from sklearn.utils._metadata_requests import _MetadataRequester, _routing_enabled
from sklearn.utils._repr_html.base import ReprHTMLMixin, _HTMLDocumentationLinkMixin
from sklearn.utils._repr_html.estimator import estimator_html_repr
from sklearn.utils._pprint import _EstimatorPrettyPrinter

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

    def __repr__(self, N_CHAR_MAX=700):
        from sklearn.utils._pprint import _EstimatorPrettyPrinter
        N_MAX_ELEMENTS_TO_SHOW = 30
        pp = _EstimatorPrettyPrinter(compact=True, indent=1, indent_at_name=True, n_max_elements_to_show=N_MAX_ELEMENTS_TO_SHOW)
        repr_ = pp.pformat(self)
        n_nonblank = len(''.join(repr_.split()))
        if n_nonblank > N_CHAR_MAX:
            lim = N_CHAR_MAX // 2
            regex = '^(\\s*\\S){%d}' % lim
            left_lim = re.match(regex, repr_).end()
            right_lim = re.match(regex, repr_[::-1]).end()
            if '\n' in repr_[left_lim:-right_lim]:
                regex += '[^\\n]*\\n'
                right_lim = re.match(regex, repr_[::-1]).end()
            ellipsis = '...'
            if left_lim + len(ellipsis) < len(repr_) - right_lim:
                repr_ = repr_[:left_lim] + '...' + repr_[-right_lim:]
        return repr_
