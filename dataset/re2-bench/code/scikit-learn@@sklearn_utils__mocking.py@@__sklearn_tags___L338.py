from sklearn.base import BaseEstimator, ClassifierMixin

class CheckingClassifier(ClassifierMixin, BaseEstimator):
    """Dummy classifier to test pipelining and meta-estimators.

    Checks some property of `X` and `y`in fit / predict.
    This allows testing whether pipelines / cross-validation or metaestimators
    changed the input.

    Can also be used to check if `fit_params` are passed correctly, and
    to force a certain score to be returned.

    Parameters
    ----------
    check_y, check_X : callable, default=None
        The callable used to validate `X` and `y`. These callable should return
        a bool where `False` will trigger an `AssertionError`. If `None`, the
        data is not validated. Default is `None`.

    check_y_params, check_X_params : dict, default=None
        The optional parameters to pass to `check_X` and `check_y`. If `None`,
        then no parameters are passed in.

    methods_to_check : "all" or list of str, default="all"
        The methods in which the checks should be applied. By default,
        all checks will be done on all methods (`fit`, `predict`,
        `predict_proba`, `decision_function` and `score`).

    foo_param : int, default=0
        A `foo` param. When `foo > 1`, the output of :meth:`score` will be 1
        otherwise it is 0.

    expected_sample_weight : bool, default=False
        Whether to check if a valid `sample_weight` was passed to `fit`.

    expected_fit_params : list of str, default=None
        A list of the expected parameters given when calling `fit`.

    Attributes
    ----------
    classes_ : int
        The classes seen during `fit`.

    n_features_in_ : int
        The number of features seen during `fit`.

    Examples
    --------
    >>> from sklearn.utils._mocking import CheckingClassifier

    This helper allow to assert to specificities regarding `X` or `y`. In this
    case we expect `check_X` or `check_y` to return a boolean.

    >>> from sklearn.datasets import load_iris
    >>> X, y = load_iris(return_X_y=True)
    >>> clf = CheckingClassifier(check_X=lambda x: x.shape == (150, 4))
    >>> clf.fit(X, y)
    CheckingClassifier(...)

    We can also provide a check which might raise an error. In this case, we
    expect `check_X` to return `X` and `check_y` to return `y`.

    >>> from sklearn.utils import check_array
    >>> clf = CheckingClassifier(check_X=check_array)
    >>> clf.fit(X, y)
    CheckingClassifier(...)
    """

    def __init__(self, *, check_y=None, check_y_params=None, check_X=None, check_X_params=None, methods_to_check='all', foo_param=0, expected_sample_weight=None, expected_fit_params=None, random_state=None):
        self.check_y = check_y
        self.check_y_params = check_y_params
        self.check_X = check_X
        self.check_X_params = check_X_params
        self.methods_to_check = methods_to_check
        self.foo_param = foo_param
        self.expected_sample_weight = expected_sample_weight
        self.expected_fit_params = expected_fit_params
        self.random_state = random_state

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags._skip_test = True
        tags.input_tags.two_d_array = False
        tags.target_tags.one_d_labels = True
        return tags
