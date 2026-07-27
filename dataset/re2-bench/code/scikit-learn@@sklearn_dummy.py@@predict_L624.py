from numbers import Integral, Real
import numpy as np
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    MultiOutputMixin,
    RegressorMixin,
    _fit_context,
)
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import (
    _check_sample_weight,
    _num_samples,
    check_array,
    check_consistent_length,
    check_is_fitted,
    validate_data,
)

class DummyRegressor(MultiOutputMixin, RegressorMixin, BaseEstimator):
    """Regressor that makes predictions using simple rules.

    This regressor is useful as a simple baseline to compare with other
    (real) regressors. Do not use it for real problems.

    Read more in the :ref:`User Guide <dummy_estimators>`.

    .. versionadded:: 0.13

    Parameters
    ----------
    strategy : {"mean", "median", "quantile", "constant"}, default="mean"
        Strategy to use to generate predictions.

        * "mean": always predicts the mean of the training set
        * "median": always predicts the median of the training set
        * "quantile": always predicts a specified quantile of the training set,
          provided with the quantile parameter.
        * "constant": always predicts a constant value that is provided by
          the user.

    constant : int or float or array-like of shape (n_outputs,), default=None
        The explicit constant as predicted by the "constant" strategy. This
        parameter is useful only for the "constant" strategy.

    quantile : float in [0.0, 1.0], default=None
        The quantile to predict using the "quantile" strategy. A quantile of
        0.5 corresponds to the median, while 0.0 to the minimum and 1.0 to the
        maximum.

    Attributes
    ----------
    constant_ : ndarray of shape (1, n_outputs)
        Mean or median or quantile of the training targets or constant value
        given by the user.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X` has
        feature names that are all strings.

    n_outputs_ : int
        Number of outputs.

    See Also
    --------
    DummyClassifier: Classifier that makes predictions using simple rules.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.dummy import DummyRegressor
    >>> X = np.array([1.0, 2.0, 3.0, 4.0])
    >>> y = np.array([2.0, 3.0, 5.0, 10.0])
    >>> dummy_regr = DummyRegressor(strategy="mean")
    >>> dummy_regr.fit(X, y)
    DummyRegressor()
    >>> dummy_regr.predict(X)
    array([5., 5., 5., 5.])
    >>> dummy_regr.score(X, y)
    0.0
    """
    _parameter_constraints: dict = {'strategy': [StrOptions({'mean', 'median', 'quantile', 'constant'})], 'quantile': [Interval(Real, 0.0, 1.0, closed='both'), None], 'constant': [Interval(Real, None, None, closed='neither'), 'array-like', None]}

    def __init__(self, *, strategy='mean', constant=None, quantile=None):
        self.strategy = strategy
        self.constant = constant
        self.quantile = quantile

    def predict(self, X, return_std=False):
        """Perform classification on test vectors X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data.

        return_std : bool, default=False
            Whether to return the standard deviation of posterior prediction.
            All zeros in this case.

            .. versionadded:: 0.20

        Returns
        -------
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Predicted target values for X.

        y_std : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Standard deviation of predictive distribution of query points.
        """
        check_is_fitted(self)
        n_samples = _num_samples(X)
        y = np.full((n_samples, self.n_outputs_), self.constant_, dtype=np.array(self.constant_).dtype)
        y_std = np.zeros((n_samples, self.n_outputs_))
        if self.n_outputs_ == 1:
            y = np.ravel(y)
            y_std = np.ravel(y_std)
        return (y, y_std) if return_std else y
