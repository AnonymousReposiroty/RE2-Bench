import numpy as np
from sklearn.base import (
    ClassifierMixin,
    RegressorMixin,
    TransformerMixin,
    _fit_context,
    clone,
)
from sklearn.utils.validation import (
    _check_feature_names_in,
    check_is_fitted,
    column_or_1d,
)

class VotingRegressor(RegressorMixin, _BaseVoting):
    """Prediction voting regressor for unfitted estimators.

    A voting regressor is an ensemble meta-estimator that fits several base
    regressors, each on the whole dataset. Then it averages the individual
    predictions to form a final prediction.

    For a detailed example, refer to
    :ref:`sphx_glr_auto_examples_ensemble_plot_voting_regressor.py`.

    Read more in the :ref:`User Guide <voting_regressor>`.

    .. versionadded:: 0.21

    Parameters
    ----------
    estimators : list of (str, estimator) tuples
        Invoking the ``fit`` method on the ``VotingRegressor`` will fit clones
        of those original estimators that will be stored in the class attribute
        ``self.estimators_``. An estimator can be set to ``'drop'`` using
        :meth:`set_params`.

        .. versionchanged:: 0.21
            ``'drop'`` is accepted. Using None was deprecated in 0.22 and
            support was removed in 0.24.

    weights : array-like of shape (n_regressors,), default=None
        Sequence of weights (`float` or `int`) to weight the occurrences of
        predicted values before averaging. Uses uniform weights if `None`.

    n_jobs : int, default=None
        The number of jobs to run in parallel for ``fit``.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    verbose : bool, default=False
        If True, the time elapsed while fitting will be printed as it
        is completed.

        .. versionadded:: 0.23

    Attributes
    ----------
    estimators_ : list of regressors
        The collection of fitted sub-estimators as defined in ``estimators``
        that are not 'drop'.

    named_estimators_ : :class:`~sklearn.utils.Bunch`
        Attribute to access any fitted sub-estimators by name.

        .. versionadded:: 0.20

    n_features_in_ : int
        Number of features seen during :term:`fit`. Only defined if the
        underlying regressor exposes such an attribute when fit.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Only defined if the
        underlying estimators expose such an attribute when fit.

        .. versionadded:: 1.0

    See Also
    --------
    VotingClassifier : Soft Voting/Majority Rule classifier.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import LinearRegression
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> from sklearn.ensemble import VotingRegressor
    >>> from sklearn.neighbors import KNeighborsRegressor
    >>> r1 = LinearRegression()
    >>> r2 = RandomForestRegressor(n_estimators=10, random_state=1)
    >>> r3 = KNeighborsRegressor()
    >>> X = np.array([[1, 1], [2, 4], [3, 9], [4, 16], [5, 25], [6, 36]])
    >>> y = np.array([2, 6, 12, 20, 30, 42])
    >>> er = VotingRegressor([('lr', r1), ('rf', r2), ('r3', r3)])
    >>> print(er.fit(X, y).predict(X))
    [ 6.8  8.4 12.5 17.8 26  34]

    In the following example, we drop the `'lr'` estimator with
    :meth:`~VotingRegressor.set_params` and fit the remaining two estimators:

    >>> er = er.set_params(lr='drop')
    >>> er = er.fit(X, y)
    >>> len(er.estimators_)
    2
    """

    def __init__(self, estimators, *, weights=None, n_jobs=None, verbose=False):
        super().__init__(estimators=estimators)
        self.weights = weights
        self.n_jobs = n_jobs
        self.verbose = verbose

    def predict(self, X):
        """Predict regression target for X.

        The predicted regression target of an input sample is computed as the
        mean predicted regression targets of the estimators in the ensemble.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted values.
        """
        check_is_fitted(self)
        return np.average(self._predict(X), axis=1, weights=self._weights_not_none)
