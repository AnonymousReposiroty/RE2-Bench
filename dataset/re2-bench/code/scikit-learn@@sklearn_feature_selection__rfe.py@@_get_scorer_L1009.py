from numbers import Integral
from sklearn.base import (
    BaseEstimator,
    MetaEstimatorMixin,
    _fit_context,
    clone,
    is_classifier,
)
from sklearn.metrics import get_scorer
from sklearn.utils import Bunch, metadata_routing
from sklearn.utils._param_validation import HasMethods, Interval, RealNotInt

class RFECV(RFE):
    """Recursive feature elimination with cross-validation to select features.

    The number of features selected is tuned automatically by fitting an :class:`RFE`
    selector on the different cross-validation splits (provided by the `cv` parameter).
    The performance of each :class:`RFE` selector is evaluated using `scoring` for
    different numbers of selected features and aggregated together. Finally, the scores
    are averaged across folds and the number of features selected is set to the number
    of features that maximize the cross-validation score.

    See glossary entry for :term:`cross-validation estimator`.

    Read more in the :ref:`User Guide <rfe>`.

    Parameters
    ----------
    estimator : ``Estimator`` instance
        A supervised learning estimator with a ``fit`` method that provides
        information about feature importance either through a ``coef_``
        attribute or through a ``feature_importances_`` attribute.

    step : int or float, default=1
        If greater than or equal to 1, then ``step`` corresponds to the
        (integer) number of features to remove at each iteration.
        If within (0.0, 1.0), then ``step`` corresponds to the percentage
        (rounded down) of features to remove at each iteration.
        Note that the last iteration may remove fewer than ``step`` features in
        order to reach ``min_features_to_select``.

    min_features_to_select : int, default=1
        The minimum number of features to be selected. This number of features
        will always be scored, even if the difference between the original
        feature count and ``min_features_to_select`` isn't divisible by
        ``step``.

        .. versionadded:: 0.20

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 5-fold cross-validation,
        - integer, to specify the number of folds.
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`~sklearn.model_selection.StratifiedKFold` is used. If the
        estimator is not a classifier or if ``y`` is neither binary nor multiclass,
        :class:`~sklearn.model_selection.KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

        .. versionchanged:: 0.22
            ``cv`` default value of None changed from 3-fold to 5-fold.

    scoring : str or callable, default=None
        Scoring method to evaluate the :class:`RFE` selectors' performance. Options:

        - str: see :ref:`scoring_string_names` for options.
        - callable: a scorer callable object (e.g., function) with signature
          ``scorer(estimator, X, y)``. See :ref:`scoring_callable` for details.
        - `None`: the `estimator`'s
          :ref:`default evaluation criterion <scoring_api_overview>` is used.

    verbose : int, default=0
        Controls verbosity of output.

    n_jobs : int or None, default=None
        Number of cores to run in parallel while fitting across folds.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

        .. versionadded:: 0.18

    importance_getter : str or callable, default='auto'
        If 'auto', uses the feature importance either through a `coef_`
        or `feature_importances_` attributes of estimator.

        Also accepts a string that specifies an attribute name/path
        for extracting feature importance.
        For example, give `regressor_.coef_` in case of
        :class:`~sklearn.compose.TransformedTargetRegressor`  or
        `named_steps.clf.feature_importances_` in case of
        :class:`~sklearn.pipeline.Pipeline` with its last step named `clf`.

        If `callable`, overrides the default feature importance getter.
        The callable is passed with the fitted estimator and it should
        return importance for each feature.

        .. versionadded:: 0.24

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        The classes labels. Only available when `estimator` is a classifier.

    estimator_ : ``Estimator`` instance
        The fitted estimator used to select features.

    cv_results_ : dict of ndarrays
        All arrays (values of the dictionary) are sorted in ascending order
        by the number of features used (i.e., the first element of the array
        represents the models that used the least number of features, while the
        last element represents the models that used all available features).

        .. versionadded:: 1.0

        This dictionary contains the following keys:

        split(k)_test_score : ndarray of shape (n_subsets_of_features,)
            The cross-validation scores across (k)th fold.

        mean_test_score : ndarray of shape (n_subsets_of_features,)
            Mean of scores over the folds.

        std_test_score : ndarray of shape (n_subsets_of_features,)
            Standard deviation of scores over the folds.

        n_features : ndarray of shape (n_subsets_of_features,)
            Number of features used at each step.

            .. versionadded:: 1.5

        split(k)_ranking : ndarray of shape (n_subsets_of_features,)
            The cross-validation rankings across (k)th fold.
            Selected (i.e., estimated best) features are assigned rank 1.
            Illustration in
            :ref:`sphx_glr_auto_examples_feature_selection_plot_rfe_with_cross_validation.py`

            .. versionadded:: 1.7

        split(k)_support : ndarray of shape (n_subsets_of_features,)
            The cross-validation supports across (k)th fold. The support
            is the mask of selected features.

            .. versionadded:: 1.7

    n_features_ : int
        The number of selected features with cross-validation.

    n_features_in_ : int
        Number of features seen during :term:`fit`. Only defined if the
        underlying estimator exposes such an attribute when fit.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    ranking_ : narray of shape (n_features,)
        The feature ranking, such that `ranking_[i]`
        corresponds to the ranking
        position of the i-th feature.
        Selected (i.e., estimated best)
        features are assigned rank 1.

    support_ : ndarray of shape (n_features,)
        The mask of selected features.

    See Also
    --------
    RFE : Recursive feature elimination.

    Notes
    -----
    The size of all values in ``cv_results_`` is equal to
    ``ceil((n_features - min_features_to_select) / step) + 1``,
    where step is the number of features removed at each iteration.

    Allows NaN/Inf in the input if the underlying estimator does as well.

    References
    ----------

    .. [1] Guyon, I., Weston, J., Barnhill, S., & Vapnik, V., "Gene selection
           for cancer classification using support vector machines",
           Mach. Learn., 46(1-3), 389--422, 2002.

    Examples
    --------
    The following example shows how to retrieve the a-priori not known 5
    informative features in the Friedman #1 dataset.

    >>> from sklearn.datasets import make_friedman1
    >>> from sklearn.feature_selection import RFECV
    >>> from sklearn.svm import SVR
    >>> X, y = make_friedman1(n_samples=50, n_features=10, random_state=0)
    >>> estimator = SVR(kernel="linear")
    >>> selector = RFECV(estimator, step=1, cv=5)
    >>> selector = selector.fit(X, y)
    >>> selector.support_
    array([ True,  True,  True,  True,  True, False, False, False, False,
           False])
    >>> selector.ranking_
    array([1, 1, 1, 1, 1, 6, 4, 3, 2, 5])

    For a detailed example of using RFECV to select features when training a
    :class:`~sklearn.linear_model.LogisticRegression`, see
    :ref:`sphx_glr_auto_examples_feature_selection_plot_rfe_with_cross_validation.py`.
    """
    _parameter_constraints: dict = {**RFE._parameter_constraints, 'min_features_to_select': [Interval(Integral, 0, None, closed='neither')], 'cv': ['cv_object'], 'scoring': [None, str, callable], 'n_jobs': [None, Integral]}
    _parameter_constraints.pop('n_features_to_select')
    __metadata_request__fit = {'groups': metadata_routing.UNUSED}

    def __init__(self, estimator, *, step=1, min_features_to_select=1, cv=None, scoring=None, verbose=0, n_jobs=None, importance_getter='auto'):
        self.estimator = estimator
        self.step = step
        self.importance_getter = importance_getter
        self.cv = cv
        self.scoring = scoring
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.min_features_to_select = min_features_to_select

    def _get_scorer(self):
        if self.scoring is None:
            scoring = 'accuracy' if is_classifier(self.estimator) else 'r2'
        else:
            scoring = self.scoring
        return get_scorer(scoring)
