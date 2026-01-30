import warnings
from numbers import Integral
import numpy as np
from sklearn.base import (
    BaseEstimator,
    MetaEstimatorMixin,
    _fit_context,
    clone,
    is_classifier,
)
from sklearn.feature_selection._base import SelectorMixin, _get_feature_importances
from sklearn.utils._param_validation import HasMethods, Interval, RealNotInt
from sklearn.utils.validation import (
    _check_method_params,
    _estimator_has,
    check_is_fitted,
    validate_data,
)

class RFE(SelectorMixin, MetaEstimatorMixin, BaseEstimator):
    """Feature ranking with recursive feature elimination.

    Given an external estimator that assigns weights to features (e.g., the
    coefficients of a linear model), the goal of recursive feature elimination
    (RFE) is to select features by recursively considering smaller and smaller
    sets of features. First, the estimator is trained on the initial set of
    features and the importance of each feature is obtained either through
    any specific attribute or callable.
    Then, the least important features are pruned from current set of features.
    That procedure is recursively repeated on the pruned set until the desired
    number of features to select is eventually reached.

    Read more in the :ref:`User Guide <rfe>`.

    Parameters
    ----------
    estimator : ``Estimator`` instance
        A supervised learning estimator with a ``fit`` method that provides
        information about feature importance
        (e.g. `coef_`, `feature_importances_`).

    n_features_to_select : int or float, default=None
        The number of features to select. If `None`, half of the features are
        selected. If integer, the parameter is the absolute number of features
        to select. If float between 0 and 1, it is the fraction of features to
        select.

        .. versionchanged:: 0.24
           Added float values for fractions.

    step : int or float, default=1
        If greater than or equal to 1, then ``step`` corresponds to the
        (integer) number of features to remove at each iteration.
        If within (0.0, 1.0), then ``step`` corresponds to the percentage
        (rounded down) of features to remove at each iteration.

    verbose : int, default=0
        Controls verbosity of output.

    importance_getter : str or callable, default='auto'
        If 'auto', uses the feature importance either through a `coef_`
        or `feature_importances_` attributes of estimator.

        Also accepts a string that specifies an attribute name/path
        for extracting feature importance (implemented with `attrgetter`).
        For example, give `regressor_.coef_` in case of
        :class:`~sklearn.compose.TransformedTargetRegressor`  or
        `named_steps.clf.feature_importances_` in case of
        class:`~sklearn.pipeline.Pipeline` with its last step named `clf`.

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

    n_features_ : int
        The number of selected features.

    n_features_in_ : int
        Number of features seen during :term:`fit`. Only defined if the
        underlying estimator exposes such an attribute when fit.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    ranking_ : ndarray of shape (n_features,)
        The feature ranking, such that ``ranking_[i]`` corresponds to the
        ranking position of the i-th feature. Selected (i.e., estimated
        best) features are assigned rank 1.

    support_ : ndarray of shape (n_features,)
        The mask of selected features.

    See Also
    --------
    RFECV : Recursive feature elimination with built-in cross-validated
        selection of the best number of features.
    SelectFromModel : Feature selection based on thresholds of importance
        weights.
    SequentialFeatureSelector : Sequential cross-validation based feature
        selection. Does not rely on importance weights.

    Notes
    -----
    Allows NaN/Inf in the input if the underlying estimator does as well.

    References
    ----------

    .. [1] Guyon, I., Weston, J., Barnhill, S., & Vapnik, V., "Gene selection
           for cancer classification using support vector machines",
           Mach. Learn., 46(1-3), 389--422, 2002.

    Examples
    --------
    The following example shows how to retrieve the 5 most informative
    features in the Friedman #1 dataset.

    >>> from sklearn.datasets import make_friedman1
    >>> from sklearn.feature_selection import RFE
    >>> from sklearn.svm import SVR
    >>> X, y = make_friedman1(n_samples=50, n_features=10, random_state=0)
    >>> estimator = SVR(kernel="linear")
    >>> selector = RFE(estimator, n_features_to_select=5, step=1)
    >>> selector = selector.fit(X, y)
    >>> selector.support_
    array([ True,  True,  True,  True,  True, False, False, False, False,
           False])
    >>> selector.ranking_
    array([1, 1, 1, 1, 1, 6, 4, 3, 2, 5])
    """
    _parameter_constraints: dict = {'estimator': [HasMethods(['fit'])], 'n_features_to_select': [None, Interval(RealNotInt, 0, 1, closed='right'), Interval(Integral, 0, None, closed='neither')], 'step': [Interval(Integral, 0, None, closed='neither'), Interval(RealNotInt, 0, 1, closed='neither')], 'verbose': ['verbose'], 'importance_getter': [str, callable]}

    def __init__(self, estimator, *, n_features_to_select=None, step=1, verbose=0, importance_getter='auto'):
        self.estimator = estimator
        self.n_features_to_select = n_features_to_select
        self.step = step
        self.importance_getter = importance_getter
        self.verbose = verbose

    def _fit(self, X, y, step_score=None, **fit_params):
        X, y = validate_data(self, X, y, accept_sparse='csc', ensure_min_features=2, ensure_all_finite=False, multi_output=True)
        n_features = X.shape[1]
        if self.n_features_to_select is None:
            n_features_to_select = n_features // 2
        elif isinstance(self.n_features_to_select, Integral):
            n_features_to_select = self.n_features_to_select
            if n_features_to_select > n_features:
                warnings.warn(f'Found n_features_to_select={n_features_to_select!r} > n_features={n_features!r}. There will be no feature selection and all features will be kept.', UserWarning)
        else:
            n_features_to_select = int(n_features * self.n_features_to_select)
        if 0.0 < self.step < 1.0:
            step = int(max(1, self.step * n_features))
        else:
            step = int(self.step)
        support_ = np.ones(n_features, dtype=bool)
        ranking_ = np.ones(n_features, dtype=int)
        if step_score:
            self.step_n_features_ = []
            self.step_scores_ = []
            self.step_support_ = []
            self.step_ranking_ = []
        while np.sum(support_) > n_features_to_select:
            features = np.arange(n_features)[support_]
            estimator = clone(self.estimator)
            if self.verbose > 0:
                print('Fitting estimator with %d features.' % np.sum(support_))
            estimator.fit(X[:, features], y, **fit_params)
            if step_score:
                self.step_n_features_.append(len(features))
                self.step_scores_.append(step_score(estimator, features))
                self.step_support_.append(list(support_))
                self.step_ranking_.append(list(ranking_))
            importances = _get_feature_importances(estimator, self.importance_getter, transform_func='square')
            ranks = np.argsort(importances)
            ranks = np.ravel(ranks)
            threshold = min(step, np.sum(support_) - n_features_to_select)
            support_[features[ranks][:threshold]] = False
            ranking_[np.logical_not(support_)] += 1
        features = np.arange(n_features)[support_]
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X[:, features], y, **fit_params)
        if step_score:
            self.step_n_features_.append(len(features))
            self.step_scores_.append(step_score(self.estimator_, features))
            self.step_support_.append(support_)
            self.step_ranking_.append(ranking_)
        self.n_features_ = support_.sum()
        self.support_ = support_
        self.ranking_ = ranking_
        return self
