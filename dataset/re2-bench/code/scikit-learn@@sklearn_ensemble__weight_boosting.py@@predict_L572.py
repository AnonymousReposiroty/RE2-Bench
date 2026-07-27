import numpy as np
from sklearn.base import (
    ClassifierMixin,
    RegressorMixin,
    _fit_context,
    is_classifier,
    is_regressor,
)
from sklearn.utils.metadata_routing import (
    _raise_for_unsupported_routing,
    _RoutingNotSupportedMixin,
)
from sklearn.utils.validation import (
    _check_sample_weight,
    _num_samples,
    check_is_fitted,
    has_fit_parameter,
    validate_data,
)

class AdaBoostClassifier(_RoutingNotSupportedMixin, ClassifierMixin, BaseWeightBoosting):
    """An AdaBoost classifier.

    An AdaBoost [1]_ classifier is a meta-estimator that begins by fitting a
    classifier on the original dataset and then fits additional copies of the
    classifier on the same dataset but where the weights of incorrectly
    classified instances are adjusted such that subsequent classifiers focus
    more on difficult cases.

    This class implements the algorithm based on [2]_.

    Read more in the :ref:`User Guide <adaboost>`.

    .. versionadded:: 0.14

    Parameters
    ----------
    estimator : object, default=None
        The base estimator from which the boosted ensemble is built.
        Support for sample weighting is required, as well as proper
        ``classes_`` and ``n_classes_`` attributes. If ``None``, then
        the base estimator is :class:`~sklearn.tree.DecisionTreeClassifier`
        initialized with `max_depth=1`.

        .. versionadded:: 1.2
           `base_estimator` was renamed to `estimator`.

    n_estimators : int, default=50
        The maximum number of estimators at which boosting is terminated.
        In case of perfect fit, the learning procedure is stopped early.
        Values must be in the range `[1, inf)`.

    learning_rate : float, default=1.0
        Weight applied to each classifier at each boosting iteration. A higher
        learning rate increases the contribution of each classifier. There is
        a trade-off between the `learning_rate` and `n_estimators` parameters.
        Values must be in the range `(0.0, inf)`.

    random_state : int, RandomState instance or None, default=None
        Controls the random seed given at each `estimator` at each
        boosting iteration.
        Thus, it is only used when `estimator` exposes a `random_state`.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Attributes
    ----------
    estimator_ : estimator
        The base estimator from which the ensemble is grown.

        .. versionadded:: 1.2
           `base_estimator_` was renamed to `estimator_`.

    estimators_ : list of classifiers
        The collection of fitted sub-estimators.

    classes_ : ndarray of shape (n_classes,)
        The classes labels.

    n_classes_ : int
        The number of classes.

    estimator_weights_ : ndarray of floats
        Weights for each estimator in the boosted ensemble.

    estimator_errors_ : ndarray of floats
        Classification error for each estimator in the boosted
        ensemble.

    feature_importances_ : ndarray of shape (n_features,)
        The impurity-based feature importances if supported by the
        ``estimator`` (when based on decision trees).

        Warning: impurity-based feature importances can be misleading for
        high cardinality features (many unique values). See
        :func:`sklearn.inspection.permutation_importance` as an alternative.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    AdaBoostRegressor : An AdaBoost regressor that begins by fitting a
        regressor on the original dataset and then fits additional copies of
        the regressor on the same dataset but where the weights of instances
        are adjusted according to the error of the current prediction.

    GradientBoostingClassifier : GB builds an additive model in a forward
        stage-wise fashion. Regression trees are fit on the negative gradient
        of the binomial or multinomial deviance loss function. Binary
        classification is a special case where only a single regression tree is
        induced.

    sklearn.tree.DecisionTreeClassifier : A non-parametric supervised learning
        method used for classification.
        Creates a model that predicts the value of a target variable by
        learning simple decision rules inferred from the data features.

    References
    ----------
    .. [1] Y. Freund, R. Schapire, "A Decision-Theoretic Generalization of
           on-Line Learning and an Application to Boosting", 1995.

    .. [2] :doi:`J. Zhu, H. Zou, S. Rosset, T. Hastie, "Multi-class adaboost."
           Statistics and its Interface 2.3 (2009): 349-360.
           <10.4310/SII.2009.v2.n3.a8>`

    Examples
    --------
    >>> from sklearn.ensemble import AdaBoostClassifier
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=1000, n_features=4,
    ...                            n_informative=2, n_redundant=0,
    ...                            random_state=0, shuffle=False)
    >>> clf = AdaBoostClassifier(n_estimators=100, random_state=0)
    >>> clf.fit(X, y)
    AdaBoostClassifier(n_estimators=100, random_state=0)
    >>> clf.predict([[0, 0, 0, 0]])
    array([1])
    >>> clf.score(X, y)
    0.96

    For a detailed example of using AdaBoost to fit a sequence of DecisionTrees
    as weaklearners, please refer to
    :ref:`sphx_glr_auto_examples_ensemble_plot_adaboost_multiclass.py`.

    For a detailed example of using AdaBoost to fit a non-linearly separable
    classification dataset composed of two Gaussian quantiles clusters, please
    refer to :ref:`sphx_glr_auto_examples_ensemble_plot_adaboost_twoclass.py`.
    """

    def __init__(self, estimator=None, *, n_estimators=50, learning_rate=1.0, random_state=None):
        super().__init__(estimator=estimator, n_estimators=n_estimators, learning_rate=learning_rate, random_state=random_state)

    def predict(self, X):
        """Predict classes for X.

        The predicted class of an input sample is computed as the weighted mean
        prediction of the classifiers in the ensemble.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted classes.
        """
        pred = self.decision_function(X)
        if self.n_classes_ == 2:
            return self.classes_.take(pred > 0, axis=0)
        return self.classes_.take(np.argmax(pred, axis=1), axis=0)

    def decision_function(self, X):
        """Compute the decision function of ``X``.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR.

        Returns
        -------
        score : ndarray of shape of (n_samples, k)
            The decision function of the input samples. The order of
            outputs is the same as that of the :term:`classes_` attribute.
            Binary classification is a special cases with ``k == 1``,
            otherwise ``k==n_classes``. For binary classification,
            values closer to -1 or 1 mean more like the first or second
            class in ``classes_``, respectively.
        """
        check_is_fitted(self)
        X = self._check_X(X)
        n_classes = self.n_classes_
        classes = self.classes_[:, np.newaxis]
        if n_classes == 1:
            return np.zeros_like(X, shape=(X.shape[0], 1))
        pred = sum((np.where((estimator.predict(X) == classes).T, w, -1 / (n_classes - 1) * w) for estimator, w in zip(self.estimators_, self.estimator_weights_)))
        pred /= self.estimator_weights_.sum()
        if n_classes == 2:
            pred[:, 0] *= -1
            return pred.sum(axis=1)
        return pred
