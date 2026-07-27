import numpy as np
from sklearn.base import ClassifierMixin, RegressorMixin, _fit_context
from sklearn.ensemble._base import BaseEnsemble, _partition_estimators
from sklearn.utils import Bunch, _safe_indexing, check_random_state, column_or_1d
from sklearn.utils.metadata_routing import (
    MetadataRouter,
    MethodMapping,
    _raise_for_params,
    _routing_enabled,
    get_routing_for_object,
    process_routing,
)
from sklearn.utils.parallel import Parallel, delayed
from sklearn.utils.validation import (
    _check_method_params,
    _check_sample_weight,
    _estimator_has,
    check_is_fitted,
    has_fit_parameter,
    validate_data,
)

class BaggingClassifier(ClassifierMixin, BaseBagging):
    """A Bagging classifier.

    A Bagging classifier is an ensemble meta-estimator that fits base
    classifiers each on random subsets of the original dataset and then
    aggregate their individual predictions (either by voting or by averaging)
    to form a final prediction. Such a meta-estimator can typically be used as
    a way to reduce the variance of a black-box estimator (e.g., a decision
    tree), by introducing randomization into its construction procedure and
    then making an ensemble out of it.

    This algorithm encompasses several works from the literature. When random
    subsets of the dataset are drawn as random subsets of the samples, then
    this algorithm is known as Pasting [1]_. If samples are drawn with
    replacement, then the method is known as Bagging [2]_. When random subsets
    of the dataset are drawn as random subsets of the features, then the method
    is known as Random Subspaces [3]_. Finally, when base estimators are built
    on subsets of both samples and features, then the method is known as
    Random Patches [4]_.

    Read more in the :ref:`User Guide <bagging>`.

    .. versionadded:: 0.15

    Parameters
    ----------
    estimator : object, default=None
        The base estimator to fit on random subsets of the dataset.
        If None, then the base estimator is a
        :class:`~sklearn.tree.DecisionTreeClassifier`.

        .. versionadded:: 1.2
           `base_estimator` was renamed to `estimator`.

    n_estimators : int, default=10
        The number of base estimators in the ensemble.

    max_samples : int or float, default=1.0
        The number of samples to draw from X to train each base estimator (with
        replacement by default, see `bootstrap` for more details).

        - If int, then draw `max_samples` samples.
        - If float, then draw `max_samples * X.shape[0]` unweighted samples
          or `max_samples * sample_weight.sum()` weighted samples.

    max_features : int or float, default=1.0
        The number of features to draw from X to train each base estimator (
        without replacement by default, see `bootstrap_features` for more
        details).

        - If int, then draw `max_features` features.
        - If float, then draw `max(1, int(max_features * n_features_in_))` features.

    bootstrap : bool, default=True
        Whether samples are drawn with replacement. If False, sampling without
        replacement is performed. If fitting with `sample_weight`, it is
        strongly recommended to choose True, as only drawing with replacement
        will ensure the expected frequency semantics of `sample_weight`.

    bootstrap_features : bool, default=False
        Whether features are drawn with replacement.

    oob_score : bool, default=False
        Whether to use out-of-bag samples to estimate
        the generalization error. Only available if bootstrap=True.

    warm_start : bool, default=False
        When set to True, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit
        a whole new ensemble. See :term:`the Glossary <warm_start>`.

        .. versionadded:: 0.17
           *warm_start* constructor parameter.

    n_jobs : int, default=None
        The number of jobs to run in parallel for both :meth:`fit` and
        :meth:`predict`. ``None`` means 1 unless in a
        :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors. See :term:`Glossary <n_jobs>` for more details.

    random_state : int, RandomState instance or None, default=None
        Controls the random resampling of the original dataset
        (sample wise and feature wise).
        If the base estimator accepts a `random_state` attribute, a different
        seed is generated for each instance in the ensemble.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    verbose : int, default=0
        Controls the verbosity when fitting and predicting.

    Attributes
    ----------
    estimator_ : estimator
        The base estimator from which the ensemble is grown.

        .. versionadded:: 1.2
           `base_estimator_` was renamed to `estimator_`.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    estimators_ : list of estimators
        The collection of fitted base estimators.

    estimators_samples_ : list of arrays
        The subset of drawn samples (i.e., the in-bag samples) for each base
        estimator. Each subset is defined by an array of the indices selected.

    estimators_features_ : list of arrays
        The subset of drawn features for each base estimator.

    classes_ : ndarray of shape (n_classes,)
        The classes labels.

    n_classes_ : int or list
        The number of classes.

    oob_score_ : float
        Score of the training dataset obtained using an out-of-bag estimate.
        This attribute exists only when ``oob_score`` is True.

    oob_decision_function_ : ndarray of shape (n_samples, n_classes)
        Decision function computed with out-of-bag estimate on the training
        set. If n_estimators is small it might be possible that a data point
        was never left out during the bootstrap. In this case,
        `oob_decision_function_` might contain NaN. This attribute exists
        only when ``oob_score`` is True.

    See Also
    --------
    BaggingRegressor : A Bagging regressor.

    References
    ----------

    .. [1] L. Breiman, "Pasting small votes for classification in large
           databases and on-line", Machine Learning, 36(1), 85-103, 1999.

    .. [2] L. Breiman, "Bagging predictors", Machine Learning, 24(2), 123-140,
           1996.

    .. [3] T. Ho, "The random subspace method for constructing decision
           forests", Pattern Analysis and Machine Intelligence, 20(8), 832-844,
           1998.

    .. [4] G. Louppe and P. Geurts, "Ensembles on Random Patches", Machine
           Learning and Knowledge Discovery in Databases, 346-361, 2012.

    Examples
    --------
    >>> from sklearn.svm import SVC
    >>> from sklearn.ensemble import BaggingClassifier
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=100, n_features=4,
    ...                            n_informative=2, n_redundant=0,
    ...                            random_state=0, shuffle=False)
    >>> clf = BaggingClassifier(estimator=SVC(),
    ...                         n_estimators=10, random_state=0).fit(X, y)
    >>> clf.predict([[0, 0, 0, 0]])
    array([1])
    """

    def __init__(self, estimator=None, n_estimators=10, *, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=None, random_state=None, verbose=0):
        super().__init__(estimator=estimator, n_estimators=n_estimators, max_samples=max_samples, max_features=max_features, bootstrap=bootstrap, bootstrap_features=bootstrap_features, oob_score=oob_score, warm_start=warm_start, n_jobs=n_jobs, random_state=random_state, verbose=verbose)

    def predict_proba(self, X, **params):
        """Predict class probabilities for X.

        The predicted class probabilities of an input sample is computed as
        the mean predicted class probabilities of the base estimators in the
        ensemble. If base estimators do not implement a ``predict_proba``
        method, then it resorts to voting and the predicted class probabilities
        of an input sample represents the proportion of estimators predicting
        each class.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.

        **params : dict
            Parameters routed to the `predict_proba` (if available) or the `predict`
            method (otherwise) of the sub-estimators via the metadata routing API.

            .. versionadded:: 1.7

                Only available if
                `sklearn.set_config(enable_metadata_routing=True)` is set. See
                :ref:`Metadata Routing User Guide <metadata_routing>` for more
                details.

        Returns
        -------
        p : ndarray of shape (n_samples, n_classes)
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_`.
        """
        _raise_for_params(params, self, 'predict_proba')
        check_is_fitted(self)
        X = validate_data(self, X, accept_sparse=['csr', 'csc'], dtype=None, ensure_all_finite=False, reset=False)
        if _routing_enabled():
            routed_params = process_routing(self, 'predict_proba', **params)
        else:
            routed_params = Bunch()
            routed_params.estimator = Bunch(predict_proba=Bunch())
        n_jobs, _, starts = _partition_estimators(self.n_estimators, self.n_jobs)
        all_proba = Parallel(n_jobs=n_jobs, verbose=self.verbose, **self._parallel_args())((delayed(_parallel_predict_proba)(self.estimators_[starts[i]:starts[i + 1]], self.estimators_features_[starts[i]:starts[i + 1]], X, self.n_classes_, predict_params=routed_params.estimator.get('predict', None), predict_proba_params=routed_params.estimator.get('predict_proba', None)) for i in range(n_jobs)))
        proba = sum(all_proba) / self.n_estimators
        return proba

    def predict_log_proba(self, X, **params):
        """Predict class log-probabilities for X.

        The predicted class log-probabilities of an input sample is computed as
        the log of the mean predicted class probabilities of the base
        estimators in the ensemble.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.

        **params : dict
            Parameters routed to the `predict_log_proba`, the `predict_proba` or the
            `proba` method of the sub-estimators via the metadata routing API. The
            routing is tried in the mentioned order depending on whether this method is
            available on the sub-estimator.

            .. versionadded:: 1.7

                Only available if
                `sklearn.set_config(enable_metadata_routing=True)` is set. See
                :ref:`Metadata Routing User Guide <metadata_routing>` for more
                details.

        Returns
        -------
        p : ndarray of shape (n_samples, n_classes)
            The class log-probabilities of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_`.
        """
        _raise_for_params(params, self, 'predict_log_proba')
        check_is_fitted(self)
        if hasattr(self.estimator_, 'predict_log_proba'):
            X = validate_data(self, X, accept_sparse=['csr', 'csc'], dtype=None, ensure_all_finite=False, reset=False)
            if _routing_enabled():
                routed_params = process_routing(self, 'predict_log_proba', **params)
            else:
                routed_params = Bunch()
                routed_params.estimator = Bunch(predict_log_proba=Bunch())
            n_jobs, _, starts = _partition_estimators(self.n_estimators, self.n_jobs)
            all_log_proba = Parallel(n_jobs=n_jobs, verbose=self.verbose)((delayed(_parallel_predict_log_proba)(self.estimators_[starts[i]:starts[i + 1]], self.estimators_features_[starts[i]:starts[i + 1]], X, self.n_classes_, params=routed_params.estimator.predict_log_proba) for i in range(n_jobs)))
            log_proba = all_log_proba[0]
            for j in range(1, len(all_log_proba)):
                log_proba = np.logaddexp(log_proba, all_log_proba[j])
            log_proba -= np.log(self.n_estimators)
        else:
            log_proba = np.log(self.predict_proba(X, **params))
        return log_proba
