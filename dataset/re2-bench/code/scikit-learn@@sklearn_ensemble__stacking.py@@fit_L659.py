import numpy as np
from sklearn.base import (
    ClassifierMixin,
    RegressorMixin,
    TransformerMixin,
    _fit_context,
    clone,
    is_classifier,
    is_regressor,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.utils._param_validation import HasMethods, StrOptions
from sklearn.utils.metadata_routing import (
    MetadataRouter,
    MethodMapping,
    _raise_for_params,
    _routing_enabled,
    process_routing,
)
from sklearn.utils.multiclass import check_classification_targets, type_of_target

class StackingClassifier(ClassifierMixin, _BaseStacking):
    """Stack of estimators with a final classifier.

    Stacked generalization consists in stacking the output of individual
    estimator and use a classifier to compute the final prediction. Stacking
    allows to use the strength of each individual estimator by using their
    output as input of a final estimator.

    Note that `estimators_` are fitted on the full `X` while `final_estimator_`
    is trained using cross-validated predictions of the base estimators using
    `cross_val_predict`.

    Read more in the :ref:`User Guide <stacking>`.

    .. versionadded:: 0.22

    Parameters
    ----------
    estimators : list of (str, estimator)
        Base estimators which will be stacked together. Each element of the
        list is defined as a tuple of string (i.e. name) and an estimator
        instance. An estimator can be set to 'drop' using `set_params`.

        The type of estimator is generally expected to be a classifier.
        However, one can pass a regressor for some use case (e.g. ordinal
        regression).

    final_estimator : estimator, default=None
        A classifier which will be used to combine the base estimators.
        The default classifier is a
        :class:`~sklearn.linear_model.LogisticRegression`.

    cv : int, cross-validation generator, iterable, or "prefit", default=None
        Determines the cross-validation splitting strategy used in
        `cross_val_predict` to train `final_estimator`. Possible inputs for
        cv are:

        * None, to use the default 5-fold cross validation,
        * integer, to specify the number of folds in a (Stratified) KFold,
        * An object to be used as a cross-validation generator,
        * An iterable yielding train, test splits,
        * `"prefit"`, to assume the `estimators` are prefit. In this case, the
          estimators will not be refitted.

        For integer/None inputs, if the estimator is a classifier and y is
        either binary or multiclass,
        :class:`~sklearn.model_selection.StratifiedKFold` is used.
        In all other cases, :class:`~sklearn.model_selection.KFold` is used.
        These splitters are instantiated with `shuffle=False` so the splits
        will be the same across calls.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

        If "prefit" is passed, it is assumed that all `estimators` have
        been fitted already. The `final_estimator_` is trained on the `estimators`
        predictions on the full training set and are **not** cross validated
        predictions. Please note that if the models have been trained on the same
        data to train the stacking model, there is a very high risk of overfitting.

        .. versionadded:: 1.1
            The 'prefit' option was added in 1.1

        .. note::
           A larger number of split will provide no benefits if the number
           of training samples is large enough. Indeed, the training time
           will increase. ``cv`` is not used for model evaluation but for
           prediction.

    stack_method : {'auto', 'predict_proba', 'decision_function', 'predict'},             default='auto'
        Methods called for each base estimator. It can be:

        * if 'auto', it will try to invoke, for each estimator,
          `'predict_proba'`, `'decision_function'` or `'predict'` in that
          order.
        * otherwise, one of `'predict_proba'`, `'decision_function'` or
          `'predict'`. If the method is not implemented by the estimator, it
          will raise an error.

    n_jobs : int, default=None
        The number of jobs to run in parallel for `fit` of all `estimators`.
        `None` means 1 unless in a `joblib.parallel_backend` context. -1 means
        using all processors. See :term:`Glossary <n_jobs>` for more details.

    passthrough : bool, default=False
        When False, only the predictions of estimators will be used as
        training data for `final_estimator`. When True, the
        `final_estimator` is trained on the predictions as well as the
        original training data.

    verbose : int, default=0
        Verbosity level.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,) or list of ndarray if `y`         is of type `"multilabel-indicator"`.
        Class labels.

    estimators_ : list of estimators
        The elements of the `estimators` parameter, having been fitted on the
        training data. If an estimator has been set to `'drop'`, it
        will not appear in `estimators_`. When `cv="prefit"`, `estimators_`
        is set to `estimators` and is not fitted again.

    named_estimators_ : :class:`~sklearn.utils.Bunch`
        Attribute to access any fitted sub-estimators by name.

    n_features_in_ : int
        Number of features seen during :term:`fit`. Only defined if the
        underlying estimator exposes such an attribute when fit.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Only defined if the
        underlying estimators expose such an attribute when fit.

        .. versionadded:: 1.0

    final_estimator_ : estimator
        The classifier fit on the output of `estimators_` and responsible for
        final predictions.

    stack_method_ : list of str
        The method used by each base estimator.

    See Also
    --------
    StackingRegressor : Stack of estimators with a final regressor.

    Notes
    -----
    When `predict_proba` is used by each estimator (i.e. most of the time for
    `stack_method='auto'` or specifically for `stack_method='predict_proba'`),
    the first column predicted by each estimator will be dropped in the case
    of a binary classification problem. Indeed, both feature will be perfectly
    collinear.

    In some cases (e.g. ordinal regression), one can pass regressors as the
    first layer of the :class:`StackingClassifier`. However, note that `y` will
    be internally encoded in a numerically increasing order or lexicographic
    order. If this ordering is not adequate, one should manually numerically
    encode the classes in the desired order.

    References
    ----------
    .. [1] Wolpert, David H. "Stacked generalization." Neural networks 5.2
       (1992): 241-259.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.svm import LinearSVC
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.preprocessing import StandardScaler
    >>> from sklearn.pipeline import make_pipeline
    >>> from sklearn.ensemble import StackingClassifier
    >>> X, y = load_iris(return_X_y=True)
    >>> estimators = [
    ...     ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
    ...     ('svr', make_pipeline(StandardScaler(),
    ...                           LinearSVC(random_state=42)))
    ... ]
    >>> clf = StackingClassifier(
    ...     estimators=estimators, final_estimator=LogisticRegression()
    ... )
    >>> from sklearn.model_selection import train_test_split
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     X, y, stratify=y, random_state=42
    ... )
    >>> clf.fit(X_train, y_train).score(X_test, y_test)
    0.9...
    """
    _parameter_constraints: dict = {**_BaseStacking._parameter_constraints, 'stack_method': [StrOptions({'auto', 'predict_proba', 'decision_function', 'predict'})]}

    def __init__(self, estimators, final_estimator=None, *, cv=None, stack_method='auto', n_jobs=None, passthrough=False, verbose=0):
        super().__init__(estimators=estimators, final_estimator=final_estimator, cv=cv, stack_method=stack_method, n_jobs=n_jobs, passthrough=passthrough, verbose=verbose)

    def fit(self, X, y, **fit_params):
        """Fit the estimators.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            Target values. Note that `y` will be internally encoded in
            numerically increasing order or lexicographic order. If the order
            matter (e.g. for ordinal regression), one should numerically encode
            the target `y` before calling :term:`fit`.

        **fit_params : dict
            Parameters to pass to the underlying estimators.

            .. versionadded:: 1.6

                Only available if `enable_metadata_routing=True`, which can be
                set by using ``sklearn.set_config(enable_metadata_routing=True)``.
                See :ref:`Metadata Routing User Guide <metadata_routing>` for
                more details.

        Returns
        -------
        self : object
            Returns a fitted instance of estimator.
        """
        _raise_for_params(fit_params, self, 'fit', allow=['sample_weight'])
        check_classification_targets(y)
        if type_of_target(y) == 'multilabel-indicator':
            self._label_encoder = [LabelEncoder().fit(yk) for yk in y.T]
            self.classes_ = [le.classes_ for le in self._label_encoder]
            y_encoded = np.array([self._label_encoder[target_idx].transform(target) for target_idx, target in enumerate(y.T)]).T
        else:
            self._label_encoder = LabelEncoder().fit(y)
            self.classes_ = self._label_encoder.classes_
            y_encoded = self._label_encoder.transform(y)
        return super().fit(X, y_encoded, **fit_params)
