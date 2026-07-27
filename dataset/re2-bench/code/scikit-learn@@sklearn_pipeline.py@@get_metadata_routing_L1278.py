from itertools import chain, islice
from sklearn.utils._param_validation import HasMethods, Hidden
from sklearn.utils.metadata_routing import (
    MetadataRouter,
    MethodMapping,
    _raise_for_params,
    _routing_enabled,
    get_routing_for_object,
    process_routing,
)
from sklearn.utils.metaestimators import _BaseComposition, available_if

class Pipeline(_BaseComposition):
    """
    A sequence of data transformers with an optional final predictor.

    `Pipeline` allows you to sequentially apply a list of transformers to
    preprocess the data and, if desired, conclude the sequence with a final
    :term:`predictor` for predictive modeling.

    Intermediate steps of the pipeline must be transformers, that is, they
    must implement `fit` and `transform` methods.
    The final :term:`estimator` only needs to implement `fit`.
    The transformers in the pipeline can be cached using ``memory`` argument.

    The purpose of the pipeline is to assemble several steps that can be
    cross-validated together while setting different parameters. For this, it
    enables setting parameters of the various steps using their names and the
    parameter name separated by a `'__'`, as in the example below. A step's
    estimator may be replaced entirely by setting the parameter with its name
    to another estimator, or a transformer removed by setting it to
    `'passthrough'` or `None`.

    For an example use case of `Pipeline` combined with
    :class:`~sklearn.model_selection.GridSearchCV`, refer to
    :ref:`sphx_glr_auto_examples_compose_plot_compare_reduction.py`. The
    example :ref:`sphx_glr_auto_examples_compose_plot_digits_pipe.py` shows how
    to grid search on a pipeline using `'__'` as a separator in the parameter names.

    Read more in the :ref:`User Guide <pipeline>`.

    .. versionadded:: 0.5

    Parameters
    ----------
    steps : list of tuples
        List of (name of step, estimator) tuples that are to be chained in
        sequential order. To be compatible with the scikit-learn API, all steps
        must define `fit`. All non-last steps must also define `transform`. See
        :ref:`Combining Estimators <combining_estimators>` for more details.

    transform_input : list of str, default=None
        The names of the :term:`metadata` parameters that should be transformed by the
        pipeline before passing it to the step consuming it.

        This enables transforming some input arguments to ``fit`` (other than ``X``)
        to be transformed by the steps of the pipeline up to the step which requires
        them. Requirement is defined via :ref:`metadata routing <metadata_routing>`.
        For instance, this can be used to pass a validation set through the pipeline.

        You can only set this if metadata routing is enabled, which you
        can enable using ``sklearn.set_config(enable_metadata_routing=True)``.

        .. versionadded:: 1.6

    memory : str or object with the joblib.Memory interface, default=None
        Used to cache the fitted transformers of the pipeline. The last step
        will never be cached, even if it is a transformer. By default, no
        caching is performed. If a string is given, it is the path to the
        caching directory. Enabling caching triggers a clone of the transformers
        before fitting. Therefore, the transformer instance given to the
        pipeline cannot be inspected directly. Use the attribute ``named_steps``
        or ``steps`` to inspect estimators within the pipeline. Caching the
        transformers is advantageous when fitting is time consuming. See
        :ref:`sphx_glr_auto_examples_neighbors_plot_caching_nearest_neighbors.py`
        for an example on how to enable caching.

    verbose : bool, default=False
        If True, the time elapsed while fitting each step will be printed as it
        is completed.

    Attributes
    ----------
    named_steps : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.
        Read-only attribute to access any step parameter by user given name.
        Keys are step names and values are steps parameters.

    classes_ : ndarray of shape (n_classes,)
        The classes labels. Only exist if the last step of the pipeline is a
        classifier.

    n_features_in_ : int
        Number of features seen during :term:`fit`. Only defined if the
        underlying first estimator in `steps` exposes such an attribute
        when fit.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Only defined if the
        underlying estimator exposes such an attribute when fit.

        .. versionadded:: 1.0

    See Also
    --------
    make_pipeline : Convenience function for simplified pipeline construction.

    Examples
    --------
    >>> from sklearn.svm import SVC
    >>> from sklearn.preprocessing import StandardScaler
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.pipeline import Pipeline
    >>> X, y = make_classification(random_state=0)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y,
    ...                                                     random_state=0)
    >>> pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC())])
    >>> # The pipeline can be used as any other estimator
    >>> # and avoids leaking the test set into the train set
    >>> pipe.fit(X_train, y_train).score(X_test, y_test)
    0.88
    >>> # An estimator's parameter can be set using '__' syntax
    >>> pipe.set_params(svc__C=10).fit(X_train, y_train).score(X_test, y_test)
    0.76
    """
    _parameter_constraints: dict = {'steps': [list, Hidden(tuple)], 'transform_input': [list, None], 'memory': [None, str, HasMethods(['cache'])], 'verbose': ['boolean']}

    def __init__(self, steps, *, transform_input=None, memory=None, verbose=False):
        self.steps = steps
        self.transform_input = transform_input
        self.memory = memory
        self.verbose = verbose

    def _iter(self, with_final=True, filter_passthrough=True):
        """
        Generate (idx, (name, trans)) tuples from self.steps

        When filter_passthrough is True, 'passthrough' and None transformers
        are filtered out.
        """
        stop = len(self.steps)
        if not with_final:
            stop -= 1
        for idx, (name, trans) in enumerate(islice(self.steps, 0, stop)):
            if not filter_passthrough:
                yield (idx, name, trans)
            elif trans is not None and trans != 'passthrough':
                yield (idx, name, trans)

    def get_metadata_routing(self):
        """Get metadata routing of this object.

        Please check :ref:`User Guide <metadata_routing>` on how the routing
        mechanism works.

        Returns
        -------
        routing : MetadataRouter
            A :class:`~sklearn.utils.metadata_routing.MetadataRouter` encapsulating
            routing information.
        """
        router = MetadataRouter(owner=self)
        for _, name, trans in self._iter(with_final=False, filter_passthrough=True):
            method_mapping = MethodMapping()
            if hasattr(trans, 'fit_transform'):
                method_mapping.add(caller='fit', callee='fit_transform').add(caller='fit_transform', callee='fit_transform').add(caller='fit_predict', callee='fit_transform')
            else:
                method_mapping.add(caller='fit', callee='fit').add(caller='fit', callee='transform').add(caller='fit_transform', callee='fit').add(caller='fit_transform', callee='transform').add(caller='fit_predict', callee='fit').add(caller='fit_predict', callee='transform')
            method_mapping.add(caller='predict', callee='transform').add(caller='predict', callee='transform').add(caller='predict_proba', callee='transform').add(caller='decision_function', callee='transform').add(caller='predict_log_proba', callee='transform').add(caller='transform', callee='transform').add(caller='inverse_transform', callee='inverse_transform').add(caller='score', callee='transform')
            router.add(method_mapping=method_mapping, **{name: trans})
        final_name, final_est = self.steps[-1]
        if final_est is None or final_est == 'passthrough':
            return router
        method_mapping = MethodMapping()
        if hasattr(final_est, 'fit_transform'):
            method_mapping.add(caller='fit_transform', callee='fit_transform')
        else:
            method_mapping.add(caller='fit', callee='fit').add(caller='fit', callee='transform')
        method_mapping.add(caller='fit', callee='fit').add(caller='predict', callee='predict').add(caller='fit_predict', callee='fit_predict').add(caller='predict_proba', callee='predict_proba').add(caller='decision_function', callee='decision_function').add(caller='predict_log_proba', callee='predict_log_proba').add(caller='transform', callee='transform').add(caller='inverse_transform', callee='inverse_transform').add(caller='score', callee='score')
        router.add(method_mapping=method_mapping, **{final_name: final_est})
        return router
