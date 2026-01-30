import numpy as np
from scipy import sparse
from sklearn.base import TransformerMixin, _fit_context, clone
from sklearn.preprocessing import FunctionTransformer
from sklearn.utils import Bunch
from sklearn.utils._set_output import _get_container_adapter, _safe_set_output
from sklearn.utils.metadata_routing import (
    MetadataRouter,
    MethodMapping,
    _raise_for_params,
    _routing_enabled,
    get_routing_for_object,
    process_routing,
)
from sklearn.utils.metaestimators import _BaseComposition, available_if
from sklearn.utils.parallel import Parallel, delayed

class FeatureUnion(TransformerMixin, _BaseComposition):
    """Concatenates results of multiple transformer objects.

    This estimator applies a list of transformer objects in parallel to the
    input data, then concatenates the results. This is useful to combine
    several feature extraction mechanisms into a single transformer.

    Parameters of the transformers may be set using its name and the parameter
    name separated by a '__'. A transformer may be replaced entirely by
    setting the parameter with its name to another transformer, removed by
    setting to 'drop' or disabled by setting to 'passthrough' (features are
    passed without transformation).

    Read more in the :ref:`User Guide <feature_union>`.

    .. versionadded:: 0.13

    Parameters
    ----------
    transformer_list : list of (str, transformer) tuples
        List of transformer objects to be applied to the data. The first
        half of each tuple is the name of the transformer. The transformer can
        be 'drop' for it to be ignored or can be 'passthrough' for features to
        be passed unchanged.

        .. versionadded:: 1.1
           Added the option `"passthrough"`.

        .. versionchanged:: 0.22
           Deprecated `None` as a transformer in favor of 'drop'.

    n_jobs : int, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

        .. versionchanged:: v0.20
           `n_jobs` default changed from 1 to None

    transformer_weights : dict, default=None
        Multiplicative weights for features per transformer.
        Keys are transformer names, values the weights.
        Raises ValueError if key not present in ``transformer_list``.

    verbose : bool, default=False
        If True, the time elapsed while fitting each transformer will be
        printed as it is completed.

    verbose_feature_names_out : bool, default=True
        If True, :meth:`get_feature_names_out` will prefix all feature names
        with the name of the transformer that generated that feature.
        If False, :meth:`get_feature_names_out` will not prefix any feature
        names and will error if feature names are not unique.

        .. versionadded:: 1.5

    Attributes
    ----------
    named_transformers : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.
        Read-only attribute to access any transformer parameter by user
        given name. Keys are transformer names and values are
        transformer parameters.

        .. versionadded:: 1.2

    n_features_in_ : int
        Number of features seen during :term:`fit`. Only defined if the
        underlying first transformer in `transformer_list` exposes such an
        attribute when fit.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when
        `X` has feature names that are all strings.

        .. versionadded:: 1.3

    See Also
    --------
    make_union : Convenience function for simplified feature union
        construction.

    Examples
    --------
    >>> from sklearn.pipeline import FeatureUnion
    >>> from sklearn.decomposition import PCA, TruncatedSVD
    >>> union = FeatureUnion([("pca", PCA(n_components=1)),
    ...                       ("svd", TruncatedSVD(n_components=2))])
    >>> X = [[0., 1., 3], [2., 2., 5]]
    >>> union.fit_transform(X)
    array([[-1.5       ,  3.04, -0.872],
           [ 1.5       ,  5.72,  0.463]])
    >>> # An estimator's parameter can be set using '__' syntax
    >>> union.set_params(svd__n_components=1).fit_transform(X)
    array([[-1.5       ,  3.04],
           [ 1.5       ,  5.72]])

    For a more detailed example of usage, see
    :ref:`sphx_glr_auto_examples_compose_plot_feature_union.py`.
    """

    def __init__(self, transformer_list, *, n_jobs=None, transformer_weights=None, verbose=False, verbose_feature_names_out=True):
        self.transformer_list = transformer_list
        self.n_jobs = n_jobs
        self.transformer_weights = transformer_weights
        self.verbose = verbose
        self.verbose_feature_names_out = verbose_feature_names_out

    def _validate_transformers(self):
        names, transformers = zip(*self.transformer_list)
        self._validate_names(names)
        for t in transformers:
            if t in ('drop', 'passthrough'):
                continue
            if not (hasattr(t, 'fit') or hasattr(t, 'fit_transform')) or not hasattr(t, 'transform'):
                raise TypeError("All estimators should implement fit and transform. '%s' (type %s) doesn't" % (t, type(t)))

    def _validate_transformer_weights(self):
        if not self.transformer_weights:
            return
        transformer_names = set((name for name, _ in self.transformer_list))
        for name in self.transformer_weights:
            if name not in transformer_names:
                raise ValueError(f'Attempting to weight transformer "{name}", but it is not present in transformer_list.')

    def _iter(self):
        """
        Generate (name, trans, weight) tuples excluding None and
        'drop' transformers.
        """
        get_weight = (self.transformer_weights or {}).get
        for name, trans in self.transformer_list:
            if trans == 'drop':
                continue
            if trans == 'passthrough':
                trans = FunctionTransformer(feature_names_out='one-to-one')
            yield (name, trans, get_weight(name))

    def fit_transform(self, X, y=None, **params):
        """Fit all transformers, transform the data and concatenate results.

        Parameters
        ----------
        X : iterable or array-like, depending on transformers
            Input data to be transformed.

        y : array-like of shape (n_samples, n_outputs), default=None
            Targets for supervised learning.

        **params : dict, default=None
            - If `enable_metadata_routing=False` (default):
              Parameters directly passed to the `fit` methods of the
              sub-transformers.

            - If `enable_metadata_routing=True`:
              Parameters safely routed to the `fit` methods of the
              sub-transformers. See :ref:`Metadata Routing User Guide
              <metadata_routing>` for more details.

            .. versionchanged:: 1.5
                `**params` can now be routed via metadata routing API.

        Returns
        -------
        X_t : array-like or sparse matrix of                 shape (n_samples, sum_n_components)
            The `hstack` of results of transformers. `sum_n_components` is the
            sum of `n_components` (output dimension) over transformers.
        """
        if _routing_enabled():
            routed_params = process_routing(self, 'fit_transform', **params)
        else:
            routed_params = Bunch()
            for name, obj in self.transformer_list:
                if hasattr(obj, 'fit_transform'):
                    routed_params[name] = Bunch(fit_transform={})
                    routed_params[name].fit_transform = params
                else:
                    routed_params[name] = Bunch(fit={})
                    routed_params[name] = Bunch(transform={})
                    routed_params[name].fit = params
        results = self._parallel_func(X, y, _fit_transform_one, routed_params)
        if not results:
            return np.zeros((X.shape[0], 0))
        Xs, transformers = zip(*results)
        self._update_transformer_list(transformers)
        return self._hstack(Xs)

    def _log_message(self, name, idx, total):
        if not self.verbose:
            return None
        return '(step %d of %d) Processing %s' % (idx, total, name)

    def _parallel_func(self, X, y, func, routed_params):
        """Runs func in parallel on X and y"""
        self.transformer_list = list(self.transformer_list)
        self._validate_transformers()
        self._validate_transformer_weights()
        transformers = list(self._iter())
        return Parallel(n_jobs=self.n_jobs)((delayed(func)(transformer, X, y, weight, message_clsname='FeatureUnion', message=self._log_message(name, idx, len(transformers)), params=routed_params[name]) for idx, (name, transformer, weight) in enumerate(transformers, 1)))

    def _hstack(self, Xs):
        for X, (name, _) in zip(Xs, self.transformer_list):
            if hasattr(X, 'shape') and len(X.shape) != 2:
                raise ValueError(f"Transformer '{name}' returned an array or dataframe with {len(X.shape)} dimensions, but expected 2 dimensions (n_samples, n_features).")
        adapter = _get_container_adapter('transform', self)
        if adapter and all((adapter.is_supported_container(X) for X in Xs)):
            return adapter.hstack(Xs)
        if any((sparse.issparse(f) for f in Xs)):
            return sparse.hstack(Xs).tocsr()
        return np.hstack(Xs)

    def _update_transformer_list(self, transformers):
        transformers = iter(transformers)
        self.transformer_list[:] = [(name, old if old == 'drop' else next(transformers)) for name, old in self.transformer_list]
