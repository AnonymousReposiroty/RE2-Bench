from numbers import Integral, Real
from sklearn.base import (
    ClassifierMixin,
    MultiOutputMixin,
    RegressorMixin,
    TransformerMixin,
    _fit_context,
    is_classifier,
)
from sklearn.tree import (
    BaseDecisionTree,
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    ExtraTreeClassifier,
    ExtraTreeRegressor,
)
from sklearn.utils._param_validation import Interval, RealNotInt, StrOptions
from sklearn.utils.validation import (
    _check_feature_names_in,
    _check_sample_weight,
    _num_samples,
    check_is_fitted,
    validate_data,
)

class RandomTreesEmbedding(TransformerMixin, BaseForest):
    """
    An ensemble of totally random trees.

    An unsupervised transformation of a dataset to a high-dimensional
    sparse representation. A datapoint is coded according to which leaf of
    each tree it is sorted into. Using a one-hot encoding of the leaves,
    this leads to a binary coding with as many ones as there are trees in
    the forest.

    The dimensionality of the resulting representation is
    ``n_out <= n_estimators * max_leaf_nodes``. If ``max_leaf_nodes == None``,
    the number of leaf nodes is at most ``n_estimators * 2 ** max_depth``.

    For an example of applying Random Trees Embedding to non-linear
    classification, see
    :ref:`sphx_glr_auto_examples_ensemble_plot_random_forest_embedding.py`.

    Read more in the :ref:`User Guide <random_trees_embedding>`.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees in the forest.

        .. versionchanged:: 0.22
           The default value of ``n_estimators`` changed from 10 to 100
           in 0.22.

    max_depth : int, default=5
        The maximum depth of each tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` is the minimum
          number of samples for each split.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_samples_leaf : int or float, default=1
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` is the minimum
          number of samples for each node.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_weight_fraction_leaf : float, default=0.0
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_leaf_nodes : int, default=None
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    min_impurity_decrease : float, default=0.0
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.

        The weighted impurity decrease equation is the following::

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.

        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.

        .. versionadded:: 0.19

    sparse_output : bool, default=True
        Whether or not to return a sparse CSR matrix, as default behavior,
        or to return a dense array compatible with dense pipeline operators.

    n_jobs : int, default=None
        The number of jobs to run in parallel. :meth:`fit`, :meth:`transform`,
        :meth:`decision_path` and :meth:`apply` are all parallelized over the
        trees. ``None`` means 1 unless in a :obj:`joblib.parallel_backend`
        context. ``-1`` means using all processors. See :term:`Glossary
        <n_jobs>` for more details.

    random_state : int, RandomState instance or None, default=None
        Controls the generation of the random `y` used to fit the trees
        and the draw of the splits for each feature at the trees' nodes.
        See :term:`Glossary <random_state>` for details.

    verbose : int, default=0
        Controls the verbosity when fitting and predicting.

    warm_start : bool, default=False
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit a whole
        new forest. See :term:`Glossary <warm_start>` and
        :ref:`tree_ensemble_warm_start` for details.

    Attributes
    ----------
    estimator_ : :class:`~sklearn.tree.ExtraTreeRegressor` instance
        The child estimator template used to create the collection of fitted
        sub-estimators.

        .. versionadded:: 1.2
           `base_estimator_` was renamed to `estimator_`.

    estimators_ : list of :class:`~sklearn.tree.ExtraTreeRegressor` instances
        The collection of fitted sub-estimators.

    feature_importances_ : ndarray of shape (n_features,)
        The feature importances (the higher, the more important the feature).

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    n_outputs_ : int
        The number of outputs when ``fit`` is performed.

    one_hot_encoder_ : OneHotEncoder instance
        One-hot encoder used to create the sparse embedding.

    estimators_samples_ : list of arrays
        The subset of drawn samples (i.e., the in-bag samples) for each base
        estimator. Each subset is defined by an array of the indices selected.

        .. versionadded:: 1.4

    See Also
    --------
    ExtraTreesClassifier : An extra-trees classifier.
    ExtraTreesRegressor : An extra-trees regressor.
    RandomForestClassifier : A random forest classifier.
    RandomForestRegressor : A random forest regressor.
    sklearn.tree.ExtraTreeClassifier: An extremely randomized
        tree classifier.
    sklearn.tree.ExtraTreeRegressor : An extremely randomized
        tree regressor.

    References
    ----------
    .. [1] P. Geurts, D. Ernst., and L. Wehenkel, "Extremely randomized trees",
           Machine Learning, 63(1), 3-42, 2006.
    .. [2] Moosmann, F. and Triggs, B. and Jurie, F.  "Fast discriminative
           visual codebooks using randomized clustering forests"
           NIPS 2007.

    Examples
    --------
    >>> from sklearn.ensemble import RandomTreesEmbedding
    >>> X = [[0,0], [1,0], [0,1], [-1,0], [0,-1]]
    >>> random_trees = RandomTreesEmbedding(
    ...    n_estimators=5, random_state=0, max_depth=1).fit(X)
    >>> X_sparse_embedding = random_trees.transform(X)
    >>> X_sparse_embedding.toarray()
    array([[0., 1., 1., 0., 1., 0., 0., 1., 1., 0.],
           [0., 1., 1., 0., 1., 0., 0., 1., 1., 0.],
           [0., 1., 0., 1., 0., 1., 0., 1., 0., 1.],
           [1., 0., 1., 0., 1., 0., 1., 0., 1., 0.],
           [0., 1., 1., 0., 1., 0., 0., 1., 1., 0.]])
    """
    _parameter_constraints: dict = {'n_estimators': [Interval(Integral, 1, None, closed='left')], 'n_jobs': [Integral, None], 'verbose': ['verbose'], 'warm_start': ['boolean'], **BaseDecisionTree._parameter_constraints, 'sparse_output': ['boolean']}
    for param in ('max_features', 'ccp_alpha', 'splitter', 'monotonic_cst'):
        _parameter_constraints.pop(param)
    criterion = 'squared_error'
    max_features = 1

    def __init__(self, n_estimators=100, *, max_depth=5, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_leaf_nodes=None, min_impurity_decrease=0.0, sparse_output=True, n_jobs=None, random_state=None, verbose=0, warm_start=False):
        super().__init__(estimator=ExtraTreeRegressor(), n_estimators=n_estimators, estimator_params=('criterion', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'min_weight_fraction_leaf', 'max_features', 'max_leaf_nodes', 'min_impurity_decrease', 'random_state'), bootstrap=False, oob_score=False, n_jobs=n_jobs, random_state=random_state, verbose=verbose, warm_start=warm_start, max_samples=None)
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.sparse_output = sparse_output

    def transform(self, X):
        """
        Transform dataset.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Input data to be transformed. Use ``dtype=np.float32`` for maximum
            efficiency. Sparse matrices are also supported, use sparse
            ``csr_matrix`` for maximum efficiency.

        Returns
        -------
        X_transformed : sparse matrix of shape (n_samples, n_out)
            Transformed dataset.
        """
        check_is_fitted(self)
        return self.one_hot_encoder_.transform(self.apply(X))
