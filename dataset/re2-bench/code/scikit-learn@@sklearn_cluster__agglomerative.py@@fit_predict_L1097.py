from numbers import Integral, Real
from sklearn.base import (
    BaseEstimator,
    ClassNamePrefixFeaturesOutMixin,
    ClusterMixin,
    _fit_context,
)
from sklearn.metrics.pairwise import _VALID_METRICS, paired_distances
from sklearn.utils._param_validation import (
    HasMethods,
    Interval,
    StrOptions,
    validate_params,
)

class AgglomerativeClustering(ClusterMixin, BaseEstimator):
    """
    Agglomerative Clustering.

    Recursively merges pair of clusters of sample data; uses linkage distance.

    Read more in the :ref:`User Guide <hierarchical_clustering>`.

    Parameters
    ----------
    n_clusters : int or None, default=2
        The number of clusters to find. It must be ``None`` if
        ``distance_threshold`` is not ``None``.

    metric : str or callable, default="euclidean"
        Metric used to compute the linkage. Can be "euclidean", "l1", "l2",
        "manhattan", "cosine", or "precomputed". If linkage is "ward", only
        "euclidean" is accepted. If "precomputed", a distance matrix is needed
        as input for the fit method. If connectivity is None, linkage is
        "single" and affinity is not "precomputed" any valid pairwise distance
        metric can be assigned.

        For an example of agglomerative clustering with different metrics, see
        :ref:`sphx_glr_auto_examples_cluster_plot_agglomerative_clustering_metrics.py`.

        .. versionadded:: 1.2

    memory : str or object with the joblib.Memory interface, default=None
        Used to cache the output of the computation of the tree.
        By default, no caching is done. If a string is given, it is the
        path to the caching directory.

    connectivity : array-like, sparse matrix, or callable, default=None
        Connectivity matrix. Defines for each sample the neighboring
        samples following a given structure of the data.
        This can be a connectivity matrix itself or a callable that transforms
        the data into a connectivity matrix, such as derived from
        `kneighbors_graph`. Default is ``None``, i.e, the
        hierarchical clustering algorithm is unstructured.

        For an example of connectivity matrix using
        :class:`~sklearn.neighbors.kneighbors_graph`, see
        :ref:`sphx_glr_auto_examples_cluster_plot_ward_structured_vs_unstructured.py`.

    compute_full_tree : 'auto' or bool, default='auto'
        Stop early the construction of the tree at ``n_clusters``. This is
        useful to decrease computation time if the number of clusters is not
        small compared to the number of samples. This option is useful only
        when specifying a connectivity matrix. Note also that when varying the
        number of clusters and using caching, it may be advantageous to compute
        the full tree. It must be ``True`` if ``distance_threshold`` is not
        ``None``. By default `compute_full_tree` is "auto", which is equivalent
        to `True` when `distance_threshold` is not `None` or that `n_clusters`
        is inferior to the maximum between 100 or `0.02 * n_samples`.
        Otherwise, "auto" is equivalent to `False`.

    linkage : {'ward', 'complete', 'average', 'single'}, default='ward'
        Which linkage criterion to use. The linkage criterion determines which
        distance to use between sets of observation. The algorithm will merge
        the pairs of cluster that minimize this criterion.

        - 'ward' minimizes the variance of the clusters being merged.
        - 'average' uses the average of the distances of each observation of
          the two sets.
        - 'complete' or 'maximum' linkage uses the maximum distances between
          all observations of the two sets.
        - 'single' uses the minimum of the distances between all observations
          of the two sets.

        .. versionadded:: 0.20
            Added the 'single' option

        For examples comparing different `linkage` criteria, see
        :ref:`sphx_glr_auto_examples_cluster_plot_linkage_comparison.py`.

    distance_threshold : float, default=None
        The linkage distance threshold at or above which clusters will not be
        merged. If not ``None``, ``n_clusters`` must be ``None`` and
        ``compute_full_tree`` must be ``True``.

        .. versionadded:: 0.21

    compute_distances : bool, default=False
        Computes distances between clusters even if `distance_threshold` is not
        used. This can be used to make dendrogram visualization, but introduces
        a computational and memory overhead.

        .. versionadded:: 0.24

        For an example of dendrogram visualization, see
        :ref:`sphx_glr_auto_examples_cluster_plot_agglomerative_dendrogram.py`.

    Attributes
    ----------
    n_clusters_ : int
        The number of clusters found by the algorithm. If
        ``distance_threshold=None``, it will be equal to the given
        ``n_clusters``.

    labels_ : ndarray of shape (n_samples)
        Cluster labels for each point.

    n_leaves_ : int
        Number of leaves in the hierarchical tree.

    n_connected_components_ : int
        The estimated number of connected components in the graph.

        .. versionadded:: 0.21
            ``n_connected_components_`` was added to replace ``n_components_``.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    children_ : array-like of shape (n_samples-1, 2)
        The children of each non-leaf node. Values less than `n_samples`
        correspond to leaves of the tree which are the original samples.
        A node `i` greater than or equal to `n_samples` is a non-leaf
        node and has children `children_[i - n_samples]`. Alternatively
        at the i-th iteration, children[i][0] and children[i][1]
        are merged to form node `n_samples + i`.

    distances_ : array-like of shape (n_nodes-1,)
        Distances between nodes in the corresponding place in `children_`.
        Only computed if `distance_threshold` is used or `compute_distances`
        is set to `True`.

    See Also
    --------
    FeatureAgglomeration : Agglomerative clustering but for features instead of
        samples.
    ward_tree : Hierarchical clustering with ward linkage.

    Examples
    --------
    >>> from sklearn.cluster import AgglomerativeClustering
    >>> import numpy as np
    >>> X = np.array([[1, 2], [1, 4], [1, 0],
    ...               [4, 2], [4, 4], [4, 0]])
    >>> clustering = AgglomerativeClustering().fit(X)
    >>> clustering
    AgglomerativeClustering()
    >>> clustering.labels_
    array([1, 1, 1, 0, 0, 0])

    For a comparison of Agglomerative clustering with other clustering algorithms, see
    :ref:`sphx_glr_auto_examples_cluster_plot_cluster_comparison.py`
    """
    _parameter_constraints: dict = {'n_clusters': [Interval(Integral, 1, None, closed='left'), None], 'metric': [StrOptions(set(_VALID_METRICS) | {'precomputed'}), callable], 'memory': [str, HasMethods('cache'), None], 'connectivity': ['array-like', 'sparse matrix', callable, None], 'compute_full_tree': [StrOptions({'auto'}), 'boolean'], 'linkage': [StrOptions(set(_TREE_BUILDERS.keys()))], 'distance_threshold': [Interval(Real, 0, None, closed='left'), None], 'compute_distances': ['boolean']}

    def __init__(self, n_clusters=2, *, metric='euclidean', memory=None, connectivity=None, compute_full_tree='auto', linkage='ward', distance_threshold=None, compute_distances=False):
        self.n_clusters = n_clusters
        self.distance_threshold = distance_threshold
        self.memory = memory
        self.connectivity = connectivity
        self.compute_full_tree = compute_full_tree
        self.linkage = linkage
        self.metric = metric
        self.compute_distances = compute_distances

    def fit_predict(self, X, y=None):
        """Fit and return the result of each sample's clustering assignment.

        In addition to fitting, this method also return the result of the
        clustering assignment for each sample in the training set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or                 (n_samples, n_samples)
            Training instances to cluster, or distances between instances if
            ``affinity='precomputed'``.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Cluster labels.
        """
        return super().fit_predict(X, y)
