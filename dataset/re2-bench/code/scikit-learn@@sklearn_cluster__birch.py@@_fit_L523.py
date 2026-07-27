import warnings
from numbers import Integral, Real
import numpy as np
from scipy import sparse
from sklearn._config import config_context
from sklearn.base import (
    BaseEstimator,
    ClassNamePrefixFeaturesOutMixin,
    ClusterMixin,
    TransformerMixin,
    _fit_context,
)
from sklearn.cluster import AgglomerativeClustering
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import pairwise_distances_argmin
from sklearn.utils._param_validation import Interval
from sklearn.utils.extmath import row_norms
from sklearn.utils.validation import check_is_fitted, validate_data

class Birch(ClassNamePrefixFeaturesOutMixin, ClusterMixin, TransformerMixin, BaseEstimator):
    """Implements the BIRCH clustering algorithm.

    It is a memory-efficient, online-learning algorithm provided as an
    alternative to :class:`MiniBatchKMeans`. It constructs a tree
    data structure with the cluster centroids being read off the leaf.
    These can be either the final cluster centroids or can be provided as input
    to another clustering algorithm such as :class:`AgglomerativeClustering`.

    Read more in the :ref:`User Guide <birch>`.

    .. versionadded:: 0.16

    Parameters
    ----------
    threshold : float, default=0.5
        The radius of the subcluster obtained by merging a new sample and the
        closest subcluster should be lesser than the threshold. Otherwise a new
        subcluster is started. Setting this value to be very low promotes
        splitting and vice-versa.

    branching_factor : int, default=50
        Maximum number of CF subclusters in each node. If a new samples enters
        such that the number of subclusters exceed the branching_factor then
        that node is split into two nodes with the subclusters redistributed
        in each. The parent subcluster of that node is removed and two new
        subclusters are added as parents of the 2 split nodes.

    n_clusters : int, instance of sklearn.cluster model or None, default=3
        Number of clusters after the final clustering step, which treats the
        subclusters from the leaves as new samples.

        - `None` : the final clustering step is not performed and the
          subclusters are returned as they are.

        - :mod:`sklearn.cluster` Estimator : If a model is provided, the model
          is fit treating the subclusters as new samples and the initial data
          is mapped to the label of the closest subcluster.

        - `int` : the model fit is :class:`AgglomerativeClustering` with
          `n_clusters` set to be equal to the int.

    compute_labels : bool, default=True
        Whether or not to compute labels for each fit.

    Attributes
    ----------
    root_ : _CFNode
        Root of the CFTree.

    dummy_leaf_ : _CFNode
        Start pointer to all the leaves.

    subcluster_centers_ : ndarray
        Centroids of all subclusters read directly from the leaves.

    subcluster_labels_ : ndarray
        Labels assigned to the centroids of the subclusters after
        they are clustered globally.

    labels_ : ndarray of shape (n_samples,)
        Array of labels assigned to the input data.
        if partial_fit is used instead of fit, they are assigned to the
        last batch of data.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    MiniBatchKMeans : Alternative implementation that does incremental updates
        of the centers' positions using mini-batches.

    Notes
    -----
    The tree data structure consists of nodes with each node consisting of
    a number of subclusters. The maximum number of subclusters in a node
    is determined by the branching factor. Each subcluster maintains a
    linear sum, squared sum and the number of samples in that subcluster.
    In addition, each subcluster can also have a node as its child, if the
    subcluster is not a member of a leaf node.

    For a new point entering the root, it is merged with the subcluster closest
    to it and the linear sum, squared sum and the number of samples of that
    subcluster are updated. This is done recursively till the properties of
    the leaf node are updated.

    See :ref:`sphx_glr_auto_examples_cluster_plot_birch_vs_minibatchkmeans.py` for a
    comparison with :class:`~sklearn.cluster.MiniBatchKMeans`.

    References
    ----------
    * Tian Zhang, Raghu Ramakrishnan, Maron Livny
      BIRCH: An efficient data clustering method for large databases.
      https://www.cs.sfu.ca/CourseCentral/459/han/papers/zhang96.pdf

    * Roberto Perdisci
      JBirch - Java implementation of BIRCH clustering algorithm
      https://code.google.com/archive/p/jbirch

    Examples
    --------
    >>> from sklearn.cluster import Birch
    >>> X = [[0, 1], [0.3, 1], [-0.3, 1], [0, -1], [0.3, -1], [-0.3, -1]]
    >>> brc = Birch(n_clusters=None)
    >>> brc.fit(X)
    Birch(n_clusters=None)
    >>> brc.predict(X)
    array([0, 0, 0, 1, 1, 1])

    For a comparison of the BIRCH clustering algorithm with other clustering algorithms,
    see :ref:`sphx_glr_auto_examples_cluster_plot_cluster_comparison.py`
    """
    _parameter_constraints: dict = {'threshold': [Interval(Real, 0.0, None, closed='neither')], 'branching_factor': [Interval(Integral, 1, None, closed='neither')], 'n_clusters': [None, ClusterMixin, Interval(Integral, 1, None, closed='left')], 'compute_labels': ['boolean']}

    def __init__(self, *, threshold=0.5, branching_factor=50, n_clusters=3, compute_labels=True):
        self.threshold = threshold
        self.branching_factor = branching_factor
        self.n_clusters = n_clusters
        self.compute_labels = compute_labels

    def _fit(self, X, partial):
        has_root = getattr(self, 'root_', None)
        first_call = not (partial and has_root)
        X = validate_data(self, X, accept_sparse='csr', reset=first_call, dtype=[np.float64, np.float32])
        threshold = self.threshold
        branching_factor = self.branching_factor
        n_samples, n_features = X.shape
        if first_call:
            self.root_ = _CFNode(threshold=threshold, branching_factor=branching_factor, is_leaf=True, n_features=n_features, dtype=X.dtype)
            self.dummy_leaf_ = _CFNode(threshold=threshold, branching_factor=branching_factor, is_leaf=True, n_features=n_features, dtype=X.dtype)
            self.dummy_leaf_.next_leaf_ = self.root_
            self.root_.prev_leaf_ = self.dummy_leaf_
        if not sparse.issparse(X):
            iter_func = iter
        else:
            iter_func = _iterate_sparse_X
        for sample in iter_func(X):
            subcluster = _CFSubcluster(linear_sum=sample)
            split = self.root_.insert_cf_subcluster(subcluster)
            if split:
                new_subcluster1, new_subcluster2 = _split_node(self.root_, threshold, branching_factor)
                del self.root_
                self.root_ = _CFNode(threshold=threshold, branching_factor=branching_factor, is_leaf=False, n_features=n_features, dtype=X.dtype)
                self.root_.append_subcluster(new_subcluster1)
                self.root_.append_subcluster(new_subcluster2)
        centroids = np.concatenate([leaf.centroids_ for leaf in self._get_leaves()])
        self.subcluster_centers_ = centroids
        self._n_features_out = self.subcluster_centers_.shape[0]
        self._global_clustering(X)
        return self

    def _get_leaves(self):
        """
        Retrieve the leaves of the CF Node.

        Returns
        -------
        leaves : list of shape (n_leaves,)
            List of the leaf nodes.
        """
        leaf_ptr = self.dummy_leaf_.next_leaf_
        leaves = []
        while leaf_ptr is not None:
            leaves.append(leaf_ptr)
            leaf_ptr = leaf_ptr.next_leaf_
        return leaves

    def _predict(self, X):
        """Predict data using the ``centroids_`` of subclusters."""
        kwargs = {'Y_norm_squared': self._subcluster_norms}
        with config_context(assume_finite=True):
            argmin = pairwise_distances_argmin(X, self.subcluster_centers_, metric_kwargs=kwargs)
        return self.subcluster_labels_[argmin]

    def _global_clustering(self, X=None):
        """
        Global clustering for the subclusters obtained after fitting
        """
        clusterer = self.n_clusters
        centroids = self.subcluster_centers_
        compute_labels = X is not None and self.compute_labels
        not_enough_centroids = False
        if isinstance(clusterer, Integral):
            clusterer = AgglomerativeClustering(n_clusters=self.n_clusters)
            if len(centroids) < self.n_clusters:
                not_enough_centroids = True
        self._subcluster_norms = row_norms(self.subcluster_centers_, squared=True)
        if clusterer is None or not_enough_centroids:
            self.subcluster_labels_ = np.arange(len(centroids))
            if not_enough_centroids:
                warnings.warn('Number of subclusters found (%d) by BIRCH is less than (%d). Decrease the threshold.' % (len(centroids), self.n_clusters), ConvergenceWarning)
        else:
            self.subcluster_labels_ = clusterer.fit_predict(self.subcluster_centers_)
        if compute_labels:
            self.labels_ = self._predict(X)
