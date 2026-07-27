from numbers import Integral
import numpy as np
from scipy.linalg import norm
from sklearn.utils._param_validation import Interval, StrOptions

class SpectralBiclustering(BaseSpectral):
    """Spectral biclustering (Kluger, 2003) [1]_.

    Partitions rows and columns under the assumption that the data has
    an underlying checkerboard structure. For instance, if there are
    two row partitions and three column partitions, each row will
    belong to three biclusters, and each column will belong to two
    biclusters. The outer product of the corresponding row and column
    label vectors gives this checkerboard structure.

    Read more in the :ref:`User Guide <spectral_biclustering>`.

    Parameters
    ----------
    n_clusters : int or tuple (n_row_clusters, n_column_clusters), default=3
        The number of row and column clusters in the checkerboard
        structure.

    method : {'bistochastic', 'scale', 'log'}, default='bistochastic'
        Method of normalizing and converting singular vectors into
        biclusters. May be one of 'scale', 'bistochastic', or 'log'.
        The authors recommend using 'log'. If the data is sparse,
        however, log normalization will not work, which is why the
        default is 'bistochastic'.

        .. warning::
           if `method='log'`, the data must not be sparse.

    n_components : int, default=6
        Number of singular vectors to check.

    n_best : int, default=3
        Number of best singular vectors to which to project the data
        for clustering.

    svd_method : {'randomized', 'arpack'}, default='randomized'
        Selects the algorithm for finding singular vectors. May be
        'randomized' or 'arpack'. If 'randomized', uses
        :func:`~sklearn.utils.extmath.randomized_svd`, which may be faster
        for large matrices. If 'arpack', uses
        `scipy.sparse.linalg.svds`, which is more accurate, but
        possibly slower in some cases.

    n_svd_vecs : int, default=None
        Number of vectors to use in calculating the SVD. Corresponds
        to `ncv` when `svd_method=arpack` and `n_oversamples` when
        `svd_method` is 'randomized`.

    mini_batch : bool, default=False
        Whether to use mini-batch k-means, which is faster but may get
        different results.

    init : {'k-means++', 'random'} or ndarray of shape (n_clusters, n_features),             default='k-means++'
        Method for initialization of k-means algorithm; defaults to
        'k-means++'.

    n_init : int, default=10
        Number of random initializations that are tried with the
        k-means algorithm.

        If mini-batch k-means is used, the best initialization is
        chosen and the algorithm runs once. Otherwise, the algorithm
        is run for each initialization and the best solution chosen.

    random_state : int, RandomState instance, default=None
        Used for randomizing the singular value decomposition and the k-means
        initialization. Use an int to make the randomness deterministic.
        See :term:`Glossary <random_state>`.

    Attributes
    ----------
    rows_ : array-like of shape (n_row_clusters, n_rows)
        Results of the clustering. `rows[i, r]` is True if
        cluster `i` contains row `r`. Available only after calling ``fit``.

    columns_ : array-like of shape (n_column_clusters, n_columns)
        Results of the clustering, like `rows`.

    row_labels_ : array-like of shape (n_rows,)
        Row partition labels.

    column_labels_ : array-like of shape (n_cols,)
        Column partition labels.

    biclusters_ : tuple of two ndarrays
        The tuple contains the `rows_` and `columns_` arrays.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    SpectralCoclustering : Clusters rows and columns of an array `X` to solve the
        relaxed normalized cut of the bipartite graph created from `X`.

    References
    ----------

    .. [1] :doi:`Kluger, Yuval, et. al., 2003. Spectral biclustering of microarray
           data: coclustering genes and conditions.
           <10.1101/gr.648603>`

    Examples
    --------
    >>> from sklearn.cluster import SpectralBiclustering
    >>> import numpy as np
    >>> X = np.array([[1, 1], [2, 1], [1, 0],
    ...               [4, 7], [3, 5], [3, 6]])
    >>> clustering = SpectralBiclustering(n_clusters=2, random_state=0).fit(X)
    >>> clustering.row_labels_
    array([1, 1, 1, 0, 0, 0], dtype=int32)
    >>> clustering.column_labels_
    array([1, 0], dtype=int32)
    >>> clustering
    SpectralBiclustering(n_clusters=2, random_state=0)

    For a more detailed example, see
    :ref:`sphx_glr_auto_examples_bicluster_plot_spectral_biclustering.py`
    """
    _parameter_constraints: dict = {**BaseSpectral._parameter_constraints, 'n_clusters': [Interval(Integral, 1, None, closed='left'), tuple], 'method': [StrOptions({'bistochastic', 'scale', 'log'})], 'n_components': [Interval(Integral, 1, None, closed='left')], 'n_best': [Interval(Integral, 1, None, closed='left')]}

    def __init__(self, n_clusters=3, *, method='bistochastic', n_components=6, n_best=3, svd_method='randomized', n_svd_vecs=None, mini_batch=False, init='k-means++', n_init=10, random_state=None):
        super().__init__(n_clusters, svd_method, n_svd_vecs, mini_batch, init, n_init, random_state)
        self.method = method
        self.n_components = n_components
        self.n_best = n_best

    def _fit_best_piecewise(self, vectors, n_best, n_clusters):
        """Find the ``n_best`` vectors that are best approximated by piecewise
        constant vectors.

        The piecewise vectors are found by k-means; the best is chosen
        according to Euclidean distance.

        """

        def make_piecewise(v):
            centroid, labels = self._k_means(v.reshape(-1, 1), n_clusters)
            return centroid[labels].ravel()
        piecewise_vectors = np.apply_along_axis(make_piecewise, axis=1, arr=vectors)
        dists = np.apply_along_axis(norm, axis=1, arr=vectors - piecewise_vectors)
        result = vectors[np.argsort(dists)[:n_best]]
        return result
