import numpy as np

class BiclusterMixin:
    """Mixin class for all bicluster estimators in scikit-learn.

    This mixin defines the following functionality:

    - `biclusters_` property that returns the row and column indicators;
    - `get_indices` method that returns the row and column indices of a bicluster;
    - `get_shape` method that returns the shape of a bicluster;
    - `get_submatrix` method that returns the submatrix corresponding to a bicluster.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.base import BaseEstimator, BiclusterMixin
    >>> class DummyBiClustering(BiclusterMixin, BaseEstimator):
    ...     def fit(self, X, y=None):
    ...         self.rows_ = np.ones(shape=(1, X.shape[0]), dtype=bool)
    ...         self.columns_ = np.ones(shape=(1, X.shape[1]), dtype=bool)
    ...         return self
    >>> X = np.array([[1, 1], [2, 1], [1, 0],
    ...               [4, 7], [3, 5], [3, 6]])
    >>> bicluster = DummyBiClustering().fit(X)
    >>> hasattr(bicluster, "biclusters_")
    True
    >>> bicluster.get_indices(0)
    (array([0, 1, 2, 3, 4, 5]), array([0, 1]))
    """

    def get_indices(self, i):
        """Row and column indices of the `i`'th bicluster.

        Only works if ``rows_`` and ``columns_`` attributes exist.

        Parameters
        ----------
        i : int
            The index of the cluster.

        Returns
        -------
        row_ind : ndarray, dtype=np.intp
            Indices of rows in the dataset that belong to the bicluster.
        col_ind : ndarray, dtype=np.intp
            Indices of columns in the dataset that belong to the bicluster.
        """
        rows = self.rows_[i]
        columns = self.columns_[i]
        return (np.nonzero(rows)[0], np.nonzero(columns)[0])

    def get_shape(self, i):
        """Shape of the `i`'th bicluster.

        Parameters
        ----------
        i : int
            The index of the cluster.

        Returns
        -------
        n_rows : int
            Number of rows in the bicluster.

        n_cols : int
            Number of columns in the bicluster.
        """
        indices = self.get_indices(i)
        return tuple((len(i) for i in indices))
