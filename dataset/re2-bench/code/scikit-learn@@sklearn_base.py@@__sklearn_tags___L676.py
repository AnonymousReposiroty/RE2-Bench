class ClusterMixin:
    """Mixin class for all cluster estimators in scikit-learn.

    - set estimator type to `"clusterer"` through the `estimator_type` tag;
    - `fit_predict` method returning the cluster labels associated to each sample.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.base import BaseEstimator, ClusterMixin
    >>> class MyClusterer(ClusterMixin, BaseEstimator):
    ...     def fit(self, X, y=None):
    ...         self.labels_ = np.ones(shape=(len(X),), dtype=np.int64)
    ...         return self
    >>> X = [[1, 2], [2, 3], [3, 4]]
    >>> MyClusterer().fit_predict(X)
    array([1, 1, 1])
    """

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type = 'clusterer'
        if tags.transformer_tags is not None:
            tags.transformer_tags.preserves_dtype = []
        return tags
