from sklearn.utils._tags import (
    ClassifierTags,
    RegressorTags,
    Tags,
    TargetTags,
    TransformerTags,
    get_tags,
)

def is_clusterer(estimator):
    """Return True if the given estimator is (probably) a clusterer.

    .. versionadded:: 1.6

    Parameters
    ----------
    estimator : estimator instance
        Estimator object to test.

    Returns
    -------
    out : bool
        True if estimator is a clusterer and False otherwise.

    Examples
    --------
    >>> from sklearn.base import is_clusterer
    >>> from sklearn.cluster import KMeans
    >>> from sklearn.svm import SVC, SVR
    >>> classifier = SVC()
    >>> regressor = SVR()
    >>> kmeans = KMeans()
    >>> is_clusterer(classifier)
    False
    >>> is_clusterer(regressor)
    False
    >>> is_clusterer(kmeans)
    True
    """
    return get_tags(estimator).estimator_type == "clusterer"
