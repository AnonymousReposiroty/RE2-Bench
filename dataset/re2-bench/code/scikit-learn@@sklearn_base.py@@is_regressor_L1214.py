from sklearn.utils._tags import (
    ClassifierTags,
    RegressorTags,
    Tags,
    TargetTags,
    TransformerTags,
    get_tags,
)

def is_regressor(estimator):
    """Return True if the given estimator is (probably) a regressor.

    Parameters
    ----------
    estimator : estimator instance
        Estimator object to test.

    Returns
    -------
    out : bool
        True if estimator is a regressor and False otherwise.

    Examples
    --------
    >>> from sklearn.base import is_regressor
    >>> from sklearn.cluster import KMeans
    >>> from sklearn.svm import SVC, SVR
    >>> classifier = SVC()
    >>> regressor = SVR()
    >>> kmeans = KMeans()
    >>> is_regressor(classifier)
    False
    >>> is_regressor(regressor)
    True
    >>> is_regressor(kmeans)
    False
    """
    return get_tags(estimator).estimator_type == "regressor"
