from sklearn.utils._tags import (
    ClassifierTags,
    RegressorTags,
    Tags,
    TargetTags,
    TransformerTags,
    get_tags,
)

def is_outlier_detector(estimator):
    """Return True if the given estimator is (probably) an outlier detector.

    Parameters
    ----------
    estimator : estimator instance
        Estimator object to test.

    Returns
    -------
    out : bool
        True if estimator is an outlier detector and False otherwise.
    """
    return get_tags(estimator).estimator_type == "outlier_detector"
